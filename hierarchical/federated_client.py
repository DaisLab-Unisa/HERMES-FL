"""
Federated Client — hierarchical topology.

Identical to the star-topology client except that it connects to an
EdgeServer (on edge_port) instead of the central server directly.
The edge server relays all SocketIO events transparently so this file
requires no protocol-level changes.
"""

import logging
import time
import os
import socketio
import numpy as np
from typing import Dict
import threading

from model_manager import ModelManager
from utils import (object_to_pickle_string, pickle_string_to_object,
                   ckks_encrypt_weights, ckks_decrypt_weights)


class ContextFilter(logging.Filter):
    """A logging filter that injects client_id into log records."""

    def __init__(self, client_id: str):
        super().__init__()
        self.client_id = client_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.client_id = self.client_id
        return True


class FederatedClient:
    """Client that connects to an EdgeServer in the hierarchical topology."""

    def __init__(self, config: Dict, dataset_path: str):
        self.local_model: ModelManager = None
        self.config: Dict = config
        self.dataset_path: str = dataset_path
        self.client_id: str = os.path.basename(dataset_path)
        self.logger: logging.LoggerAdapter = self._setup_logger()

        self.encryption_mode: str = self.config.get('encryption_mode', 'no_encryption')
        if self.encryption_mode == 'ckks':
            self.logger.info("CKKS encryption enabled. Context will be requested from the TA.")
            self.ckks_context = None        # full context (with secret key); set in _on_init
            self.keys_received_event = threading.Event()

        self.sio = socketio.Client(logger=True, request_timeout=10, reconnection=False)
        self._register_event_handlers()
        self.connect_to_edge()

    def _setup_logger(self) -> logging.LoggerAdapter:
        worker_id = self.config.get('worker_id', 'N/A')
        logger_name = f"HierClient-W{worker_id}-{self.client_id}"
        base_logger = logging.getLogger(logger_name)

        if not base_logger.hasHandlers():
            datestr = time.strftime('%d%m')
            timestr = time.strftime('%m%d%H%M')
            log_dir = os.path.join(self.config.get('log_dir', 'logs'), datestr, "FL-Client-LOG")
            os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(
                os.path.join(log_dir, f'{timestr}_{self.client_id}.log'))
            file_handler.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.WARN)

            formatter = logging.Formatter('%(asctime)s - %(client_id)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            base_logger.setLevel(logging.INFO)
            base_logger.addHandler(file_handler)
            base_logger.addHandler(stream_handler)

        adapter = logging.LoggerAdapter(base_logger, {'client_id': self.client_id})
        log_filter = ContextFilter(self.client_id)
        if not any(isinstance(f, ContextFilter) for f in adapter.logger.filters):
            adapter.logger.addFilter(log_filter)
        return adapter

    def connect_to_edge(self) -> None:
        # Connect to the edge server on edge_port (not central_port)
        edge_address = f"http://{self.config['ip_address']}:{self.config['edge_port']}"
        self.logger.info("Connecting to edge server at %s", edge_address)
        try:
            self.sio.connect(edge_address, transports=['websocket'])
            self.sio.emit('client_wake_up')
            self.sio.wait()
        except socketio.exceptions.ConnectionError as e:
            self.logger.error("Failed to connect to edge server: %s", e)
            exit(1)

    def _register_event_handlers(self) -> None:
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)
        self.sio.on('shutdown', self._on_shutdown)
        self.sio.on('init', self._on_init)
        self.sio.on('request_update', self._on_request_update)
        self.sio.on('stop_and_eval', self._on_stop_and_eval)
        self.sio.on('distribute_calibration', self._on_distribute_calibration)

    def _on_connect(self):
        self.logger.info("Connected to edge server with SID: %s", self.sio.sid)

    def _on_disconnect(self):
        self.logger.info("Disconnected from edge server.")
        self.sio.disconnect()

    def _on_shutdown(self):
        self.logger.info("Received shutdown signal. Disconnecting.")
        self.sio.disconnect()

    def _on_init(self, data: Dict):
        self.logger.info("Received 'init' from edge server.")
        if self.encryption_mode == 'ckks':
            ta_address = data['ta_address']
            ta_sio = socketio.Client(reconnection=False)

            @ta_sio.on('distribute_keys')
            def on_receive_ckks(key_data):
                import tenseal as ts
                ctx_bytes = pickle_string_to_object(key_data['ckks_context'])
                self.ckks_context = ts.context_from(ctx_bytes)
                self.keys_received_event.set()
                ta_sio.disconnect()

            ta_sio.connect(ta_address, transports=['websocket'])
            ta_sio.emit('request_keys', {'role': 'client'})
            if not self.keys_received_event.wait(timeout=30):
                self.logger.error("CRITICAL: Did not receive CKKS context from TA.")
                exit(1)

        self._initialize_model_and_report_ready()

    def _initialize_model_and_report_ready(self):
        self.local_model = ModelManager(config=self.config, dataset_path=self.dataset_path)
        samples_per_class = self.local_model.get_samples_per_class()
        self.sio.emit('client_ready', {
            'samples_per_class': object_to_pickle_string(samples_per_class),
            'model_name': self.config.get('model_name', '')
        })

    def _on_distribute_calibration(self, data: Dict):
        calibration_val = pickle_string_to_object(data['calibration_val'])
        self.local_model.set_calibration_term(calibration_val)

    def _process_server_weights(self, data: Dict, key: str = 'current_weights'):
        """Returns the weights carried by data[key], decrypting them if needed.

        The payload is the plaintext w^(0) at t = 0 (and in every round of the plaintext
        baseline), or the CKKS ciphertext of a weighted average computed by the edge or
        central server, which only the clients can decrypt with the secret key (Eq. 6).
        """
        weights_data = pickle_string_to_object(data[key])

        first_element = weights_data[0]
        if isinstance(first_element, dict) and 'chunks' in first_element:
            # CKKS format: List[{'shape': ..., 'chunks': [bytes, ...]}]
            try:
                return ckks_decrypt_weights(self.ckks_context, weights_data)
            except Exception as e:
                self.logger.error("CKKS decryption failed: %s", e, exc_info=True)
                return None
        elif isinstance(first_element, np.ndarray):
            return weights_data
        else:
            self.logger.error("Unknown weights format: %s", type(first_element))
            return None

    def _on_request_update(self, data: Dict):
        worker_thread = threading.Thread(target=self._update_worker, args=(data,))
        worker_thread.daemon = True
        worker_thread.start()

    def _update_worker(self, data: Dict):
        try:
            round_num = data['round_number']
            print(f"[{self.client_id}] Round {round_num}: training for {data['epochs']} epochs...")
            averaged_weights = self._process_server_weights(data)
            if averaged_weights is None:
                return

            # FedProx: the proximal term must reference the CENTRAL global model,
            # not the edge-averaged model which changes each edge round.
            # For edge round 1 the two coincide (global_weights absent → use averaged_weights).
            # For edge rounds 2+ the edge passes 'global_weights' containing the
            # original central model so the reference stays fixed.
            if 'global_weights' in data:
                prox_weights = self._process_server_weights(data, key='global_weights') or averaged_weights
            else:
                prox_weights = averaged_weights

            self.local_model.set_weights(averaged_weights)
            semaphore = self.config.get('training_semaphore')
            if semaphore:
                semaphore.acquire()
            try:
                _, train_map, train_loss, train_size = self.local_model.train(
                    epochs=data['epochs'],
                    lr=data['learning_rate'],
                    batch_size=data['batch_size'],
                    algorithm=self.config.get("aggregation_algorithm", "FedAvg"),
                    global_weights=prox_weights,
                    mu=self.config.get("fedprox_mu", 0.0)
                )
            finally:
                if semaphore:
                    semaphore.release()
            print(f"[{self.client_id}] Round {round_num}: training done — loss={train_loss:.4f}")

            local_weights = self.local_model.get_weights()
            if self.encryption_mode == 'ckks':
                # Encrypt raw weights — edge handles weighting homomorphically.
                weights_to_send = ckks_encrypt_weights(self.ckks_context, local_weights)
            else:
                weights_to_send = local_weights

            avg_f1 = (np.mean(list(train_map['f1_score']))
                      if isinstance(train_map['f1_score'], list) else train_map['f1_score'])
            avg_acc = (np.mean(list(train_map['accuracy']))
                       if isinstance(train_map['accuracy'], list) else train_map['accuracy'])

            self.sio.emit('client_update', {
                'round_number': data['round_number'],
                'train_loss': train_loss,
                'avg_f1': avg_f1,
                'avg_acc': avg_acc,
                'train_size': train_size,
                'weights': object_to_pickle_string(weights_to_send)
            })
        except Exception as e:
            self.logger.error("Error in update worker: %s", e, exc_info=True)

    def _on_stop_and_eval(self, data: Dict):
        final_weights = self._process_server_weights(data)
        if final_weights is None:
            return

        # Global model evaluated on the local validation subset (model selection,
        # early stopping) and on the local test subset (reported metrics).
        self.local_model.set_weights(final_weights)
        self.sio.emit('client_eval', self.local_model.evaluate_global_model(data['batch_size']))

        if data.get('STOP', False):
            self.logger.info("Federated training finished. Disconnecting.")
            # Brief pause so client_eval is sent before the connection closes.
            # exit(0) doesn't work in threads; sio.disconnect() unblocks sio.wait().
            time.sleep(0.5)
            self.sio.disconnect()
