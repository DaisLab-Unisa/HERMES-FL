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
    """
    Implements the client-side logic for federated learning (star topology).

    Supports plaintext and homomorphically-encrypted weight exchange,
    and FedAvg / FedProx / FedLC aggregation algorithms.
    """

    def __init__(self, config: Dict, dataset_path: str):
        self.local_model: ModelManager = None
        self.config: Dict = config
        self.dataset_path: str = dataset_path
        self.client_id: str = os.path.basename(dataset_path)
        self.logger: logging.LoggerAdapter = self._setup_logger()

        # Encryption
        self.encryption_mode: str = self.config.get('encryption_mode', 'no_encryption')
        if self.encryption_mode == 'ckks':
            self.logger.info("CKKS encryption enabled. Context will be requested from the TA.")
            self.ckks_context = None
            self.keys_received_event = threading.Event()

        self.sio = socketio.Client(logger=True, request_timeout=10, reconnection=True)
        self._register_event_handlers()
        self.connect_to_server()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.LoggerAdapter:
        worker_id = self.config.get('worker_id', 'N/A')
        # Include client_id so each client gets its own logger and log file.
        logger_name = f"FederatedClient-W{worker_id}-{self.client_id}"
        base_logger = logging.getLogger(logger_name)

        if not base_logger.hasHandlers():
            datestr = time.strftime('%d%m')
            timestr = time.strftime('%m%d%H%M')
            log_dir_base = self.config.get('log_dir', 'logs')
            log_dir = os.path.join(log_dir_base, datestr, "FL-Client-LOG")
            os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(os.path.join(log_dir, f'{timestr}_{self.client_id}.log'))
            file_handler.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)

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

    def connect_to_server(self) -> None:
        server_address = "http://" + self.config['ip_address'] + ":" + str(self.config['port'])
        self.logger.info("Connecting to server at %s", server_address)
        try:
            self.sio.connect(server_address, transports=['websocket'])
            self.logger.info("Sending wake up message to the server.")
            self.sio.emit('client_wake_up')
            self.sio.wait()
        except socketio.exceptions.ConnectionError as e:
            self.logger.error("Failed to connect to the server: %s", e)
            exit(1)

    def _register_event_handlers(self) -> None:
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)
        self.sio.on('reconnect', self._on_reconnect)
        self.sio.on('shutdown', self._on_shutdown)
        self.sio.on('init', self._on_init)
        self.sio.on('request_update', self._on_request_update)
        self.sio.on('stop_and_eval', self._on_stop_and_eval)
        self.sio.on('distribute_calibration', self._on_distribute_calibration)

    # ------------------------------------------------------------------
    # Connection events
    # ------------------------------------------------------------------

    def _on_connect(self):
        self.logger.info("Successfully connected with SID: %s", self.sio.sid)

    def _on_disconnect(self):
        self.logger.info("Disconnected from the server.")
        self.sio.disconnect()

    def _on_reconnect(self):
        self.logger.info("Reconnected to the server.")

    def _on_shutdown(self):
        self.logger.info("Received shutdown signal. Exiting.")
        self.sio.disconnect()
        exit(0)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _on_init(self, data: Dict):
        """
        Handles the initialization signal from the server.
        If encryption is enabled, connects to the Trusted Authority first.
        """
        self.logger.info("Received initialization signal from server.")
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
        """Initializes the local ModelManager and notifies the server."""
        self.logger.info("Initializing local model on dataset: %s", self.dataset_path)
        self.local_model = ModelManager(
            config=self.config,
            dataset_path=self.dataset_path
        )
        samples_per_class = self.local_model.get_samples_per_class()
        self.logger.info("Model initialized. Sending 'client_ready'.")
        self.sio.emit('client_ready', {
            'samples_per_class': object_to_pickle_string(samples_per_class),
            'model_name': self.config.get('model_name', ''),
        })

    def _on_distribute_calibration(self, data: Dict):
        self.logger.info("Received static calibration term from the server.")
        calibration_val = pickle_string_to_object(data['calibration_val'])
        self.local_model.set_calibration_term(calibration_val)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_server_weights(self, data: Dict):
        """Returns the global weights received from the server, decrypting them if needed.

        The payload is the plaintext w^(0) at t = 0 (and in every round of the plaintext
        baseline), or the CKKS ciphertext of the weighted average computed by the server,
        which only the clients can decrypt with the secret key (Eq. 6).
        """
        weights_data = pickle_string_to_object(data['current_weights'])

        first_element = weights_data[0]
        if isinstance(first_element, dict) and 'chunks' in first_element:
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

    # ------------------------------------------------------------------
    # Training round
    # ------------------------------------------------------------------

    def _on_request_update(self, data: Dict):
        """Spawns a worker thread to handle the local update without blocking the event loop."""
        self.logger.info("Received update request for round %s.", data['round_number'])
        worker_thread = threading.Thread(target=self._update_worker, args=(data,))
        worker_thread.daemon = True
        worker_thread.start()

    def _update_worker(self, data: Dict):
        """Full federated learning update cycle for a single round."""
        try:
            round_num = data['round_number']
            self.logger.info("Worker thread started for round %s.", round_num)
            print(f"[{self.client_id}] Round {round_num}: training for {data['epochs']} epochs...")

            averaged_weights = self._process_server_weights(data)
            if averaged_weights is None:
                self.logger.error("Worker: failed to obtain valid weights. Aborting round.")
                return

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
                    global_weights=averaged_weights,
                    mu=self.config.get("fedprox_mu", 0.0)
                )
            finally:
                if semaphore:
                    semaphore.release()
            print(f"[{self.client_id}] Round {round_num}: training done — loss={train_loss:.4f}")

            local_weights = self.local_model.get_weights()

            if self.encryption_mode == 'ckks':
                weights_to_send = ckks_encrypt_weights(self.ckks_context, local_weights)
            else:
                weights_to_send = local_weights

            avg_f1 = (np.mean(list(train_map['f1_score']))
                      if isinstance(train_map['f1_score'], list)
                      else train_map['f1_score'])
            avg_acc = (np.mean(list(train_map['accuracy']))
                       if isinstance(train_map['accuracy'], list)
                       else train_map['accuracy'])

            response = {
                'round_number': data['round_number'],
                'train_loss': train_loss,
                'avg_f1': avg_f1,
                'avg_acc': avg_acc,
                'train_size': train_size,
                'weights': object_to_pickle_string(weights_to_send)
            }

            self.sio.emit('client_update', response)
            self.logger.info("Round %s — train_loss=%.4f, f1=%.4f",
                             data['round_number'], train_loss, avg_f1)

        except Exception as e:
            self.logger.error("Error in update worker: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _on_stop_and_eval(self, data: Dict):
        """
        Applies the aggregated model, evaluates it on the local validation and test
        subsets, and reports both results to the server (validation drives model
        selection and early stopping, test is only reported).
        """
        self.logger.info("Received aggregated model for evaluation.")
        final_weights = self._process_server_weights(data)
        if final_weights is None:
            self.logger.error("Evaluation failed: could not process server weights.")
            return

        self.local_model.set_weights(final_weights)
        response = self.local_model.evaluate_global_model(data['batch_size'])
        self.sio.emit('client_eval', response)

        if data.get('STOP', False):
            self.logger.info("Federated training finished. Shutting down client.")
            exit(0)