import logging
import random
import time
import os
import threading
import numpy as np
import json
from typing import Dict, List, Set
from threading import Lock
from flask import Flask, request
from flask_socketio import SocketIO, emit

from utils import object_to_pickle_string, pickle_string_to_object
from aggregator import Aggregator


def load_json(filename: str) -> Dict:
    """Loads and returns the contents of a JSON file."""
    with open(filename) as f:
        return json.load(f)


class FederatedServer:
    """
    Implements the server-side logic for federated learning orchestration.

    Manages client connections, initiates training rounds, aggregates model
    updates (plaintext and homomorphically encrypted), and evaluates the
    global model.

    Key parameters (in config):
      - models_percentage : fraction of clients selected per training round.
    """

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.logger: logging.Logger = self._setup_logger()
        self.logger.info("Server configuration: %s", self.config)
        self.lock = Lock()

        # Federated learning state
        self.min_num_workers: int = self.config['MIN_NUM_WORKERS']
        self.num_clients_per_round: int = 0
        self.registered_clients: Set[str] = set()
        self.current_round: int = -1
        self.is_training_finished: bool = False
        self.initialization_done: bool = False  # guard against duplicate _on_client_ready triggers

        # Per-round buffers
        self.client_updates_this_round: List[Dict] = []
        self.client_evaluations_this_round: List[Dict] = []
        self.client_metrics_buffer: List[Dict] = []

        self.aggregator = Aggregator(self.config, self.logger)
        self.client_stats_buffer: List[Dict] = []
        self.static_calibration_term: np.ndarray = None

        # Encryption
        self.encryption_mode: str = self.config.get('encryption_mode', 'no_encryption')
        self.ckks_context = None
        if self.encryption_mode == 'ckks':
            self.logger.info("CKKS mode: fetching public context from TA...")
            self._fetch_ckks_context()

        self.logger.info(
            "Clients per round: %d/%d (models_percentage=%.2f)",
            self.min_num_workers, self.config['num_clients'],
            self.config.get('models_percentage', 1.0)
        )

        # Flask & SocketIO setup
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            ping_timeout=3600,
            ping_interval=5,
            max_http_buffer_size=int(1e32)
        )
        self._register_routes_and_handlers()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"FederatedServer-W{self.config.get('worker_id', 'N/A')}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        log_dir = os.path.join(self.config.get('log_dir', 'logs'), datestr, "FL-Server-LOG")
        os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(os.path.join(log_dir, f'{timestr}.log'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _fetch_ckks_context(self) -> None:
        """Connect to the TA and retrieve the public CKKS context (no secret key)."""
        import tenseal as ts
        import socketio as sio_module
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        ta_sio = sio_module.Client(reconnection=False)
        received = threading.Event()

        @ta_sio.on('distribute_keys')
        def on_keys(data):
            ctx_bytes = pickle_string_to_object(data['ckks_context'])
            self.ckks_context = ts.context_from(ctx_bytes)
            received.set()
            ta_sio.disconnect()

        ta_sio.connect(ta_address, transports=['websocket'])
        ta_sio.emit('request_keys', {'role': 'server'})
        if not received.wait(timeout=30):
            self.logger.error("Star server: CKKS public context not received from TA!")

    def run(self) -> None:
        """Binds and serves, retrying on the next port if config['port'] is taken.

        The node may be shared with other users' processes, so the configured
        port is only a preferred starting point — self.config['port'] is
        updated in place on every attempt so callers polling it (e.g. for a
        readiness check) see the port actually in use.
        """
        from utils import run_with_port_retry

        def _on_attempt(port: int) -> None:
            self.config['port'] = port
            self.logger.info("Federated Server starting on %s:%d", self.config['ip_address'], port)

        def _bind_and_serve(port: int) -> None:
            self.socketio.run(self.app, host=self.config['ip_address'], port=port)

        run_with_port_retry(_bind_and_serve, self.config['port'],
                            on_attempt=_on_attempt, logger=self.logger)

    def _register_routes_and_handlers(self) -> None:
        """Registers Flask routes and all SocketIO event handlers."""
        self.app.route('/')(self._health_check)

        self.socketio.on('connect')(self._on_connect)
        self.socketio.on('disconnect')(self._on_disconnect)
        self.socketio.on('reconnect')(self._on_reconnect)
        self.socketio.on('client_wake_up')(self._on_client_wake_up)
        self.socketio.on('client_ready')(self._on_client_ready)
        self.socketio.on('client_update')(self._on_client_update)
        self.socketio.on('client_eval')(self._on_client_eval)

    def _health_check(self):
        """HTTP endpoint used by grid search runner to verify the server is up."""
        run_id = self.config.get('server_run_id', 'Server is running')
        return run_id, 200

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------

    def _start_next_training_round(self) -> None:
        """
        Increments round counter, selects clients using min_num_workers,
        and broadcasts the update request to selected clients only.
        """
        self.current_round += 1
        self.logger.info("--- Starting Round %d ---", self.current_round)
        print(f"[Server] Round {self.current_round}/{self.config['global_epoch']} starting "
              f"({self.min_num_workers}/{len(self.registered_clients)} clients selected)...")

        self.client_updates_this_round.clear()
        self.client_metrics_buffer.clear()

        all_available_clients = list(self.registered_clients)
        selected_clients = random.sample(all_available_clients, self.min_num_workers)
        self.num_clients_per_round = len(selected_clients)

        self.logger.info(
            "Requesting updates from %d/%d clients.",
            self.min_num_workers, len(all_available_clients)
        )

        request_data = {
            'epochs': self.config['local_epoch'],
            'batch_size': self.config['batch_size'],
            'learning_rate': self.config['learning_rate'],
            'round_number': self.current_round,
            # Plaintext w^(0) at t = 0, then the aggregated global weights
            # (encrypted under CKKS, plaintext in the baseline).
            'current_weights': object_to_pickle_string(self.aggregator.current_weights),
        }
        for sid in selected_clients:
            emit('request_update', request_data, room=sid)

    def _trigger_global_evaluation(self) -> None:
        """Broadcasts the aggregated model to all clients for evaluation."""
        self.logger.info("Triggering global evaluation on %d clients.", len(self.registered_clients))
        self.client_evaluations_this_round.clear()

        data_to_send = {
            'batch_size': self.config['batch_size'],
            'current_weights': object_to_pickle_string(self.aggregator.current_weights),
            'STOP': self.is_training_finished,
            'round_number': self.current_round
        }
        for sid in self.registered_clients:
            emit('stop_and_eval', data_to_send, room=sid)

    def _aggregate_updates(self) -> None:
        """Aggregates client weight updates (plaintext or encrypted)."""
        train_losses = [x['train_loss'] for x in self.client_updates_this_round]
        train_sizes = [x['train_size'] for x in self.client_updates_this_round]

        if self.encryption_mode == 'ckks':
            self.aggregator.aggregate_encrypted_updates_ckks(
                self.client_updates_this_round, self.ckks_context
            )
        else:
            self.aggregator.aggregate_weights(
                self.client_updates_this_round,
                self.config['aggregation_algorithm']
            )

        self.aggregator.aggregate_train_loss_weighted(train_losses, train_sizes, self.current_round)

    def _shutdown_server(self) -> None:
        """Sends shutdown to clients and stops the SocketIO server."""
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        emit('shutdown', {'ta_address': ta_address})
        time.sleep(1)
        self.logger.info("Shutting down the server.")
        self.socketio.stop()

    # ------------------------------------------------------------------
    # SocketIO event handlers
    # ------------------------------------------------------------------

    def _on_connect(self):
        self.logger.info("Client connected: %s", request.sid)

    def _on_disconnect(self):
        self.logger.info("Client disconnected: %s", request.sid)
        if request.sid in self.registered_clients:
            self.registered_clients.remove(request.sid)

    def _on_reconnect(self):
        self.logger.info("Client reconnected: %s", request.sid)

    def _on_client_wake_up(self):
        """Responds to a waking client with the Trusted Authority address."""
        self.logger.info("Client %s waking up. Sending 'init'.", request.sid)
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        emit('init', {'ta_address': ta_address})

    def _on_client_ready(self, data: Dict):
        """
        Registers a client as ready. Once all expected clients are ready,
        triggers initialization and the first (or second) training run.
        """
        self.logger.info("Client %s is ready.", request.sid)

        # Reject clients whose model doesn't match this server's expected model.
        client_model = data.get('model_name', '')
        expected_model = self.config.get('model_name', '')
        if client_model and expected_model and client_model != expected_model:
            self.logger.warning(
                "Rejecting client %s: model mismatch — got '%s', expected '%s'.",
                request.sid, client_model, expected_model
            )
            emit('shutdown', {})
            return

        with self.lock:
            self.registered_clients.add(request.sid)

            samples_per_class = pickle_string_to_object(data['samples_per_class'])
            self.client_stats_buffer.append(samples_per_class)

            # Guard: only start initialization once, when all expected clients are ready.
            should_start = (
                len(self.client_stats_buffer) >= self.config['num_clients']
                and not self.initialization_done
                and self.current_round == -1
            )
            if should_start:
                self.initialization_done = True

        if should_start:
            self.logger.info("All clients ready. Starting initialization...")
            self._finalize_initialization_and_start_training()

    def _finalize_initialization_and_start_training(self) -> None:
        """Computes FedLC calibration term (if needed) and starts the first training round."""
        if self.config.get("aggregation_algorithm") == "FedLC":
            self.logger.info("Computing static calibration term for FedLC...")
            total_samples_per_class = np.sum(self.client_stats_buffer, axis=0)
            total_samples_per_class[total_samples_per_class == 0] = 1
            calibration_val = total_samples_per_class ** (-1 / 4)
            self.static_calibration_term = calibration_val

            data_to_send = {'calibration_val': object_to_pickle_string(self.static_calibration_term)}
            for sid in self.registered_clients:
                emit('distribute_calibration', data_to_send, room=sid)

        self.logger.info("Initialization complete. Starting federated training.")
        time.sleep(10)
        self._start_next_training_round()

    def _on_client_update(self, data: Dict):
        """
        Receives a local model update. When all round updates arrive,
        triggers aggregation and global evaluation.
        """
        with self.lock:
            self.logger.info("Received update from %s for round %d.",
                             request.sid, data['round_number'])

            if data['round_number'] != self.current_round:
                self.logger.warning(
                    "Stale update: round %d received, current is %d. Ignoring.",
                    data['round_number'], self.current_round
                )
                return

            try:
                data['weights'] = pickle_string_to_object(data['weights'])
            except Exception as e:
                self.logger.error("Error unpickling weights from %s: %s", request.sid, e)
                return

            self.client_updates_this_round.append(data)

            if len(self.client_updates_this_round) >= self.num_clients_per_round:
                self.logger.info(
                    "All %d updates for round %d received. Aggregating...",
                    self.num_clients_per_round, self.current_round
                )
                self._aggregate_updates()

                if self.current_round >= self.config['global_epoch'] - 1:
                    self.logger.info("Maximum global rounds reached. Finishing.")
                    self.is_training_finished = True

                self._trigger_global_evaluation()

    def _on_client_eval(self, data: Dict):
        """
        Receives an evaluation result. When all clients have reported,
        aggregates metrics then either starts the next round or shuts down.
        """
        with self.lock:
            self.logger.info("Received evaluation from client %s.", request.sid)
            self.client_evaluations_this_round.append(data)

            if len(self.client_evaluations_this_round) < len(self.registered_clients):
                return

            self.logger.info("All evaluations received for round %d.", self.current_round)

            is_early_stopping = self.aggregator.aggregate_evaluation_results(
                self.client_evaluations_this_round, self.current_round
            )

            if is_early_stopping:
                self.logger.info("Early stopping condition met.")
                self.is_training_finished = True

            if self.is_training_finished:
                self.logger.info("Training phase complete.")
                self.aggregator.log_best_model_stats()
                self.aggregator.save_results()

                self.logger.info("Federated process fully complete. Shutting down.")
                for sid in self.registered_clients:
                    emit("shutdown", room=sid)
                self._shutdown_server()
            else:
                self.logger.info("Proceeding to next round.")
                self._start_next_training_round()