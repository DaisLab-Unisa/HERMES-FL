"""
Central Server — root node of the hierarchical FL topology.

Topology overview:
  Central Server  ←→  Edge Server 1  ←→  [Client 1, Client 2, ...]
                  ←→  Edge Server 2  ←→  [Client 3, Client 4, ...]

The central server never talks directly to training clients.
It receives aggregated updates from edge servers and distributes the
global model back down to them. Edge servers then relay it to their clients.

Round flow:
  1. Edge servers connect and send 'edge_wake_up'.
  2. Central sends 'init' with TA address (if encryption enabled).
  3. Edge servers initialize and send 'edge_ready' with combined dataset stats.
  4. Central sends 'request_edge_update' to all edge servers.
  5. Each edge server trains its cluster, aggregates locally, and sends
     'edge_update' with the cluster-aggregated weights.
  6. Central aggregates all edge updates → global model.
  7. Central triggers global evaluation via 'stop_and_eval_edge'.
  8. Edge servers evaluate and send 'edge_eval' results.
  9. Repeat from step 4 until max rounds or early stopping.
 10. Central sends 'shutdown' to all edge servers.
"""

import logging
import random
import threading
import time
import os
import numpy as np
import json
from typing import Dict, List, Set
from threading import Lock
from flask import Flask, request
from flask_socketio import SocketIO, emit

from utils import object_to_pickle_string, pickle_string_to_object
from aggregator import Aggregator


class CentralServer:
    """Root aggregator in the hierarchical FL topology."""

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.logger: logging.Logger = self._setup_logger()
        self.logger.info("Central Server configuration: %s", self.config)
        self.lock = Lock()

        self.num_edge_servers: int = self.config['num_edge_servers']
        self.min_edge_servers_per_round: int = max(
            1, int(self.num_edge_servers * self.config.get('edge_participation', 1.0))
        )
        self.registered_edges: Set[str] = set()
        self._training_started: bool = False
        self.current_round: int = -1
        self.is_training_finished: bool = False
        self.num_edges_per_round: int = 0

        self.edge_updates_this_round: List[Dict] = []
        self.edge_evaluations_this_round: List[Dict] = []
        self.edge_stats_buffer: List[np.ndarray] = []
        self.static_calibration_term: np.ndarray = None

        self.aggregator = Aggregator(self.config, self.logger)

        self.encryption_mode: str = self.config.get('encryption_mode', 'no_encryption')
        self.ckks_context = None
        if self.encryption_mode == 'ckks':
            self._fetch_ckks_context()

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
        logger = logging.getLogger(f"CentralServer-W{self.config.get('worker_id', 'N/A')}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        log_dir = os.path.join(self.config.get('log_dir', 'logs'), datestr, "Central-Server-LOG")
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

    def run(self) -> None:
        """Binds and serves, retrying on the next port if central_port is taken.

        The node may be shared with other users' processes, so the configured
        port is only a preferred starting point — self.config['central_port']
        is updated in place on every attempt so callers polling it (e.g. for
        a readiness check) see the port actually in use.
        """
        from utils import run_with_port_retry

        def _on_attempt(port: int) -> None:
            self.config['central_port'] = port
            self.logger.info("Central Server starting on %s:%d", self.config['ip_address'], port)

        def _bind_and_serve(port: int) -> None:
            self.socketio.run(self.app, host=self.config['ip_address'], port=port)

        run_with_port_retry(_bind_and_serve, self.config['central_port'],
                            on_attempt=_on_attempt, logger=self.logger)

    def _register_routes_and_handlers(self) -> None:
        self.app.route('/')(self._health_check)
        self.socketio.on('connect')(self._on_connect)
        self.socketio.on('disconnect')(self._on_disconnect)
        self.socketio.on('edge_wake_up')(self._on_edge_wake_up)
        self.socketio.on('edge_ready')(self._on_edge_ready)
        self.socketio.on('edge_update')(self._on_edge_update)
        self.socketio.on('edge_eval')(self._on_edge_eval)

    def _health_check(self):
        return self.config.get('server_run_id', 'Central Server is running'), 200

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------

    def _start_next_round(self) -> None:
        self.current_round += 1
        self.logger.info("--- Central: Starting Round %d ---", self.current_round)

        self.edge_updates_this_round.clear()

        all_edges = list(self.registered_edges)
        selected_edges = random.sample(all_edges, self.min_edge_servers_per_round)
        self.num_edges_per_round = len(selected_edges)

        self.logger.info("Central: Requesting updates from %d/%d edge servers.",
                         self.num_edges_per_round, len(all_edges))

        request_data = {
            'epochs': self.config['local_epoch'],
            'batch_size': self.config['batch_size'],
            'learning_rate': self.config['learning_rate'],
            'round_number': self.current_round,
            # Plaintext w^(0) at t = 0, then the aggregated global weights
            # (encrypted under CKKS, plaintext in the baseline).
            'current_weights': object_to_pickle_string(self.aggregator.current_weights),
            'edge_rounds': self.config.get('edge_rounds', 1),
        }
        for sid in selected_edges:
            emit('request_edge_update', request_data, room=sid)

    def _trigger_global_evaluation(self) -> None:
        self.logger.info("Central: Triggering global evaluation on %d edge servers.",
                         len(self.registered_edges))
        self.edge_evaluations_this_round.clear()

        data_to_send = {
            'batch_size': self.config['batch_size'],
            'current_weights': object_to_pickle_string(self.aggregator.current_weights),
            'STOP': self.is_training_finished,
            'round_number': self.current_round
        }
        for sid in self.registered_edges:
            emit('stop_and_eval_edge', data_to_send, room=sid)

    def _fetch_ckks_context(self) -> None:
        """Connect to the TA and retrieve the public CKKS context (no secret key)."""
        import socketio as sio_module
        import tenseal as ts

        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        ta_sio = sio_module.Client(reconnection=False)
        received_event = threading.Event()

        @ta_sio.on('distribute_keys')
        def on_keys(data):
            ctx_bytes = pickle_string_to_object(data['ckks_context'])
            self.ckks_context = ts.context_from(ctx_bytes)
            received_event.set()
            ta_sio.disconnect()

        ta_sio.connect(ta_address, transports=['websocket'])
        ta_sio.emit('request_keys', {'role': 'server'})
        if not received_event.wait(timeout=30):
            self.logger.error("Central: CKKS public context not received from TA within timeout!")

    def _aggregate_edge_updates(self) -> None:
        """Aggregates edge cluster updates (plaintext or encrypted)."""
        train_losses = [x['train_loss'] for x in self.edge_updates_this_round]
        train_sizes = [x['train_size'] for x in self.edge_updates_this_round]

        self.aggregator.aggregate_train_loss_weighted(train_losses, train_sizes, self.current_round)

        algorithm = self.config['aggregation_algorithm']

        if self.encryption_mode == 'ckks':
            self.aggregator.aggregate_encrypted_updates_ckks(self.edge_updates_this_round, self.ckks_context)
        else:
            self.aggregator.aggregate_weights(self.edge_updates_this_round, algorithm)

    def _finish_round_aggregation(self) -> None:
        """Post-aggregation step, run once the global model for this round is ready."""
        if self.current_round >= self.config['global_epoch'] - 1:
            self.is_training_finished = True
        self._trigger_global_evaluation()

    def _shutdown_server(self) -> None:
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        emit('shutdown', {'ta_address': ta_address})
        time.sleep(1)
        self.logger.info("Central Server shutting down.")
        self.socketio.stop()

    # ------------------------------------------------------------------
    # SocketIO event handlers
    # ------------------------------------------------------------------

    def _on_connect(self):
        self.logger.info("Edge server connected: %s", request.sid)

    def _on_disconnect(self):
        self.logger.info("Edge server disconnected: %s", request.sid)
        self.registered_edges.discard(request.sid)

    def _on_edge_wake_up(self):
        """Edge server woke up — send TA address and init signal."""
        self.logger.info("Edge %s waking up.", request.sid)
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        emit('init', {'ta_address': ta_address})

    def _on_edge_ready(self, data: Dict):
        """Edge server is ready with its cluster initialized."""
        self.logger.info("Edge %s is ready.", request.sid)

        with self.lock:
            self.registered_edges.add(request.sid)

            samples_per_class = pickle_string_to_object(data['samples_per_class'])
            self.edge_stats_buffer.append(samples_per_class)

            should_start = (
                len(self.edge_stats_buffer) >= self.num_edge_servers
                and not self._training_started
                and self.current_round == -1
            )
            if should_start:
                self._training_started = True

        if should_start:
            self.logger.info("All %d edge servers ready. Starting training.", self.num_edge_servers)
            self._finalize_initialization_and_start_training()

    def _finalize_initialization_and_start_training(self) -> None:
        if self.config.get("aggregation_algorithm") == "FedLC":
            total_samples_per_class = np.sum(self.edge_stats_buffer, axis=0)
            total_samples_per_class[total_samples_per_class == 0] = 1
            calibration_val = total_samples_per_class ** (-1 / 4)
            self.static_calibration_term = calibration_val
            data_to_send = {'calibration_val': object_to_pickle_string(self.static_calibration_term)}
            for sid in self.registered_edges:
                emit('distribute_calibration', data_to_send, room=sid)

        self.logger.info("Central: initialization complete. Starting federated training.")
        time.sleep(10)
        self._start_next_round()

    def _on_edge_update(self, data: Dict):
        """Receive aggregated update from one edge server."""
        with self.lock:
            self.logger.info("Central: received edge update from %s for round %d.",
                             request.sid, data['round_number'])

            if data['round_number'] != self.current_round:
                self.logger.warning("Central: stale update (round %d vs current %d). Ignoring.",
                                    data['round_number'], self.current_round)
                return

            try:
                data['weights'] = pickle_string_to_object(data['weights'])
            except Exception as e:
                self.logger.error("Central: error unpickling edge weights: %s", e)
                return

            data['_sid'] = request.sid
            self.edge_updates_this_round.append(data)

            if len(self.edge_updates_this_round) >= self.num_edges_per_round:
                self.logger.info("Central: all %d edge updates received. Aggregating globally...",
                                 self.num_edges_per_round)
                self._aggregate_edge_updates()
                self._finish_round_aggregation()

    def _on_edge_eval(self, data: Dict):
        """Receive evaluation result from one edge server."""
        with self.lock:
            self.edge_evaluations_this_round.append(data)

            if len(self.edge_evaluations_this_round) < len(self.registered_edges):
                return

            self.logger.info("Central: all evaluations received for round %d.", self.current_round)

            is_early_stopping = self.aggregator.aggregate_evaluation_results(
                self.edge_evaluations_this_round, self.current_round
            )
            if is_early_stopping:
                self.is_training_finished = True

            if self.is_training_finished:
                self.aggregator.log_best_model_stats()
                self.aggregator.save_results()

                self.logger.info("Central: process complete. Shutting down.")
                for sid in self.registered_edges:
                    emit("shutdown", room=sid)
                self._shutdown_server()
            else:
                self._start_next_round()
