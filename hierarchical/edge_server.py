"""
Edge Server — intermediate aggregator in the hierarchical FL topology.

Dual role:
  - Acts as a SERVER for a cluster of training clients (Flask-SocketIO on edge_port).
  - Acts as a CLIENT to the central server (socketio.Client on central_port).

HierFedAvg round flow (τ₂ = edge_rounds):
  Each global round from the central triggers τ₂ edge rounds at the edge before
  any communication back to the central server.

  1. Connect to central, send 'edge_wake_up'.
  2. Receive 'init' from central → retrieve TA keys if encryption enabled.
  3. Start local Flask-SocketIO server for cluster clients.
  4. Wait for all clients → send 'edge_ready' to central.
  5. On 'request_edge_update' from central (1 per global round):
       For edge_round in range(edge_rounds):           ← τ₂ iterations
         a. Select min_clients_per_round clients.
         b. Send 'request_update' with current edge model.
         c. Collect client updates → local FedAvg → update edge model.
       → Send 'edge_update' (final local average) to central.
  6. On 'stop_and_eval_edge': relay global model to ALL clients for evaluation,
     collect 'client_eval' results, aggregate, send 'edge_eval' to central.
     Evaluation always happens once per global round regardless of edge_rounds.

Why τ₂ > 1 matters:
  With edge_rounds=1 the edge is a transparent relay — mathematically equivalent
  to star topology. With edge_rounds > 1 the edge refines its local model across
  multiple client rounds before talking to the central, reducing central
  communication by a factor of edge_rounds while preserving convergence guarantees
  (HierFedAvg, Liu et al. 2020).
"""

import logging
import random
import time
import os
import threading
import numpy as np
import socketio as sio_module
from typing import Dict, List, Set, Optional
from threading import Lock
from flask import Flask, request
from flask_socketio import SocketIO, emit

from utils import (object_to_pickle_string, pickle_string_to_object,
                   ckks_sum_weights, ckks_multiply_weights_by_scalar)


class EdgeServer:
    """Intermediate aggregator: server to clients, client to central server."""

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.edge_id: str = config.get('edge_id', 'edge-0')
        self.logger: logging.Logger = self._setup_logger()
        self.lock = Lock()

        # --- Client-facing state ---
        # num_clients is set per-edge by run_hierarchical (dynamic distribution)
        self.num_clients: int = self.config['num_clients']
        self.min_clients_per_round: int = max(
            1, int(self.num_clients * self.config.get('models_percentage', 1.0))
        )
        self.registered_clients: Set[str] = set()
        self.client_stats_buffer: List[np.ndarray] = []
        self.client_updates_this_round: List[Dict] = []
        self.client_evaluations_this_round: List[Dict] = []
        self.num_clients_per_round: int = 0
        self.static_calibration_term: np.ndarray = None

        # --- HierFedAvg τ₂ state ---
        self.edge_rounds: int = max(1, config.get('edge_rounds', 1))
        self.current_edge_round: int = 0          # which edge round we are in [0, edge_rounds)
        self.current_edge_weights: Optional[list] = None  # plaintext avg (no_encryption) or E(avgⱼ) (ckks)
        self.edge_total_size: int = 0            # sum of participating clients' sizes in the current edge round
        self.cluster_total_size: int = 0          # sum of ALL clients' sizes (set once at edge_ready time)
        self._last_request_data: Optional[Dict] = None     # original request from central (for τ₂ > 1)
        self._expected_eval_count: int = 0        # snapshot of registered_clients at stop_and_eval time

        # --- Central-facing state ---
        self.current_round: int = -1
        self.is_training_finished: bool = False
        self._central_sio = None

        # Encryption
        self.encryption_mode: str = self.config.get('encryption_mode', 'no_encryption')
        self.ckks_context = None          # public CKKS context (no secret key); set in _on_central_init
        self._keys_received_event = threading.Event()

        # Shutdown coordination: set to True once a planned shutdown is underway so that
        # the fallback _on_central_disconnect handler does not double-relay shutdown.
        self._shutdown_initiated: bool = False

        # Guard against duplicate edge_ready when multiple clients finish simultaneously.
        self.initialization_done: bool = False

        # Local Flask-SocketIO server (for clients).
        # async_mode='threading' is required because _central_sio callbacks run in
        # standard OS threads and call self.socketio.emit() from outside Flask-SocketIO's
        # event loop. Without explicit threading mode, Flask-SocketIO may auto-detect
        # eventlet and queue messages that the eventlet hub (in a daemon thread) never
        # dispatches when called from a non-green-thread context.
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            async_mode='threading',
            ping_timeout=3600,
            ping_interval=5,
            max_http_buffer_size=int(1e32)
        )
        self._register_client_handlers()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        # Suppress werkzeug's "write() before start_response" AssertionErrors that
        # appear when clients call sio.disconnect() and engine.io sends a stray HTTP
        # request to the already-closed WebSocket on the edge's local server.
        logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

        logger = logging.getLogger(f"EdgeServer-{self.config.get('edge_id', 'N/A')}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        log_dir = os.path.join(self.config.get('log_dir', 'logs'), datestr, "Edge-Server-LOG")
        os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(os.path.join(log_dir, f'{timestr}_{self.edge_id}.log'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _register_client_handlers(self) -> None:
        self.app.route('/')(self._health_check)
        self.socketio.on('connect')(self._on_client_connect)
        self.socketio.on('disconnect')(self._on_client_disconnect)
        self.socketio.on('client_wake_up')(self._on_client_wake_up)
        self.socketio.on('client_ready')(self._on_client_ready)
        self.socketio.on('client_update')(self._on_client_update)
        self.socketio.on('client_eval')(self._on_client_eval)

    def _health_check(self):
        return "Edge Server is running", 200

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._connect_to_central()

        from utils import run_with_port_retry

        def _on_attempt(port: int) -> None:
            # self.config is the same dict object the caller (run_hierarchical)
            # holds a reference to, so this mutation is visible there too —
            # no separate shared state needed to report the real port back.
            self.config['edge_port'] = port
            self.logger.info("Edge %s: local server starting on port %d.", self.edge_id, port)

        def _bind_and_serve(port: int) -> None:
            self.socketio.run(self.app, host=self.config['ip_address'], port=port)

        def _serve_with_retry() -> None:
            try:
                run_with_port_retry(_bind_and_serve, self.config['edge_port'],
                                    on_attempt=_on_attempt, logger=self.logger)
            except Exception:
                self.logger.exception("Edge %s: local server failed permanently.", self.edge_id)

        local_thread = threading.Thread(target=_serve_with_retry, daemon=True)
        local_thread.start()

        self._central_sio.wait()

    # ------------------------------------------------------------------
    # Connection to central server
    # ------------------------------------------------------------------

    def _connect_to_central(self) -> None:
        central_url = (f"http://{self.config['ip_address']}:"
                       f"{self.config['central_port']}")
        self.logger.info("Edge %s: connecting to central server at %s.", self.edge_id, central_url)
        self._central_sio = sio_module.Client(logger=False, request_timeout=30, reconnection=False)
        self._register_central_handlers()
        self._central_sio.connect(central_url, transports=['websocket'])
        self._central_sio.emit('edge_wake_up')

    def _register_central_handlers(self) -> None:
        self._central_sio.on('init', self._on_central_init)
        self._central_sio.on('distribute_calibration', self._on_central_calibration)
        self._central_sio.on('request_edge_update', self._on_request_edge_update)
        self._central_sio.on('stop_and_eval_edge', self._on_stop_and_eval_edge)
        self._central_sio.on('shutdown', self._on_central_shutdown)
        self._central_sio.on('disconnect', self._on_central_disconnect)

    # ------------------------------------------------------------------
    # Central server event handlers
    # ------------------------------------------------------------------

    def _on_central_init(self, data: Dict):
        self.logger.info("Edge %s: received 'init' from central.", self.edge_id)

        # Run the TA key fetch in a background thread so that a transient
        # ConnectionError never kills _central_sio's receive thread.
        if self.encryption_mode == 'ckks':
            threading.Thread(
                target=self._fetch_ckks_context_from_ta,
                args=(data['ta_address'],),
                daemon=True,
            ).start()

        self.logger.info("Edge %s: waiting for %d cluster clients to connect...",
                         self.edge_id, self.num_clients)

    def _fetch_ckks_context_from_ta(self, ta_address: str, max_retries: int = 5) -> None:
        """Fetch the public CKKS context from the TA with exponential-backoff retry."""
        for attempt in range(max_retries):
            try:
                ta_sio = sio_module.Client(reconnection=False)

                @ta_sio.on('distribute_keys')
                def on_receive_ckks(key_data):
                    import tenseal as ts
                    ctx_bytes = pickle_string_to_object(key_data['ckks_context'])
                    self.ckks_context = ts.context_from(ctx_bytes)
                    self._keys_received_event.set()
                    ta_sio.disconnect()

                ta_sio.connect(ta_address, transports=['websocket'])
                ta_sio.emit('request_keys', {'role': 'edge'})
                break
            except Exception as e:
                self.logger.warning(
                    "Edge %s: TA connect attempt %d/%d failed: %s",
                    self.edge_id, attempt + 1, max_retries, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)
                else:
                    self.logger.error(
                        "Edge %s: CRITICAL — could not connect to TA after %d attempts.",
                        self.edge_id, max_retries,
                    )
                    return

        if not self._keys_received_event.wait(timeout=30):
            self.logger.error("Edge %s: CRITICAL — CKKS context not received from TA.", self.edge_id)

    def _on_central_calibration(self, data: Dict):
        self.logger.info("Edge %s: relaying calibration term to clients.", self.edge_id)
        for sid in self.registered_clients:
            self.socketio.emit('distribute_calibration', data, room=sid, namespace='/')

    def _on_request_edge_update(self, data: Dict):
        """
        Central requests one global round.
        The edge runs edge_rounds (τ₂) edge rounds before replying.
        """
        self.current_round = data['round_number']
        self.current_edge_round = 0
        self.current_edge_weights = None
        self._last_request_data = data

        self.logger.info(
            "Edge %s: global round %d -> running %d edge round(s) (tau2=%d).",
            self.edge_id, self.current_round, self.edge_rounds, self.edge_rounds
        )
        self._start_edge_round(data)

    def _start_edge_round(self, request_data: Dict) -> None:
        """
        Kick off one edge round: select clients and send them request_update
        with the weights contained in request_data.
        """
        self.client_updates_this_round.clear()

        all_clients = list(self.registered_clients)
        selected = random.sample(all_clients, min(self.min_clients_per_round, len(all_clients)))
        self.num_clients_per_round = len(selected)

        for sid in selected:
            self.socketio.emit('request_update', request_data, room=sid, namespace='/')

        self.logger.info(
            "Edge %s: edge round %d/%d — sent to %d/%d clients.",
            self.edge_id, self.current_edge_round + 1, self.edge_rounds,
            len(selected), len(all_clients)
        )

    def _on_stop_and_eval_edge(self, data: Dict):
        """Relay global model to ALL clients for evaluation (once per global round)."""
        self.is_training_finished = data.get('STOP', False)
        self.client_evaluations_this_round.clear()
        # Snapshot the count BEFORE emitting: clients with STOP=True will disconnect
        # ~0.5 s after sending client_eval, shrinking registered_clients asynchronously.
        # Using a fixed expected count avoids the race in _on_client_eval.
        self._expected_eval_count = len(self.registered_clients)
        self.logger.info("Edge %s: relaying 'stop_and_eval' to all clients (STOP=%s).",
                         self.edge_id, self.is_training_finished)
        for sid in self.registered_clients:
            self.socketio.emit('stop_and_eval', data, room=sid, namespace='/')

    def _relay_shutdown_to_clients(self) -> None:
        """
        Send the graceful 'shutdown' event to all connected clients, then
        force-disconnect any that did not respond in time.

        Called from both _on_central_shutdown (planned) and _on_central_disconnect
        (fallback when the central closes the connection before the shutdown event
        is delivered — race condition between socketio.stop() and event delivery).
        """
        client_sids = list(self.registered_clients)  # snapshot for thread safety
        self.logger.info("Edge %s: relaying shutdown to %d client(s).",
                         self.edge_id, len(client_sids))
        for sid in client_sids:
            self.socketio.emit('shutdown', room=sid, namespace='/')
        time.sleep(1)
        # Force-disconnect any clients whose sio.wait() is still blocked.
        # In the star topology socketio.stop() closes connections automatically;
        # here the edge's local server is a separate process so we must do it.
        for sid in client_sids:
            try:
                self.socketio.server.disconnect(sid, namespace='/')
            except Exception:
                pass

    def _on_central_shutdown(self, data: Dict = None):
        self.logger.info("Edge %s: received 'shutdown' from central.", self.edge_id)
        self._shutdown_initiated = True
        self._relay_shutdown_to_clients()
        self._central_sio.disconnect()

    def _on_central_disconnect(self):
        """
        Fallback: called whenever the edge loses its connection to the central server.
        If a planned 'shutdown' event already triggered _on_central_shutdown, the flag
        prevents a double relay. Otherwise (central crashed or socketio.stop() raced
        ahead of event delivery), we shut down clients here so client threads can finish.
        """
        self.logger.info("Edge %s: connection to central server lost.", self.edge_id)
        if not self._shutdown_initiated:
            self.logger.info("Edge %s: no prior shutdown event — relaying shutdown to clients.",
                             self.edge_id)
            self._relay_shutdown_to_clients()

    # ------------------------------------------------------------------
    # Client-facing server event handlers
    # ------------------------------------------------------------------

    def _on_client_connect(self):
        self.logger.info("Edge %s: client connected: %s", self.edge_id, request.sid)

    def _on_client_disconnect(self):
        self.logger.info("Edge %s: client disconnected: %s", self.edge_id, request.sid)
        self.registered_clients.discard(request.sid)

    def _on_client_wake_up(self):
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        emit('init', {'ta_address': ta_address})

    def _on_client_ready(self, data: Dict):
        client_model = data.get('model_name', '')
        expected_model = self.config.get('model_name', '')
        if client_model and expected_model and client_model != expected_model:
            self.logger.warning(
                "Edge %s: rejecting client %s — model mismatch (got '%s', expected '%s').",
                self.edge_id, request.sid, client_model, expected_model
            )
            emit('shutdown', {})
            return

        combined = None
        with self.lock:
            self.logger.info("Edge %s: client %s ready.", self.edge_id, request.sid)
            self.registered_clients.add(request.sid)

            samples_per_class = pickle_string_to_object(data['samples_per_class'])
            self.client_stats_buffer.append(samples_per_class)

            should_notify = (
                len(self.registered_clients) >= self.num_clients
                and not self.initialization_done
            )
            if should_notify:
                self.initialization_done = True
                combined = np.sum(self.client_stats_buffer, axis=0)
                self.cluster_total_size = int(np.sum(combined))

        if should_notify:
            self._central_sio.emit('edge_ready', {
                'samples_per_class': object_to_pickle_string(combined)
            })
            self.logger.info("Edge %s: all %d clients ready (cluster size=%d). Sent 'edge_ready' to central.",
                             self.edge_id, self.num_clients, self.cluster_total_size)

    def _on_client_update(self, data: Dict):
        """Accumulate client updates; when the edge round is complete, process it."""
        with self.lock:
            if data.get('round_number') != self.current_round:
                self.logger.warning(
                    "Edge %s: stale client update (round %d vs current %d). Ignoring.",
                    self.edge_id, data.get('round_number'), self.current_round
                )
                return
            self.logger.info("Edge %s: received client update from %s (edge round %d/%d).",
                             self.edge_id, request.sid,
                             self.current_edge_round + 1, self.edge_rounds)
            try:
                data['weights'] = pickle_string_to_object(data['weights'])
            except Exception as e:
                self.logger.error("Edge %s: error unpickling weights: %s", self.edge_id, e)
                return

            data['_sid'] = request.sid
            self.client_updates_this_round.append(data)

            if len(self.client_updates_this_round) >= self.num_clients_per_round:
                self._finish_edge_round()

    def _finish_edge_round(self) -> None:
        """
        Aggregate the current edge round's client updates.

        - If more edge rounds remain (current_edge_round < edge_rounds - 1):
            update the edge model and start the next edge round with it.
        - If this was the last edge round:
            send the final edge average to the central server.

        Mathematical invariant (HierFedAvg) — both modes:
            The edge always stores its LOCAL AVERAGE so the central can
            re-weight it correctly:

              global = Σⱼ(local_avg_j × size_j) / Σⱼ(size_j)
                     ≡ Σⱼ Σᵢ(wᵢⱼ·sᵢⱼ) / total        ← global FedAvg

        """
        train_losses = [x['train_loss'] for x in self.client_updates_this_round]
        train_sizes  = [x['train_size']  for x in self.client_updates_this_round]
        self.edge_total_size = sum(train_sizes)

        # With async key fetch the context may still be arriving; wait briefly.
        if self.encryption_mode != 'no_encryption' and not self._keys_received_event.is_set():
            self.logger.info("Edge %s: waiting for TA keys before aggregation...", self.edge_id)
            if not self._keys_received_event.wait(timeout=30):
                self.logger.error("Edge %s: TA keys still not available. Skipping aggregation.", self.edge_id)
                return

        # Inline aggregation — edge never trains, no Aggregator/ModelManager needed.
        # no_encryption: current_edge_weights = plaintext cluster average.
        # ckks: current_edge_weights = E(avgⱼ) — encrypted cluster average, never decrypted here.
        client_weights = [u['weights'] for u in self.client_updates_this_round]

        if self.encryption_mode == 'no_encryption':
            if client_weights and self.edge_total_size > 0:
                summed = [w * train_sizes[0] for w in client_weights[0]]
                for i in range(1, len(client_weights)):
                    for j in range(len(summed)):
                        summed[j] += client_weights[i][j] * train_sizes[i]
                self.current_edge_weights = [w / self.edge_total_size for w in summed]
            else:
                self.current_edge_weights = client_weights[0] if client_weights else []

        else:
            # Standard CKKS-FL: homomorphic weighted average — no decryption at the edge.
            # For each client i: E(wᵢ) × (sᵢ / totalⱼ)  [1 CKKS level consumed]
            # Sum: Σᵢ E(wᵢ × sᵢ/totalⱼ) = E(avgⱼ)
            # Post-rescale scale is global_scale²/prime (same for all scalars) → addition safe.
            if client_weights and self.edge_total_size > 0:
                fractions = [s / self.edge_total_size for s in train_sizes]
                weighted_enc = ckks_multiply_weights_by_scalar(
                    client_weights[0], fractions[0], self.ckks_context
                )
                for i in range(1, len(client_weights)):
                    w_i = ckks_multiply_weights_by_scalar(
                        client_weights[i], fractions[i], self.ckks_context
                    )
                    weighted_enc = ckks_sum_weights(weighted_enc, w_i, self.ckks_context)
                self.current_edge_weights = weighted_enc  # E(avgⱼ) at level 1
            else:
                self.current_edge_weights = client_weights[0] if client_weights else None

        self.current_edge_round += 1

        if self.current_edge_round < self.edge_rounds:
            self.logger.info(
                "Edge %s: edge round %d done -> starting edge round %d/%d.",
                self.edge_id, self.current_edge_round, self.current_edge_round + 1, self.edge_rounds
            )
            self._start_next_edge_round()
        else:
            avg_loss = (float(np.average(train_losses, weights=train_sizes))
                        if self.edge_total_size > 0 else 0.0)
            self._forward_to_central(avg_loss)

    def _start_next_edge_round(self) -> None:
        """Send the current edge model to clients as the starting point for the next edge round.

        current_edge_weights is the cluster average: plaintext in the baseline, E(avgⱼ)
        (level 1, never decrypted here) under CKKS, which the clients decrypt directly.
        """
        next_request_data = {
            'epochs':          self._last_request_data['epochs'],
            'batch_size':      self._last_request_data['batch_size'],
            'learning_rate':   self._last_request_data['learning_rate'],
            'round_number':    self._last_request_data['round_number'],
            'current_weights': object_to_pickle_string(self.current_edge_weights),
            # FedProx: clients must always use the CENTRAL global model as the
            # proximal reference, not the edge-averaged model that changes each
            # edge round (Section 5.1).
            'global_weights':  self._last_request_data['current_weights'],
        }
        self._start_edge_round(next_request_data)

    def _forward_to_central(self, avg_loss: float) -> None:
        """Send the cluster contribution to the central server.

        The pair (avgⱼ, Q_Ej) of Eq. (4) is forwarded, where Q_Ej is the number of
        training samples of the selected clients of the cluster.

        no_encryption: avgⱼ is the plaintext cluster average.
        CKKS: avgⱼ = E(avgⱼ), the cluster average computed homomorphically. The central
        multiplies it by (Q_Ej / Q) homomorphically and sums across edges (Eq. 5).
        """
        # E(avgⱼ) at level 1 (ckks) or the plain cluster average (no_encryption) — either
        # way the central will do one more (homomorphic or plain) scalar multiplication.
        weights_to_send = self.current_edge_weights
        effective_size = self.edge_total_size if self.edge_total_size > 0 else self.cluster_total_size

        try:
            self._central_sio.emit('edge_update', {
                'round_number': self.current_round,
                'weights':      object_to_pickle_string(weights_to_send),
                'train_loss':   avg_loss,
                'train_size':   effective_size,
            })
            self.logger.info(
                "Edge %s: all %d edge rounds done -> sent edge_update for global round %d.",
                self.edge_id, self.edge_rounds, self.current_round
            )
        except Exception as e:
            self.logger.warning("Edge %s: could not send edge_update (round %d): %s",
                                self.edge_id, self.current_round, e)

    def _on_client_eval(self, data: Dict):
        """Aggregate cluster evaluations and forward to central (once per global round).

        Validation and test metrics are averaged separately, each weighted by the size
        of the clients' evaluation subsets; the central server repeats the same weighted
        average over the edges, which equals a single average over all clients (Eq. 11).
        """
        with self.lock:
            self.client_evaluations_this_round.append(data)

            if len(self.client_evaluations_this_round) < self._expected_eval_count:
                return

            self.logger.info("Edge %s: all client evaluations received.", self.edge_id)

            evals = self.client_evaluations_this_round
            cluster_eval = {}
            for prefix in ('val', 'test'):
                sizes = [x[f'{prefix}_size'] for x in evals]
                total = sum(sizes)
                cluster_eval[f'{prefix}_size'] = total
                for metric in ('loss', 'f1', 'acc', 'prec', 'recall'):
                    key = f'{prefix}_{metric}'
                    cluster_eval[key] = (float(np.average([x[key] for x in evals], weights=sizes))
                                         if total > 0 else 0.0)
            if cluster_eval['val_size'] == 0:
                return

            self._central_sio.emit('edge_eval', cluster_eval)
            self.logger.info("Edge %s: sent 'edge_eval'.", self.edge_id)
