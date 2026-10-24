import logging
from typing import Callable, Optional
from flask import Flask, request
from flask_socketio import SocketIO, emit
from utils import object_to_pickle_string


class TrustedAuthority:
    """
    Trusted Authority (TA) for distributing cryptographic material.

    Generates a TenSEAL CKKS context and distributes:
      * full context (with secret key) to clients (role='client')
      * public context (no secret key) to edges/central (role='edge'/'server')
    """

    def __init__(self, host: str, port: int, encryption_mode: str = 'ckks'):
        self.host = host
        self.port = port
        self.encryption_mode = encryption_mode
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.logger = self._setup_logger()

        if encryption_mode != 'ckks':
            raise ValueError(
                f"Unsupported encryption_mode '{encryption_mode}'; TrustedAuthority only supports 'ckks'."
            )
        self._init_ckks()

        self._register_handlers()

    # ------------------------------------------------------------------
    # Key / context generation
    # ------------------------------------------------------------------

    def _init_ckks(self) -> None:
        from utils import create_ckks_context
        self.logger.info("Generating TenSEAL CKKS context (poly_modulus_degree=8192)...")
        context = create_ckks_context()
        self.logger.info("CKKS context generated.")

        # Serialize before and after dropping the secret key
        self.pickled_ckks_full = object_to_pickle_string(context.serialize(save_secret_key=True))
        self.pickled_ckks_public = object_to_pickle_string(context.serialize(save_secret_key=False))

    # ------------------------------------------------------------------
    # Server plumbing
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"TrustedAuthority-{self.port}")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _register_handlers(self) -> None:
        self.app.route('/')(self._health_check)
        self.socketio.on('connect')(self._on_connect)
        self.socketio.on('disconnect')(self._on_disconnect)
        self.socketio.on('request_keys')(self._on_request_keys)
        self.socketio.on('shutdown_ta')(self._on_shutdown_ta)

    def _health_check(self):
        return "TA is running", 200

    def run(self, on_attempt: Optional[Callable[[int], None]] = None) -> None:
        """Binds and serves, retrying on the next port if self.port is taken.

        The node may be shared with other users' processes, so the port passed
        at construction time is only a preferred starting point, not a
        guarantee — self.port is updated in place on every attempt so callers
        polling it (e.g. for a readiness check) see the port actually in use.

        on_attempt, if given, is also invoked with each candidate port — used
        by callers that need to mirror the real port into a config dict that
        isn't otherwise reachable from this object (e.g. federated_grid_search's
        start_ta_thread, since TrustedAuthority is constructed from plain
        host/port args rather than holding a reference to that config).
        """
        from utils import run_with_port_retry

        def _on_attempt(port: int) -> None:
            self.port = port
            self.logger.info("Trusted Authority starting on http://%s:%d", self.host, port)
            if on_attempt:
                on_attempt(port)

        def _bind_and_serve(port: int) -> None:
            self.socketio.run(self.app, host=self.host, port=port)

        run_with_port_retry(_bind_and_serve, self.port, on_attempt=_on_attempt, logger=self.logger)

    def _on_connect(self):
        self.logger.info("Client connected: %s", request.sid)

    def _on_disconnect(self):
        self.logger.info("Client disconnected: %s", request.sid)

    def _on_shutdown_ta(self):
        print("Shutdown request received. Stopping the Trusted Authority server.")
        print("Trusted Authority server stopped successfully.")
        exit(0)

    def _on_request_keys(self, data=None):
        """Distribute the CKKS context based on requester role."""
        role = (data or {}).get('role', 'client') if isinstance(data, dict) else 'client'
        self.logger.info("Key request from %s (role=%s).", request.sid, role)

        ctx_pickled = (self.pickled_ckks_full
                       if role == 'client'
                       else self.pickled_ckks_public)
        emit('distribute_keys', {
            'encryption_mode': 'ckks',
            'ckks_context': ctx_pickled,
        })


if __name__ == '__main__':
    ta = TrustedAuthority(host='127.0.0.1', port=5001)
    ta.run()
