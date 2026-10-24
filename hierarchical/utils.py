import pickle
import codecs
import logging
from typing import Callable, List, Dict, Any, Optional, Union
import numpy as np
import time
import errno
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

# Configure a default logger for when none is provided
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def wait_for_server_ready(
        get_url: Union[str, Callable[[], str]],
        timeout: int = 60,
        expected_run_id: Optional[str] = None,
        server_thread=None,
) -> bool:
    """Poll a URL until the server responds with the expected run-id (or any
    200 if no id given).

    get_url may be a plain string, or a zero-arg callable evaluated on every
    poll — use a callable when the target port can change while polling is
    under way (e.g. a server retrying past an in-use port updates a shared
    config dict; re-reading it each iteration means we always poll whatever
    port is actually being attempted right now instead of a stale one
    captured before the first bind attempt).

    If *server_thread* is provided and it dies before the server responds,
    the function returns False immediately so dependents are not launched
    against a stale/dead server.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if server_thread is not None and not server_thread.is_alive():
            return False
        url = get_url() if callable(get_url) else get_url
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                if expected_run_id is None or response.text.strip() == expected_run_id:
                    return True
        except (RequestsConnectionError, requests.exceptions.RequestException):
            pass
        time.sleep(1)
    return False


# ===================================================================
# Port-bind retry — safe under uncoordinated port usage (e.g. shared
# HPC nodes where other users' processes may occupy any port at any
# time). The OS bind() call itself is the only trustworthy authority
# on whether a port is free; a separate pre-check-then-close probe
# always leaves a race window. This helper retries the REAL bind
# against the next candidate port whenever it fails with EADDRINUSE.
# ===================================================================

def is_address_in_use_error(exc: OSError) -> bool:
    """True if exc represents a bind() failure because the port is taken
    (POSIX EADDRINUSE, or Windows WSAEADDRINUSE / winerror 10048)."""
    if getattr(exc, 'winerror', None) == 10048:
        return True
    return exc.errno == errno.EADDRINUSE


def run_with_port_retry(
        bind_and_serve: Callable[[int], None],
        initial_port: int,
        max_attempts: int = 50,
        step: int = 1,
        on_attempt: Optional[Callable[[int], None]] = None,
        logger: logging.Logger = log,
) -> None:
    """Call bind_and_serve(port), retrying on the next port if the bind fails
    because the port is already in use.

    bind_and_serve(port) must perform the real bind/listen and then block for
    the lifetime of the server (it should raise the underlying OSError if the
    bind itself fails, before entering its serve loop).

    on_attempt(port), if given, runs immediately before each attempt so
    callers can publish the candidate port to whatever needs to connect to
    this server (e.g. a shared dict polled by a readiness check) — this must
    happen even for attempts that will go on to fail, since the caller has no
    other way to observe port changes as they happen.
    """
    port = initial_port
    last_err: Optional[OSError] = None
    for attempt in range(max_attempts):
        if on_attempt:
            on_attempt(port)
        try:
            bind_and_serve(port)
            return
        except OSError as e:
            if is_address_in_use_error(e):
                last_err = e
                logger.warning(
                    "Port %d already in use, retrying on %d (attempt %d/%d).",
                    port, port + step, attempt + 1, max_attempts
                )
                port += step
                continue
            raise
    raise RuntimeError(
        f"Could not bind starting from port {initial_port} after {max_attempts} attempts."
    ) from last_err


def object_to_pickle_string(obj: Any, file_path: str = None, save_to_file: bool = False) -> str:
    """
    Serializes a Python object into a base64 encoded string or saves it to a file.

    Args:
        obj (Any): The Python object to serialize.
        file_path (str, optional): The file path to save the object to. Defaults to None.
        save_to_file (bool): If True and file_path is provided, saves to a file
                             and returns the path. Otherwise, returns a string.

    Returns:
        str: A base64 encoded string of the pickled object or the file path if saved.
    """
    if save_to_file and file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        return file_path

    return codecs.encode(pickle.dumps(obj), "base64").decode()


def pickle_string_to_object(s: str) -> Any:
    """
    Deserializes a base64 encoded string or a pickle file path into a Python object.

    Note:
        This function handles two cases for backward compatibility:
        1. If 's' is a base64 encoded string.
        2. If 's' is a file path ending in '.pkl'.
        This dual functionality should be used with care.

    Args:
        s (str): The base64 encoded string or the file path to a pickle file.

    Returns:
        Any: The deserialized Python object.
    """
    if ".pkl" in s:
        log.debug("Deserializing object from file path: %s", s)
        with open(s, "rb") as f:
            return pickle.load(f)
    else:
        log.debug("Deserializing object from base64 string.")
        return pickle.loads(codecs.decode(s.encode(), "base64"))


# ===================================================================
# CKKS (TenSEAL) — encrypted aggregation for both topologies
# ===================================================================
# Standard CKKS-FL with homomorphic weighted averaging.  Level budget
# with coeff_mod_bit_sizes=[60, 40, 40, 60] (200 bits ≤ 218-bit limit): 2 usable levels.
# Rule: usable_levels = len(coeff_mod_bit_sizes) - 2  (the two 60-bit "bookend"
# primes are never dropped during rescaling — only the inner primes are).
#
#   fresh (level 2): client encrypts E(wᵢ)
#   Star topology (depth 1):
#     level 1: server computes Σᵢ E(wᵢ)×(Qᵢ/Q) = E(global_avg)        [−1 level]
#   Hierarchical topology (depth 2):
#     level 1: edge computes Σᵢ E(wᵢ)×(Qᵢ/Q_Eⱼ) = E(avgⱼ)            [−1 level]
#     level 0: central computes Σⱼ E(avgⱼ)×(Q_Eⱼ/Q) = E(global_avg)  [−1 level]
#   decrypt: client gets plaintext global_avg — no division needed.
#
# Key property: after mul_plain + rescale the resulting scale is
# global_scale²/prime, determined by context parameters only (NOT the
# scalar value).  Results from different scalars therefore share the
# same scale and can be safely added.  No relin/galois keys needed.
# Edge and central servers only ever hold the public context.
# ===================================================================

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False


def _require_tenseal() -> None:
    if not TENSEAL_AVAILABLE:
        raise ImportError(
            "tenseal is required for CKKS encryption. "
            "Install with: pip install tenseal"
        )


_CKKS_POLY_MOD_DEGREE = 8192
_CKKS_MAX_SLOTS = _CKKS_POLY_MOD_DEGREE // 2  # 4096


def create_ckks_context(poly_modulus_degree: int = _CKKS_POLY_MOD_DEGREE, scale_bits: int = 40):
    """Create a TenSEAL CKKS context for 2-level homomorphic aggregation.

    coeff_mod_bit_sizes=[60, 40, 40, 60]: two inner 40-bit primes give 2 usable
    mul_plain levels (one consumed at the edge, one at the central).
    Total = 200 bits < 218-bit limit for poly_modulus_degree=8192, 128-bit security.
    No relin/galois keys needed (no ciphertext-ciphertext multiplication).
    """
    _require_tenseal()
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** scale_bits
    return context


def ckks_encrypt_weights(context, weights: List[np.ndarray]) -> List[Dict[str, Any]]:
    """Encrypt a list of ndarrays with CKKS.

    Returns List[Dict{'shape': tuple, 'chunks': List[bytes]}].
    Each 'chunks' entry is a serialized CKKSVector covering up to
    _CKKS_MAX_SLOTS values.
    """
    _require_tenseal()
    max_slots = _CKKS_MAX_SLOTS
    result = []
    for w in weights:
        flat = w.flatten().tolist()
        shape = w.shape
        chunks: List[bytes] = []
        for start in range(0, max(1, len(flat)), max_slots):
            chunk = flat[start:start + max_slots]
            enc = ts.ckks_vector(context, chunk)
            chunks.append(enc.serialize())
        result.append({'shape': shape, 'chunks': chunks})
    return result


def ckks_decrypt_weights(context, encrypted_layers: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Decrypt List[Dict{'shape', 'chunks'}] → List[ndarray].

    Requires a context that contains the secret key.
    """
    _require_tenseal()
    result = []
    for layer in encrypted_layers:
        shape = layer['shape']
        total_elems = int(np.prod(shape))
        flat: List[float] = []
        for chunk_bytes in layer['chunks']:
            vec = ts.ckks_vector_from(context, chunk_bytes)
            flat.extend(vec.decrypt())
        arr = np.array(flat[:total_elems], dtype=np.float32).reshape(shape)
        result.append(arr)
    return result


def ckks_sum_weights(
        enc_a: List[Dict[str, Any]],
        enc_b: List[Dict[str, Any]],
        context
) -> List[Dict[str, Any]]:
    """Homomorphic addition of two CKKS-encrypted weight lists.

    Both inputs must come from contexts with identical parameters.
    No CKKS levels consumed.  Works with public context (no secret key).
    """
    _require_tenseal()
    result = []
    for la, lb in zip(enc_a, enc_b):
        new_chunks: List[bytes] = []
        for ca_bytes, cb_bytes in zip(la['chunks'], lb['chunks']):
            va = ts.ckks_vector_from(context, ca_bytes)
            vb = ts.ckks_vector_from(context, cb_bytes)
            new_chunks.append((va + vb).serialize())
        result.append({'shape': la['shape'], 'chunks': new_chunks})
    return result


def ckks_multiply_weights_by_scalar(
        enc_weights: List[Dict[str, Any]],
        scalar: float,
        context
) -> List[Dict[str, Any]]:
    """Homomorphic scalar multiplication for CKKS-encrypted weights.

    After mul_plain + rescale, the resulting scale is global_scale^2 / prime,
    which is the same regardless of the actual scalar value (scale is set by
    the context parameter, not by the scalar).  This means results from
    different scalars can be safely added together.

    Consumes one CKKS level.  Works with public context (no secret key).
    """
    _require_tenseal()
    result = []
    for layer in enc_weights:
        new_chunks: List[bytes] = []
        for chunk_bytes in layer['chunks']:
            vec = ts.ckks_vector_from(context, chunk_bytes)
            new_chunks.append((vec * scalar).serialize())
        result.append({'shape': layer['shape'], 'chunks': new_chunks})
    return result