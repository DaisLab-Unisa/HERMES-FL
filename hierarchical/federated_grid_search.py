"""
Grid search orchestrator for the hierarchical FL topology.

Key differences from the star topology (see star/federated_grid_search.py):
  - Spawns a CentralServer thread instead of FederatedServer.
  - Spawns N EdgeServer threads (each on its own port).
  - Clients connect to edge servers, not the central server.
  - Config requires: central_port, edge_base_port, num_edge_servers, clients_per_edge.
  - num_edge_servers is derived as num_clients // clients_per_edge.
  - MIN_NUM_WORKERS applies at the edge level (fraction of clients per edge round).
"""

import json
import threading
import time
import itertools
import socket
import uuid
import pandas as pd
import os
import multiprocessing
import queue
import shutil

from typing import Dict, Set, FrozenSet, Tuple

import central_server
import run_hierarchical
from utils import wait_for_server_ready


GRID_SEARCH_CONFIG_PATH = 'grid_search_config.json'

# Spacing between workers' preferred port blocks within this script. Kept well
# above the max ports a single config needs (n_edges + 2) and above
# find_free_port_block's scan step, so a worker skipping stuck/occupied ports
# doesn't wander into the next worker's preferred block. This is only a
# starting-point optimization to reduce retries/log noise — the actual
# authority on port availability is the real bind-retry in TrustedAuthority /
# CentralServer / EdgeServer (utils.run_with_port_retry), since the node may
# be shared with other users' processes that no local check can see coming.
WORKER_PORT_SPACING = 50

MATCH_KEYS = [
    'dataset_name', 'model_name', 'global_epoch', 'local_epoch', 'num_clients', 'num_edge_servers',
    'edge_rounds', 'learning_rate', 'models_percentage', 'aggregation_algorithm',
    'num_custom_layers', 'encryption_mode', 'image_size'
]

# Minimum clients per edge for meaningful local aggregation (HierFedAvg criterion).
MIN_CLIENTS_PER_EDGE = 2

# ----------------------------------------------------------------------------
# FINGERPRINT HELPERS
# ----------------------------------------------------------------------------

def _normalize_fp_value(v) -> str:
    """
    Normalize a configuration value to a canonical string representation.

    Booleans are handled before the numeric path because ``float(False) == 0.0``
    and ``float(True) == 1.0``, which would cause ``False`` to normalize to ``'0'``
    while the CSV (read with ``dtype=str``) stores it as ``'False'`` — a mismatch
    that makes completed runs appear as new ones.
    """
    if isinstance(v, bool):
        return str(v)
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(v)


def make_config_fingerprint(config: dict) -> FrozenSet[Tuple[str, str]]:
    """
    Produce an immutable fingerprint for a hyperparameter configuration.

    ``num_custom_layers`` is excluded for ConvNet because that model's search
    space uses ``convnet_hidden1``/``convnet_hidden2`` instead — the key is
    absent from generated ConvNet configs, while historical CSV rows carry a
    `-1` placeholder for it, so including it would make the two never match.
    """
    is_convnet = config.get('model_name') == 'ConvNet'
    effective_keys = [k for k in MATCH_KEYS if k != 'num_custom_layers' or not is_convnet]
    return frozenset(
        (k, _normalize_fp_value(config[k])) for k in effective_keys if k in config
    )


def build_fingerprint_sets(global_csv_path: str) -> Set[FrozenSet]:
    """Parse the global results CSV and build the set of already-completed configuration fingerprints.

    Every row in this CSV corresponds to a fully finished run (save_results()
    is only ever called once training completes), so presence in the file is
    sufficient — no partial/incomplete-run tracking is needed.
    """
    completed: Set[FrozenSet] = set()

    if not os.path.exists(global_csv_path):
        return completed

    try:
        df = pd.read_csv(global_csv_path, dtype=str)
    except Exception:
        return completed

    if df.empty:
        return completed

    for _, row in df.iterrows():
        row_is_convnet = str(row.get('model_name', '')) == 'ConvNet'
        effective_keys = [k for k in MATCH_KEYS if k != 'num_custom_layers' or not row_is_convnet]
        fp = frozenset(
            (k, _normalize_fp_value(row[k]))
            for k in effective_keys
            if k in row and pd.notna(row[k]) and str(row[k]) != ''
        )
        completed.add(fp)

    return completed


# ----------------------------------------------------------------------------
# CLEANUP AND DUPLICATE HANDLING
# ----------------------------------------------------------------------------

def remove_duplicate_configurations(global_csv_path: str) -> None:
    """Remove duplicate rows from the global results CSV, keeping the last occurrence."""
    if not os.path.exists(global_csv_path):
        return
    try:
        df = pd.read_csv(global_csv_path, dtype=str)
    except Exception as e:
        print(f"[CLEANUP ERROR] Cannot read CSV: {e}")
        return

    if df.empty:
        return

    original_rows = len(df)
    subset_keys = [k for k in MATCH_KEYS if k in df.columns]
    df_cleaned = df.drop_duplicates(subset=subset_keys, keep='last')
    removed_count = original_rows - len(df_cleaned)

    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} duplicate rows from the CSV.")
        df_cleaned.to_csv(global_csv_path, index=False)
    else:
        print("[CLEANUP] No duplicates found in the CSV.")


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------

def load_json(filename: str) -> dict:
    """Load and parse a JSON file."""
    with open(filename) as f:
        return json.load(f)


def generate_configurations(base_config: dict, search_space: dict):
    """Generate all hyperparameter configurations from a Cartesian product of the search space."""
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def find_free_port_block(start: int, count: int, step: int = 10) -> int:
    """Return the first starting port where 'count' consecutive ports are all free."""
    port = start
    while port + count - 1 < 65535:
        socks = []
        try:
            for i in range(count):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('127.0.0.1', port + i))
                socks.append(s)
            return port
        except OSError:
            port += step
        finally:
            for s in socks:
                try:
                    s.close()
                except Exception:
                    pass
    raise RuntimeError(f"No free block of {count} consecutive ports found from {start}.")


def pre_download_model_weights(model_names: set) -> None:
    """Pre-fetches torchvision pretrained weights once before the grid search
    starts, so parallel workers don't race to download the same weights
    concurrently on first use."""
    try:
        import torchvision.models as tv_models
    except ImportError:
        print("[PreDownload] torchvision not available, skipping.")
        return
    weight_map = {
        'GoogLeNet': (tv_models.googlenet, tv_models.GoogLeNet_Weights.IMAGENET1K_V1),
        'AlexNet':   (tv_models.alexnet,   tv_models.AlexNet_Weights.IMAGENET1K_V1),
        'ResNet18':  (tv_models.resnet18,  tv_models.ResNet18_Weights.IMAGENET1K_V1),
        'ResNet34':  (tv_models.resnet34,  tv_models.ResNet34_Weights.IMAGENET1K_V1),
        'ResNet50':  (tv_models.resnet50,  tv_models.ResNet50_Weights.IMAGENET1K_V1),
        'ResNet101': (tv_models.resnet101, tv_models.ResNet101_Weights.IMAGENET1K_V1),
    }
    print("\n=== Pre-downloading model weights ===")
    for name in sorted(model_names):
        if name not in weight_map:
            print(f"[PreDownload] {name}: no pretrained weights (skipping).")
            continue
        fn, weights_enum = weight_map[name]
        print(f"[PreDownload] Checking {name}...", flush=True)
        try:
            _m = fn(weights=weights_enum)
            del _m
            print(f"[PreDownload] {name} ready.")
        except Exception as e:
            print(f"[PreDownload] WARNING: could not pre-load {name}: {e}")
    print("=== Pre-download complete ===\n")


# ----------------------------------------------------------------------------
# TRUSTED AUTHORITY THREAD
# ----------------------------------------------------------------------------

def start_ta_thread(config: dict) -> None:
    """Instantiates and runs a TrustedAuthority server inside the calling thread (daemon)."""
    from trusted_authority import TrustedAuthority
    worker_id = config.get('worker_id', 'N/A')
    ta_port = config['ta_port']
    print(f"[Worker {worker_id}] Starting Trusted Authority (preferred port {ta_port})...")
    try:
        ta = TrustedAuthority(
            host=config['ip_address'],
            port=ta_port,
            encryption_mode=config.get('encryption_mode', 'no_encryption'),
        )
        # TrustedAuthority is built from plain host/port args, not a reference
        # to `config` — mirror every bind attempt (including retries past a
        # taken port) back into config['ta_port'] so wait_for_server_ready can
        # poll whatever port is actually being attempted right now.
        ta.run(on_attempt=lambda port: config.__setitem__('ta_port', port))
    except Exception as e:
        print(f"[Worker {worker_id}] TA CRASH: {e}")
        import traceback
        traceback.print_exc()


# ----------------------------------------------------------------------------
# CENTRAL SERVER THREAD
# ----------------------------------------------------------------------------

def start_central_server_thread(config: dict) -> None:
    """Instantiate and run a CentralServer inside the calling thread."""
    worker_id = config.get('worker_id', 'N/A')
    port = config['central_port']
    print(f"[Worker {worker_id}] Starting Central Server on port {port}...")
    try:
        server = central_server.CentralServer(config)
        server.run()
    except Exception as e:
        print(f"!!!!!!!!!! [Worker {worker_id}] CENTRAL SERVER CRASH ON PORT {port} !!!!!!!!!!!")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print(f"[Worker {worker_id}] Central Server terminated.")


# ----------------------------------------------------------------------------
# ISOLATED SUBPROCESS — one per configuration
# ----------------------------------------------------------------------------

def _execute_single_config(config: dict) -> None:
    """Runs a SINGLE FL configuration in a dedicated child process."""
    worker_id = config.get('worker_id', 'N/A')
    model_name = config.get('model_name', '?')
    encryption_mode = config.get('encryption_mode', 'no_encryption')

    # Unique ID so wait_for_server_ready can distinguish this server from a stale
    # server left over on the same port by a previous (possibly stuck) run.
    config['server_run_id'] = str(uuid.uuid4())

    try:
        if encryption_mode not in (None, 'no_encryption'):
            ta_thread = threading.Thread(
                target=start_ta_thread,
                args=(config,),
                daemon=True
            )
            ta_thread.start()
            # config['ta_port'] may change while ta_thread retries past a
            # taken port, so re-read it on every poll instead of freezing
            # the URL at the (possibly stale) preferred port.
            if not wait_for_server_ready(lambda: f"http://{config['ip_address']}:{config['ta_port']}/",
                                         server_thread=ta_thread):
                print(f"[Worker {worker_id}] WARNING: TA not ready. Proceeding anyway.")

        server_thread = threading.Thread(
            target=start_central_server_thread,
            args=(config,),
            daemon=True
        )
        server_thread.start()

        # Same reasoning: CentralServer mutates config['central_port'] in
        # place on every bind attempt (it holds the same dict reference), so
        # re-read it fresh on each poll.
        if wait_for_server_ready(lambda: f"http://{config['ip_address']}:{config['central_port']}/",
                                  expected_run_id=config['server_run_id'],
                                  server_thread=server_thread):
            run_hierarchical.main(config)
        else:
            print(f"[Worker {worker_id}] Unable to contact central server. Skipping '{model_name}'.")

        server_thread.join()

    except Exception as e:
        print(f"[Worker {worker_id}] Error in isolated subprocess for '{model_name}': {e}")
        import traceback
        traceback.print_exc()


# ----------------------------------------------------------------------------
# WORKER PROCESS
# ----------------------------------------------------------------------------

def run_grid_search_worker(worker_id: int, task_queue: multiprocessing.JoinableQueue,
                           base_port: int, csv_lock: multiprocessing.Lock,
                           port_lock: multiprocessing.Lock) -> None:
    """
    Long-running worker that pulls configurations from the queue and executes
    each one in an isolated child subprocess.
    """
    print(f"--- Starting Hierarchical Grid Search Worker {worker_id} ---")

    while True:
        try:
            config = task_queue.get(timeout=1)
        except queue.Empty:
            continue

        if config is None:
            print(f"--- Worker {worker_id} received stop signal. Shutting down. ---")
            task_queue.task_done()
            break

        splitting_dir = None
        try:
            dataset_name = config['dataset_name']
            model_name = config['model_name']

            print("\n" + "#" * 80)
            print(f"### [Worker {worker_id}] STARTING CONFIGURATION: {dataset_name} | {model_name} ###")

            config['worker_id'] = worker_id
            # Port allocation: central + TA + num_edge_servers consecutive ports.
            # This search is only a best-effort STARTING POINT to keep logs
            # predictable and reduce retries — it is not the source of
            # correctness. The node may be shared with other users' processes
            # that this check cannot see, and even our own workers' scans
            # could otherwise race each other, so port_lock serializes the
            # search across this script's own workers, and the real servers
            # (TrustedAuthority/CentralServer/EdgeServer) retry the actual
            # bind past whatever port turns out to be taken by the time they
            # start (utils.run_with_port_retry) — that retry is what's
            # actually authoritative.
            n_edges = config['num_edge_servers']
            preferred_start = base_port + (worker_id * WORKER_PORT_SPACING)
            with port_lock:
                free_start = find_free_port_block(preferred_start, count=n_edges + 2, step=1)
            config['central_port'] = free_start
            config['ta_port'] = free_start + 1
            config['edge_base_port'] = free_start + 2
            if free_start != preferred_start:
                print(f"[Worker {worker_id}] Preferred ports busy — using block "
                      f"starting at {free_start}.")

            # MIN_NUM_WORKERS here is for logging/aggregator context only;
            # each EdgeServer computes its own min_clients_per_round locally.
            base_clients_per_edge = config['num_clients'] // config['num_edge_servers']
            config['MIN_NUM_WORKERS'] = max(
                1, int(base_clients_per_edge * config.get('models_percentage', 1.0))
            )
            config['csv_lock'] = csv_lock

            run_uuid = uuid.uuid4().hex[:8]
            run_identifier = f"{dataset_name}_{model_name}_w{worker_id}_{run_uuid}"
            config['log_dir'] = os.path.join(config['base_log_path'], run_identifier)
            config['run_metrics_output_path'] = os.path.join(config['base_csv_path'], dataset_name)
            splitting_dir = os.path.join(config['base_split_data_path'], run_identifier)
            config['splitting_dir'] = splitting_dir

            os.makedirs(config['log_dir'], exist_ok=True)
            os.makedirs(splitting_dir, exist_ok=True)

            config_proc = multiprocessing.Process(
                target=_execute_single_config,
                args=(config,),
                daemon=False
            )
            config_proc.start()
            # Timeout scales with global_epoch × edge_rounds so that configs
            # with edge_rounds=2 get roughly double the wall-clock budget of
            # edge_rounds=1.
            timeout_sec = config.get(
                'subprocess_timeout_sec',
                max(14400, config.get('global_epoch', 30)
                         * max(1, config.get('edge_rounds', 1))
                         * 480)
            )
            config_proc.join(timeout=timeout_sec)
            if config_proc.is_alive():
                print(f"[Worker {worker_id}] WARNING: subprocess for '{model_name}' "
                      f"timed out after {timeout_sec}s. Terminating.")
                config_proc.terminate()
                config_proc.join()

            if config_proc.exitcode not in (0, None):
                print(f"[Worker {worker_id}] WARNING: subprocess for '{model_name}' "
                      f"exited with code {config_proc.exitcode}.")
            else:
                print(f"[Worker {worker_id}] Subprocess for '{model_name}' completed. "
                      f"Memory fully released by OS.")

        except Exception as e:
            print(f"[Worker {worker_id}] Error preparing configuration: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if splitting_dir and os.path.isdir(splitting_dir):
                try:
                    shutil.rmtree(splitting_dir)
                    print(f"[Worker {worker_id}] Removed splitting dir: {splitting_dir}")
                except Exception as e:
                    print(f"[Worker {worker_id}] Could not remove splitting dir: {e}")

            task_queue.task_done()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main(override_config: dict = None):
    """Entry point for the distributed grid search (hierarchical topology).

    Args:
        override_config: optional dict whose keys overwrite the JSON config
                         (e.g. {'csv_tag': 'ckks'}).
    """
    base_grid_config = load_json(GRID_SEARCH_CONFIG_PATH)
    if override_config:
        base_grid_config.update(override_config)
    csv_lock = multiprocessing.Lock()
    task_queue = multiprocessing.JoinableQueue()

    base_csv_path = base_grid_config.get('base_csv_path', 'csv')
    os.makedirs(base_csv_path, exist_ok=True)
    os.makedirs(base_grid_config.get('base_log_path', 'logs'), exist_ok=True)
    os.makedirs(base_grid_config.get('base_split_data_path', 'run_dataset'), exist_ok=True)

    # csv_tag lets entrypoints that share a search space but write to
    # different result trees (e.g. run_ckks.py / run_no_encryption.py) give
    # their global CSVs distinct names too, not just distinct folders — so a
    # file copied out of its results_*/csv/ directory is still identifiable
    # instead of colliding with the sibling script's identically-named file.
    csv_tag = base_grid_config.get('csv_tag', '')
    tag_suffix = f"_{csv_tag}" if csv_tag else ''
    csv_path = os.path.join(base_csv_path, f"global_grid_search_summary{tag_suffix}.csv")

    common_space = base_grid_config['common_search_space']
    model_space = base_grid_config['model_specific_search_space']
    clients_per_edge = base_grid_config.get('clients_per_edge', 2)

    fixed_params = {k: v for k, v in base_grid_config.items() if
                    k not in ['datasets', 'common_search_space', 'model_specific_search_space']}

    print("Checking and removing any duplicate configurations in the CSV...")
    remove_duplicate_configurations(csv_path)

    print("Building fingerprint set from the existing CSV (one-time read)...")
    completed_fps = build_fingerprint_sets(csv_path)
    print(f"  Completed configs: {len(completed_fps)}")

    configs_to_run_count = 0
    total_generated_configs = 0

    for dataset_info in base_grid_config['datasets']:
        for model_name, specific_params in model_space.items():
            model_base_config = fixed_params.copy()
            model_base_config.update({
                'dataset_name': dataset_info['name'],
                'dataset_path': dataset_info['path'],
                'num_classes': dataset_info['num_classes'],
                'num_images': dataset_info.get('num_images'),
                'model_name': model_name
            })

            current_search_space = {**common_space, **specific_params}

            for hyper_config in generate_configurations(model_base_config, current_search_space):
                total_generated_configs += 1

                # The hierarchical search space encodes homomorphic encryption
                # as encryption_mode='homomorphic' + he_backend='ckks' (the star topology's
                # own convention, kept here for consistency), but the shared root
                # engine (TrustedAuthority/CentralServer/EdgeServer/FederatedClient/
                # Aggregator) branches on the literal encryption_mode value itself —
                # 'ckks' for the TenSEAL path. he_backend is normalized into
                # encryption_mode here rather than touching the config file.
                if hyper_config.get('encryption_mode') == 'homomorphic':
                    hyper_config['encryption_mode'] = hyper_config.get('he_backend', 'ckks')

                # num_edge_servers is derived: num_clients // clients_per_edge.
                # This enforces e.g. 8 clients -> 4 edges of 2 clients each.
                hyper_config['num_edge_servers'] = hyper_config['num_clients'] // clients_per_edge

                # Validity check: each edge must aggregate at least
                # MIN_CLIENTS_PER_EDGE clients for local aggregation to be
                # meaningful (HierFedAvg criterion).
                n_clients = hyper_config['num_clients']
                n_edges = hyper_config['num_edge_servers']
                if n_edges == 0 or n_clients // n_edges < MIN_CLIENTS_PER_EDGE:
                    print(f"[SKIP] num_clients={n_clients}, num_edge_servers={n_edges} -> "
                          f"invalid or < {MIN_CLIENTS_PER_EDGE} client(s)/edge. Skipping.")
                    continue

                fp = make_config_fingerprint(hyper_config)

                if fp in completed_fps:
                    continue

                task_queue.put(hyper_config)
                configs_to_run_count += 1

    print("=" * 80)
    print(f"Total configurations explored:             {total_generated_configs}")
    print(f"Configurations skipped (already executed): {total_generated_configs - configs_to_run_count}")
    print(f"Configurations queued for execution:       {configs_to_run_count}")
    print("=" * 80)

    if task_queue.empty():
        print("=== No new configurations to execute. Exiting. ===")
        return

    pre_download_model_weights(set(model_space.keys()))

    num_parallel_executions = int(base_grid_config.get('num_parallel_executions', 2))
    processes = []
    print(f"\n=== Starting {num_parallel_executions} parallel workers ===")
    base_port = base_grid_config['central_port']
    port_lock = multiprocessing.Lock()

    for i in range(num_parallel_executions):
        process = multiprocessing.Process(
            target=run_grid_search_worker,
            args=(i, task_queue, base_port, csv_lock, port_lock)
        )
        processes.append(process)
        process.start()

    task_queue.join()

    for _ in range(num_parallel_executions):
        task_queue.put(None)

    for process in processes:
        process.join()

    print("\n=== Parallel execution completed. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
