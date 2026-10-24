"""
Launches the full hierarchical FL stack for a single configuration.

Client distribution across edge servers:
  base = num_clients // num_edge_servers
  Each edge gets `base` clients; the last edge absorbs any remainder.

  Examples:
    8 clients, 2 edges  → [4, 4]
    7 clients, 2 edges  → [3, 4]
    9 clients, 4 edges  → [2, 2, 2, 3]
   16 clients, 4 edges  → [4, 4, 4, 4]
"""

import os
import time
import threading

from typing import Dict, List

from data_splitter import DatasetSplitter
from federated_client import FederatedClient
from edge_server import EdgeServer
from utils import wait_for_server_ready


def _compute_clients_per_edge(num_clients: int, num_edge_servers: int) -> List[int]:
    """Distribute clients across edges; last edge absorbs any remainder."""
    base = num_clients // num_edge_servers
    remainder = num_clients % num_edge_servers
    distribution = [base] * num_edge_servers
    if remainder:
        distribution[-1] += remainder
    return distribution


def start_client_thread(config: Dict, dataset_path: str) -> None:
    try:
        FederatedClient(config, dataset_path)
    except Exception as e:
        print(f"Error starting client for {dataset_path}: {e}")
        import traceback
        traceback.print_exc()


def start_edge_server_thread(config: Dict) -> None:
    edge_id = config.get('edge_id', 'edge-?')
    try:
        edge = EdgeServer(config)
        edge.run()
    except Exception as e:
        print(f"Error in edge server {edge_id}: {e}")
        import traceback
        traceback.print_exc()


def main(config: Dict) -> None:
    """Splits the dataset and launches the full central/edge/client stack."""
    num_edge_servers = config['num_edge_servers']
    num_clients = config['num_clients']
    splitting_dir = config['splitting_dir']
    source_images_dir = config['dataset_path']
    base_edge_port = config['edge_base_port']

    max_concurrent = min(num_clients, config.get('max_concurrent_training', num_clients))
    config['training_semaphore'] = threading.Semaphore(max_concurrent)
    print(f"[Clients] Concurrent training slots: {max_concurrent}/{num_clients}.")

    clients_per_edge = _compute_clients_per_edge(num_clients, num_edge_servers)

    print(f"Hierarchical topology: {num_edge_servers} edges, "
          f"{num_clients} total clients -> distribution {clients_per_edge}")

    print(f"Preparing to split the dataset from '{source_images_dir}' for {num_clients} clients into '{splitting_dir}'.")
    splitter = DatasetSplitter(
        output_base_dir=splitting_dir,
        source_images_dir=source_images_dir,
        num_clients=num_clients
    )
    # Patient-level partition into N clients, then 70/15/15 train/valid/test per client.
    splitter.split_dataset()
    print("Dataset splitting complete.")

    # Start edge servers — each with its own num_clients override.
    # edge_config['edge_port'] is only a preferred starting point: EdgeServer
    # retries the real bind past any taken port (utils.run_with_port_retry)
    # and mutates this same dict in place, since the node may be shared with
    # other users' processes that no upfront check could have foreseen.
    edge_threads: List[threading.Thread] = []
    edge_configs: List[Dict] = []
    for edge_idx in range(num_edge_servers):
        edge_config = config.copy()
        edge_config['edge_id'] = f'edge-{edge_idx}'
        edge_config['edge_port'] = base_edge_port + edge_idx
        edge_config['num_clients'] = clients_per_edge[edge_idx]

        t = threading.Thread(
            target=start_edge_server_thread,
            args=(edge_config,),
            daemon=True
        )
        t.start()
        edge_threads.append(t)
        edge_configs.append(edge_config)
        time.sleep(1)

    # Wait for each edge to actually be listening on whatever port it ended up
    # binding to — never trust the preferred port blindly.
    for edge_idx, (edge_config, edge_thread) in enumerate(zip(edge_configs, edge_threads)):
        ready = wait_for_server_ready(
            lambda ec=edge_config: f"http://{ec['ip_address']}:{ec['edge_port']}/",
            server_thread=edge_thread
        )
        if not ready:
            print(f"WARNING: edge-{edge_idx} did not become ready in time; "
                  f"its clients may fail to connect.")

    print(f"Started {num_edge_servers} edge servers.")

    # Start clients, grouped by edge server, using a running offset.
    # Read the edge's ACTUAL bound port (edge_config['edge_port']), not the
    # preferred one recomputed from base_edge_port — they can differ if the
    # edge had to retry past a taken port.
    client_threads: List[threading.Thread] = []
    client_offset = 0
    for edge_idx in range(num_edge_servers):
        edge_port = edge_configs[edge_idx]['edge_port']
        n_this_edge = clients_per_edge[edge_idx]

        for local_idx in range(n_this_edge):
            global_idx = client_offset + local_idx
            client_dataset_path = os.path.join(splitting_dir, f'client_{global_idx}')

            client_config = config.copy()
            client_config['edge_port'] = edge_port

            t = threading.Thread(
                target=start_client_thread,
                args=(client_config, client_dataset_path),
                daemon=True
            )
            t.start()
            client_threads.append(t)
            time.sleep(2)

        client_offset += n_this_edge

    print(f"Started {num_clients} clients "
          f"({[clients_per_edge[i] for i in range(num_edge_servers)]} per edge).")

    for t in client_threads:
        t.join()

    print("All client threads have finished.")


if __name__ == '__main__':
    print("This script is intended to be run by 'federated_grid_search.py'.")
