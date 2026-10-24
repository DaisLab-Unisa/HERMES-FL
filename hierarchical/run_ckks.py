"""
Runs the hierarchical-topology FL grid search restricted to encryption_mode='homomorphic' (CKKS).

Writes to its own results_ckks/ tree (csv/log/run_dataset) under the shared
../results/results_hierarchical/ directory instead of the defaults in
grid_search_config.json, so it never writes into the same CSV file as the
sibling run_no_encryption.py script when they run concurrently, and so the
global CSV itself is named distinctly (via csv_tag) in case a file is ever
copied out of results_ckks/csv/.
"""

import multiprocessing

import federated_grid_search as fgs

# The hierarchical search space encodes homomorphic encryption as
# encryption_mode='homomorphic' + he_backend='ckks' (federated_grid_search.py
# normalizes it to the literal he_backend value per-config before it reaches
# the shared CentralServer/EdgeServer/TrustedAuthority).
ENCRYPTION_MODE = 'homomorphic'
HE_BACKEND = 'ckks'

# Port block base so this script can run alongside run_no_encryption.py
# (base 35000) without conflicting, and clear of the star topology's port blocks
# (25000/26000/27000) so both architectures can run concurrently on the
# same node. This is only a starting-point hint for find_free_port_block —
# the real bind-retry in TrustedAuthority/CentralServer/EdgeServer
# (utils.run_with_port_retry) is what actually guarantees correctness on a
# node that may be shared with other users' processes.
_PORT_BASE = 36000


def run_all():
    # common_search_space is read fresh from the JSON and narrowed to a
    # single encryption_mode/he_backend pair here, rather than edited in the
    # JSON itself, so the same grid_search_config.json keeps describing the
    # full search space and the run_*.py scripts stay simple slices of it.
    base_config = fgs.load_json(fgs.GRID_SEARCH_CONFIG_PATH)
    common_space = dict(base_config['common_search_space'])
    common_space['encryption_mode'] = [ENCRYPTION_MODE]
    common_space['he_backend'] = [HE_BACKEND]

    banner = "  ENCRYPTION: ckks  "
    line = "#" * (len(banner) + 4)
    print(f"\n{line}\n# {banner} #\n{line}\n")

    fgs.main(override_config={
        'common_search_space': common_space,
        'central_port': _PORT_BASE,
        'base_csv_path': '../results/results_hierarchical/results_ckks/csv',
        'base_log_path': '../results/results_hierarchical/results_ckks/log',
        'base_split_data_path': '../results/results_hierarchical/results_ckks/run_dataset',
        # Distinguishes this run's global CSV filename from
        # run_no_encryption.py's even if a file is copied out of
        # results_ckks/csv/ (see csv_tag in federated_grid_search.main).
        'csv_tag': 'ckks',
    })
    print("=== ckks run complete. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    run_all()
