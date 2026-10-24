"""
Runs the star-topology FL grid search restricted to encryption_mode='no_encryption'.

Writes to its own results_no_enc/ tree (csv/log/run_dataset) under the shared
../results/results_star/ directory, mirroring run_ckks.py's results_ckks/
tree, instead of the defaults in grid_search_config.json — so it never
writes into the same CSV file as run_ckks.py when the two run concurrently,
and so the global CSV itself is named distinctly from run_ckks.py's (via
csv_tag) in case a file is ever copied out of its results_no_enc/csv/
directory.

Meant to be launched alongside run_ckks.py in a separate process so the two
encryption workloads can progress in parallel without contending for the
same GPU-assignment/port ranges or writing to the same CSV file.
"""

import multiprocessing

import federated_grid_search as fgs

ENCRYPTION_MODE = 'no_encryption'

# Port block base so this script can run alongside run_ckks.py (base 26000)
# without conflicting. This is only a starting-point hint for
# find_free_port_pair — the real bind-retry in TrustedAuthority/FederatedServer
# (utils.run_with_port_retry) is what actually guarantees correctness on a
# node that may be shared with other users' processes.
_PORT_BASE = 25000


def run_all():
    # common_search_space is read fresh from the JSON and narrowed to a
    # single encryption_mode here, rather than edited in the JSON itself, so
    # the same grid_search_config.json keeps describing the full search space
    # and both run_no_encryption.py / run_ckks.py stay simple slices of it.
    base_config = fgs.load_json(fgs.GRID_SEARCH_CONFIG_PATH)
    common_space = dict(base_config['common_search_space'])
    common_space['encryption_mode'] = [ENCRYPTION_MODE]
    # he_backend is meaningless for no_encryption (never read by
    # TrustedAuthority/FederatedServer when encryption is off); pinned here
    # to a single value for consistency with run_ckks.py.
    common_space['he_backend'] = ['ckks']

    banner = f"  ENCRYPTION: {ENCRYPTION_MODE}  "
    line = "#" * (len(banner) + 4)
    print(f"\n{line}\n# {banner} #\n{line}\n")

    fgs.main(override_config={
        'common_search_space': common_space,
        'port': _PORT_BASE,
        'base_csv_path': '../results/results_star/results_no_enc/csv',
        'base_log_path': '../results/results_star/results_no_enc/log',
        'base_split_data_path': '../results/results_star/results_no_enc/run_dataset',
        # Distinguishes this run's global CSV filename from run_ckks.py's
        # even if a file is copied out of results_no_enc/csv/ (see csv_tag
        # in federated_grid_search.main).
        'csv_tag': 'no_enc',
    })
    print(f"=== {ENCRYPTION_MODE} run complete. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    run_all()