import time
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from model_manager import ModelManager
from utils import ckks_sum_weights, ckks_multiply_weights_by_scalar


class Aggregator:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.current_weights = self._initialize_weights()
        self.metrics_history: Dict[int, Dict] = {}

        # Model selection (Section 6.3): the best round maximizes the global
        # VALIDATION F1-score, and the TEST metrics of that round's global model are
        # reported, so that test data never influence selection.
        # best_val_f1 starts at -1.0 so that the very first evaluation round
        # (even with f1 == 0.0) is always selected — a model that hasn't
        # converged yet in early rounds shouldn't be skipped.
        self.best_val_f1: float = -1.0
        self.best_val_accuracy: float = 0.0
        self.best_val_precision: float = 0.0
        self.best_val_recall: float = 0.0
        self.best_val_loss: float = float('inf')
        self.best_model_weights: List[np.ndarray] = self.current_weights
        self.best_round: int = -1
        self.test_at_best: Dict[str, float] = {
            'test_f1': 0.0, 'test_acc': 0.0, 'test_prec': 0.0, 'test_recall': 0.0, 'test_loss': float('inf')
        }

        # Early stopping on the global validation loss
        self.prev_val_loss: float = None
        self.early_stop_counter: int = 0

        self.training_start_time: float = time.time()
        self.run_summary: Dict = None

        print(f"\n[START] {self.config.get('model_name', '?')} | {self._config_line()}\n")

    def _config_line(self) -> str:
        c = self.config
        parts = [
            f"Algo: {c.get('aggregation_algorithm', '?')}",
            f"GlobalEpoch: {c.get('global_epoch', '?')}",
            f"LocalEpoch: {c.get('local_epoch', '?')}",
            f"Clients: {c.get('num_clients', '?')}",
            f"LR: {c.get('learning_rate', '?')}",
            f"Batch: {c.get('batch_size', '?')}",
            f"Encrypt: {c.get('encryption_mode', 'no_encryption')}",
        ]
        if c.get('num_edge_servers'):
            parts.append(f"Edges: {c['num_edge_servers']}")
        if c.get('edge_rounds'):
            parts.append(f"EdgeRounds: {c['edge_rounds']}")
        if c.get('aggregation_algorithm') == 'FedProx':
            parts.append(f"mu: {c.get('fedprox_mu', '?')}")
        return " | ".join(parts)

    def _initialize_weights(self) -> List[np.ndarray]:
        """Initializes model weights by instantiating a temporary ModelManager."""
        self.logger.info("Initializing model weights...")
        temp_model = ModelManager(config=self.config, dataset_path="")
        initial_weights = temp_model.get_weights()
        self.logger.info(f"Trainable parameters initialized for model {self.config['model_name']}.")
        del temp_model
        return initial_weights

    def aggregate_weights(self, client_updates: List[Dict], algorithm: str) -> bool:
        """
        Plaintext baseline of the aggregation: data-weighted average of the received
        weights, w^(t+1) = Σ (Q / Q̄) · w with Q̄ = Σ Q (Eqs. 3 and 5 without encryption).
        FedAvg, FedProx, and FedLC share the same aggregation and differ only in the
        local objective minimized by the clients.
        """
        self.logger.info(f"Aggregating weights using '{algorithm}'...")
        if not client_updates:
            self.logger.warning("No client updates received for aggregation.")
            return False

        client_weights = [u['weights'] for u in client_updates]
        client_sizes = [u['train_size'] for u in client_updates]

        total_size = sum(client_sizes)
        if total_size == 0:
            self.logger.warning("Total training size is 0. Skipping aggregation.")
            return False

        if algorithm in ("FedAvg", "FedProx", "FedLC"):
            fractions = [s / total_size for s in client_sizes]
            averaged_weights = [w * fractions[0] for w in client_weights[0]]
            for client_idx in range(1, len(client_weights)):
                for i in range(len(averaged_weights)):
                    averaged_weights[i] += client_weights[client_idx][i] * fractions[client_idx]

            self.current_weights = averaged_weights
            self.logger.info(f"Weighted average ({algorithm}) complete. Total size: {total_size}.")
            return True

        else:
            raise ValueError(
                f"Aggregation algorithm '{algorithm}' is not supported. "
                f"Choose from: FedAvg, FedProx, FedLC."
            )

    def aggregate_encrypted_updates_ckks(self, updates: List[Dict], context) -> None:
        """CKKS aggregation (AggregateEncrypted routine of Algorithm 1).

        Each update is a pair (E(v), Q): an encrypted weight vector and its number of
        training samples. The weighted average is computed in the encrypted domain:
            E(v̄) = Σ E(v) ⊙ (Q / Q̄),   Q̄ = Σ Q
        with plaintext-ciphertext scalar multiplications (one CKKS level) and
        ciphertext additions. In the star topology the pairs are the client updates
        (Eq. 3); in the hierarchical topology they are the edge cluster averages (Eq. 5).
        The server holds only the public context and never decrypts.
        """
        total_N = sum(u['train_size'] for u in updates)
        if total_N == 0:
            self.logger.warning("CKKS aggregation: total_N=0, skipping.")
            return

        fractions = [u['train_size'] / total_N for u in updates]
        summed = ckks_multiply_weights_by_scalar(updates[0]['weights'], fractions[0], context)
        for i in range(1, len(updates)):
            weighted_i = ckks_multiply_weights_by_scalar(updates[i]['weights'], fractions[i], context)
            summed = ckks_sum_weights(summed, weighted_i, context)
        self.current_weights = summed
        self.logger.info("CKKS encrypted aggregation (homomorphic global weighted average) complete.")

    def aggregate_train_loss_weighted(self, client_losses: List[float], client_sizes: List[int], current_round: int):
        """Calculates and stores the weighted average of training losses from clients."""
        total_size = sum(client_sizes)
        if total_size == 0:
            return

        weighted_loss = np.average(client_losses, weights=client_sizes)
        self.logger.info(f"Round {current_round} - Weighted average training loss: {weighted_loss:.4f}")

        if current_round not in self.metrics_history:
            self.metrics_history[current_round] = {}
        self.metrics_history[current_round]['train_loss'] = weighted_loss

    @staticmethod
    def _weighted_split_metrics(eval_updates: List[Dict], prefix: str) -> Dict[str, float]:
        """Eq. (11): average of the client metrics on one split ('val' or 'test'),
        weighted by the size of the clients' evaluation subsets."""
        sizes = [u[f'{prefix}_size'] for u in eval_updates]
        if sum(sizes) == 0:
            return {}
        return {
            f'{prefix}_{m}': float(np.average([u[f'{prefix}_{m}'] for u in eval_updates], weights=sizes))
            for m in ('loss', 'f1', 'acc', 'prec', 'recall')
        }

    def aggregate_evaluation_results(self, eval_updates: List[Dict], current_round: int) -> bool:
        """Aggregates the validation and test metrics of the global model, selects the best
        round on validation F1, and checks the early stopping criterion on validation loss."""
        val = self._weighted_split_metrics(eval_updates, 'val')
        if not val:
            return False
        test = self._weighted_split_metrics(eval_updates, 'test')

        self.logger.info(
            f"--- Round {current_round} Evaluation: Val Loss {val['val_loss']:.4f}, "
            f"Val F1 {val['val_f1']:.4f}, Test F1 {test.get('test_f1', float('nan')):.4f} ---")

        self.metrics_history.setdefault(current_round, {}).update({**val, **test})

        # Model selection on the global validation F1-score
        if val['val_f1'] > self.best_val_f1:
            self.best_val_f1, self.best_val_accuracy = val['val_f1'], val['val_acc']
            self.best_val_precision, self.best_val_recall = val['val_prec'], val['val_recall']
            self.best_val_loss = val['val_loss']
            self.best_model_weights = self.current_weights
            self.best_round = current_round
            if test:
                self.test_at_best = test

        # Early stopping: global validation loss increasing for `early_stop_patience` consecutive rounds
        if self.prev_val_loss is not None and val['val_loss'] > self.prev_val_loss:
            self.early_stop_counter += 1
        else:
            self.early_stop_counter = 0

        self.prev_val_loss = val['val_loss']
        return self.early_stop_counter >= self.config['early_stop_patience']

    def log_best_model_stats(self):
        """Logs a summary of the selected round."""
        self.logger.info("=" * 30)
        self.logger.info("Federated Training Finished")
        self.logger.info(f"Best round (validation F1): {self.best_round} | Val F1: {self.best_val_f1:.4f} | "
                         f"Test F1: {self.test_at_best['test_f1']:.4f} | "
                         f"Test Acc: {self.test_at_best['test_acc']:.4f} | "
                         f"Test Loss: {self.test_at_best['test_loss']:.4f}")
        self.logger.info("=" * 30)

    def get_stats(self) -> pd.DataFrame:
        """Converts the metrics history dictionary to a Pandas DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self.metrics_history, orient='index')

    def save_results(self) -> None:
        """Saves metrics to CSV and prepares the final run summary dictionary."""
        metrics_dir = self.config['run_metrics_output_path']
        base_csv_dir = self.config.get('base_csv_path', 'csv')
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(base_csv_dir, exist_ok=True)

        timestr = time.strftime('%Y%m%d-%H%M%S')
        metrics_df = self.get_stats()
        worker_id = self.config.get('worker_id', 'NA')
        dataset_name = self.config.get('dataset_name', 'unknown')

        # Save the standard round-by-round CSV
        filename = os.path.join(metrics_dir, f"{timestr}_{dataset_name}_worker_{worker_id}.csv")
        metrics_df.to_csv(filename)
        self.logger.info(f"Saved round metrics to {filename}")

        # Prepare the summary dictionary with best values and hyperparameters
        self.run_summary = self.config.copy()

        # convnet_hidden1/2 only exist in the search space for ConvNet — default
        # them to -1 for every other model so the column stays populated (and
        # comparable) across the whole global summary CSV instead of being blank.
        self.run_summary.setdefault('convnet_hidden1', -1)
        self.run_summary.setdefault('convnet_hidden2', -1)

        self.run_summary.update({
            'best_round': self.best_round,
            # Validation metrics of the selected round (used for model selection)
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_accuracy,
            'best_val_prec': self.best_val_precision,
            'best_val_recall': self.best_val_recall,
            'best_val_loss': self.best_val_loss,
            # Test metrics of the selected round's global model (reported results)
            **self.test_at_best,
            'total_duration_sec': time.time() - self.training_start_time,
            'round_dataframe_path': filename,
        })

        model_name = self.config.get('model_name', '?')
        dur = self.run_summary['total_duration_sec']
        test_loss = self.test_at_best['test_loss']
        loss_str = f"{test_loss:.4f}" if test_loss != float('inf') else "N/A"
        print(
            f"\n[END]   {model_name}"
            f" | Best round: {self.best_round}"
            f" | Val F1: {self.best_val_f1:.4f}"
            f" | Test F1: {self.test_at_best['test_f1']:.4f}"
            f" | Test Acc: {self.test_at_best['test_acc']:.4f}"
            f" | Test Prec: {self.test_at_best['test_prec']:.4f}"
            f" | Test Recall: {self.test_at_best['test_recall']:.4f}"
            f" | Test Loss: {loss_str}"
            f" | Duration: {dur:.0f}s"
            f"\n         Config  {self._config_line()}\n"
        )

        # Cleanup: remove unnecessary paths or overly verbose dictionaries from the summary
        keys_to_remove = ['shared_csv_path', 'run_metrics_output_path', 'base_csv_path',
                          'base_log_path', 'base_plot_path', 'base_split_data_path',
                          'ip_address', 'port', 'ta_port', 'splitting_dir',
                          'dataset_path', 'csv_lock', 'log_dir', 'weighted_aggregation',
                          'MIN_NUM_WORKERS', 'central_port', 'edge_base_port', 'edge_port',
                          'server_run_id',
                          'training_semaphore', 'round_timeout_sec',
                          # Bookkeeping/infra fields — not useful in the results CSV.
                          'gpu_blacklist', 'heavy_client_threshold', 'csv_tag']

        for key in keys_to_remove:
            self.run_summary.pop(key, None)

        # csv_tag mirrors federated_grid_search.main()'s naming so entrypoints
        # that share a search space but write to different result trees (e.g.
        # run_ckks.py / run_no_encryption.py) append to the same tagged file
        # that the dedup/fingerprint pass reads from, instead of silently
        # creating an untagged sibling file.
        csv_tag = self.config.get('csv_tag', '')
        tag_suffix = f"_{csv_tag}" if csv_tag else ''
        csv_name = f"global_grid_search_summary{tag_suffix}.csv"
        global_csv_path = os.path.join(base_csv_dir, csv_name)
        summary_df = pd.DataFrame([self.run_summary])

        csv_lock = self.config.get('csv_lock')
        if csv_lock:
            csv_lock.acquire()

        try:
            if not os.path.exists(global_csv_path):
                # Create a new file if it does not exist
                summary_df.to_csv(global_csv_path, index=False)
            else:
                # Read the existing one, merge data, and overwrite to align columns correctly
                existing_df = pd.read_csv(global_csv_path)
                combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
                combined_df.to_csv(global_csv_path, index=False)

            self.logger.info(f"Saved global execution summary to {global_csv_path}")
        except Exception as e:
            self.logger.error(f"Failed to append to global summary CSV: {e}")
        finally:
            if csv_lock:
                csv_lock.release()

    def get_run_summary(self) -> Dict:
        return self.run_summary

    def get_parameter_plots(self) -> List[str]:
        """Returns a list of plot paths (Visualization logic currently disabled)."""
        return []