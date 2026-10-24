import os
import re
import shutil
import numpy as np
import pandas as pd
from typing import List, Dict, Sequence


# Multiple Sclerosis slices are named '<patient>_<sequence>_slice<k>_EDSS<s>[_aug<j>].png'
# (e.g. '11_Flair_slice0_EDSS2_aug1.png'): the leading integer is the patient
# identifier, shared by every MRI sequence and by every augmented copy of a slice.
PATIENT_ID_PATTERN = re.compile(r'^(\d+)_(?:Flair|T1|T2)_slice\d+', re.IGNORECASE)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# Train / validation / test fractions of every client-local dataset (Section 5.1).
DEFAULT_SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ('train', 'valid', 'test')


class DatasetSplitter:
    """
    Splits an image dataset into N disjoint client-local datasets for federated learning.

    The partitioning follows Section 5.1 of the paper:
      * Classes are inferred from the directory structure (one sub-folder per class).
      * When the filenames carry a patient identifier (Multiple Sclerosis collections),
        the partitioning is performed at patient level: all the slices of a patient,
        together with all the augmented copies derived from them, form a single unit,
        so that no patient is shared by two clients or by the train/valid/test subsets
        of the same client. Collections without a patient identifier are partitioned
        with the same procedure at image level (each image is a unit).
      * Units are assigned through a greedy bin-packing heuristic: they are visited
        from the largest to the smallest, and each one is placed in the bin where the
        insertion least increases the deviation of the per-class image counts from
        their target values, with ties broken pseudo-randomly.
      * Each client-local dataset is then divided, with the same procedure, into a
        training (70%), a validation (15%), and a test (15%) subset.
    """

    def __init__(self, output_base_dir: str, source_images_dir: str, num_clients: int, seed: int = 42):
        """
        Args:
            output_base_dir (str): Root directory where client data folders will be created.
            source_images_dir (str): Directory containing source images, with one subdirectory per class.
            num_clients (int): Number of clients to split the data for.
            seed (int): Seed of the pseudo-random tie-breaking.
        """
        self.output_base_dir = output_base_dir
        self.source_images_dir = source_images_dir
        self.num_clients = num_clients
        self.rng = np.random.RandomState(seed)
        self.class_map: Dict[str, int] = {}
        self.index_to_dir_map: Dict[int, str] = {}

        os.makedirs(self.output_base_dir, exist_ok=True)
        self.dataframe = self._build_dataframe_from_folders()

        if self.dataframe.empty:
            raise ValueError(f"No images found in source directory: {self.source_images_dir}")

        self.num_classes = len(self.class_map)
        self.patient_level = self._assign_units()

    # ------------------------------------------------------------------
    # Dataset scanning
    # ------------------------------------------------------------------

    def _build_dataframe_from_folders(self) -> pd.DataFrame:
        """
        Scans the source directory to build a dataframe of filenames and class labels.
        Class indices are assigned based on numeric values in directory names when possible,
        otherwise falling back to alphabetical order.
        """
        print(f"Scanning '{self.source_images_dir}' to infer classes from directories...")

        try:
            class_dirs = [
                d for d in os.listdir(self.source_images_dir)
                if os.path.isdir(os.path.join(self.source_images_dir, d))
            ]
        except FileNotFoundError:
            print(f"Error: Source image directory not found at {self.source_images_dir}")
            raise

        if not class_dirs:
            print(f"Error: No class subdirectories found in {self.source_images_dir}")
            return pd.DataFrame()

        # Attempt to map classes using numeric values found in directory names
        numeric_map = {}
        can_use_numeric_map = True
        for dir_name in class_dirs:
            match = re.search(r'\d+', dir_name)
            if match:
                numeric_map[dir_name] = int(match.group(0))
            else:
                can_use_numeric_map = False
                break

        if can_use_numeric_map and len(set(numeric_map.values())) == len(class_dirs):
            print("Using numeric values found in directory names for class mapping.")
            # Re-index to 0..J-1 so class indices can address per-class count vectors.
            ordered = sorted(class_dirs, key=lambda d: numeric_map[d])
            self.class_map = {dir_name: i for i, dir_name in enumerate(ordered)}
        else:
            print("Could not infer classes from numbers in directory names. Falling back to alphabetical order.")
            sorted_dirs = sorted(class_dirs)
            self.class_map = {dir_name: i for i, dir_name in enumerate(sorted_dirs)}

        print("--- Class Mapping Detected ---")
        for dir_name, class_idx in self.class_map.items():
            print(f"  '{dir_name}' -> Class {class_idx}")
        print("----------------------------")

        # Build reverse mapping from class index to directory name
        self.index_to_dir_map = {v: k for k, v in self.class_map.items()}

        # Populate the dataframe with filename and class label for each image
        records = []
        for dir_name, class_idx in self.class_map.items():
            class_path = os.path.join(self.source_images_dir, dir_name)
            for filename in sorted(os.listdir(class_path)):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    records.append({'filename': filename, 'class': class_idx})

        return pd.DataFrame(records)

    def _assign_units(self) -> bool:
        """
        Assigns every image to its partitioning unit: the patient identifier when all
        the filenames carry one, otherwise the image itself. Returns True for
        patient-level partitioning.
        """
        patient_ids = self.dataframe['filename'].map(
            lambda f: (m.group(1) if (m := PATIENT_ID_PATTERN.match(f)) else None)
        )
        if patient_ids.notna().all():
            self.dataframe['unit'] = 'patient_' + patient_ids
            print(f"Patient identifiers found: patient-level partitioning "
                  f"({self.dataframe['unit'].nunique()} patients).")
            return True

        # Filenames may repeat across class folders (e.g. '1.png'), so the class is
        # part of the image-level unit key.
        self.dataframe['unit'] = self.dataframe['class'].astype(str) + '/' + self.dataframe['filename']
        print("No patient identifier in filenames: image-level partitioning.")
        return False

    # ------------------------------------------------------------------
    # Greedy bin-packing
    # ------------------------------------------------------------------

    def _unit_class_counts(self, unit_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        """Per-class image counts of each unit."""
        subset = self.dataframe[self.dataframe['unit'].isin(set(unit_ids))]
        counts = {}
        for unit, group in subset.groupby('unit'):
            counts[unit] = np.bincount(group['class'].to_numpy(), minlength=self.num_classes)
        return counts

    def _greedy_partition(self, unit_ids: Sequence[str], targets: np.ndarray) -> List[List[str]]:
        """
        Greedy bin-packing of units into len(targets) bins.

        Units are visited from the one with the most images to the one with the fewest
        (ties in size are shuffled), and each is placed in the bin where the insertion
        least increases the L1 deviation of the per-class counts from the bin's targets.
        Ties between bins are broken pseudo-randomly.

        Args:
            unit_ids: units to distribute.
            targets: array of shape (num_bins, num_classes) with the target per-class counts.
        """
        unit_counts = self._unit_class_counts(unit_ids)
        units = list(unit_counts.keys())
        self.rng.shuffle(units)
        units.sort(key=lambda u: unit_counts[u].sum(), reverse=True)  # stable: shuffled ties

        num_bins = targets.shape[0]
        bin_counts = np.zeros_like(targets, dtype=float)
        bins: List[List[str]] = [[] for _ in range(num_bins)]

        for unit in units:
            u = unit_counts[unit]
            current_dev = np.abs(bin_counts - targets).sum(axis=1)
            new_dev = np.abs(bin_counts + u - targets).sum(axis=1)
            increase = new_dev - current_dev
            candidates = np.flatnonzero(np.isclose(increase, increase.min()))
            chosen = int(self.rng.choice(candidates))
            bins[chosen].append(unit)
            bin_counts[chosen] += u

        # With few units (e.g. ~7 patients per client when N=8), a bin with a small
        # target can stay empty. Each empty bin then receives the smallest unit of the
        # bin that exceeds its target the most, so that every client gets non-empty
        # training, validation, and test subsets.
        target_sizes = targets.sum(axis=1)
        for empty in [b for b in range(num_bins) if not bins[b]]:
            donors = [b for b in range(num_bins) if len(bins[b]) > 1]
            if not donors:
                break
            donor = max(donors, key=lambda b: bin_counts[b].sum() - target_sizes[b])
            unit = min(bins[donor], key=lambda x: unit_counts[x].sum())
            bins[donor].remove(unit)
            bins[empty].append(unit)
            bin_counts[donor] -= unit_counts[unit]
            bin_counts[empty] += unit_counts[unit]

        return bins

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _copy_units(self, unit_ids: Sequence[str], client_dir: str, split_name: str) -> int:
        """
        Copies the images of the given units into client_dir/split_name/<class_dir>/,
        preserving the original class subdirectory structure. Every class folder is
        created even when empty, so that ImageFolder assigns the same class indices
        on every client and split.
        """
        for class_dir_name in self.class_map:
            os.makedirs(os.path.join(client_dir, split_name, class_dir_name), exist_ok=True)

        rows = self.dataframe[self.dataframe['unit'].isin(set(unit_ids))]
        for _, row in rows.iterrows():
            class_dir_name = self.index_to_dir_map[row['class']]
            source_image_path = os.path.join(self.source_images_dir, class_dir_name, row['filename'])
            destination_image_path = os.path.join(client_dir, split_name, class_dir_name, row['filename'])

            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, destination_image_path)
            else:
                print(f"Warning: Source image not found and will be skipped: {source_image_path}")
        return len(rows)

    def _clear_existing_split(self) -> None:
        """
        Deletes all contents of the output directory to ensure a clean split.
        Warning: This is a destructive operation.
        """
        print(f"Clearing existing data in '{self.output_base_dir}'...")
        if os.path.exists(self.output_base_dir):
            for item_name in os.listdir(self.output_base_dir):
                item_path = os.path.join(self.output_base_dir, item_name)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  - Deleted directory: {item_path}")
        print("Clearing complete.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_partition(self, split_fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS
                          ) -> List[Dict[str, List[str]]]:
        """
        Computes the partition without touching the filesystem.

        Returns one dict per client mapping 'train'/'valid'/'test' to the list of units.
        """
        fractions = np.asarray(split_fractions, dtype=float)
        if len(fractions) != len(SPLIT_NAMES) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError(f"split_fractions must contain three values summing to 1, got {split_fractions}.")

        # Stage 1: units -> N clients, target |D|_c / N for every client.
        class_totals = np.bincount(self.dataframe['class'].to_numpy(), minlength=self.num_classes)
        client_targets = np.tile(class_totals / self.num_clients, (self.num_clients, 1))
        all_units = self.dataframe['unit'].unique().tolist()
        client_units = self._greedy_partition(all_units, client_targets)

        # Stage 2: each client -> train / valid / test, target fraction × client class counts.
        partition = []
        for units in client_units:
            client_totals = sum(self._unit_class_counts(units).values(), np.zeros(self.num_classes))
            split_targets = np.outer(fractions, client_totals)
            split_units = self._greedy_partition(units, split_targets)
            partition.append(dict(zip(SPLIT_NAMES, split_units)))
        return partition

    def split_dataset(self, split_fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS) -> None:
        """
        Performs the main dataset splitting operation.

        Clears any previous splits, partitions the units among the clients, divides each
        client-local dataset into train/valid/test subsets, and copies the images to
        output_base_dir/client_<i>/{train,valid,test}/<class_dir>/.
        """
        self._clear_existing_split()
        partition = self.compute_partition(split_fractions)

        for i, client_splits in enumerate(partition):
            client_dir = os.path.join(self.output_base_dir, f"client_{i}")
            os.makedirs(client_dir, exist_ok=True)
            print(f"Processing data for client_{i}...")
            for split_name in SPLIT_NAMES:
                n_images = self._copy_units(client_splits[split_name], client_dir, split_name)
                print(f"  - Created {split_name} set with {n_images} samples "
                      f"({len(client_splits[split_name])} units).")
