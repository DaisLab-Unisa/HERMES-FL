import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
import time
import numpy as np
from typing import Tuple, Dict, List


# Lightweight ConvNet baseline of [11]: one 7x7 convolution (4 filters, stride 3)
# followed by three fully connected layers, trained from scratch on grayscale inputs.
class ConvNet(torch.nn.Module):
    def __init__(self, image_size=32, hidden1=64, hidden2=32, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        conv_out_size = ((image_size - 7) // 3) + 1
        flattened_size = 4 * conv_out_size * conv_out_size
        self.fc1 = torch.nn.Linear(flattened_size, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelManager:
    """
    Handles the creation, local training, and evaluation of the client models.

    Pre-trained backbones are frozen and only the classification head is trained
    and exchanged; the ConvNet baseline is trained and exchanged in full.
    """

    def __init__(self, config: Dict, dataset_path: str):
        self.config = config
        self.model_name = config.get('model_name', 'ResNet18')
        self.dataset_path = dataset_path
        self.num_classes = config.get('num_classes', 2)
        self.device = self._get_device(config.get('device', 'cuda'))
        self.model = self._initialize_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.transform_pipeline = self._get_transforms()
        self.calibration_term = torch.zeros(self.num_classes, device=self.device)

    def _get_device(self, device_str: str) -> torch.device:
        if device_str == 'cuda' and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            blacklist = self.config.get('gpu_blacklist', [])
            usable_gpus = [i for i in range(num_gpus) if i not in blacklist]
            if usable_gpus:
                worker_id = self.config.get('worker_id', 0) or 0
                gpu_index = usable_gpus[worker_id % len(usable_gpus)]
                return torch.device(f"cuda:{gpu_index}")
        return torch.device("cpu")

    def _get_transforms(self):
        image_size = self.config.get('image_size', 224)
        if self.model_name == 'ConvNet':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        else:  # Pre-trained backbones (ResNet, GoogLeNet, AlexNet): ImageNet normalization
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _initialize_model(self) -> nn.Module:
        print(f"Initializing model: {self.model_name}")
        if 'ResNet' in self.model_name:
            return self._initialize_resnet()
        elif self.model_name == 'GoogLeNet':
            return self._initialize_googlenet()
        elif self.model_name == 'AlexNet':
            return self._initialize_alexnet()
        elif self.model_name == 'ConvNet':
            return self._initialize_convnet()
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

    def _initialize_convnet(self) -> nn.Module:
        return ConvNet(
            image_size=self.config.get('image_size', 32),
            hidden1=self.config.get('convnet_hidden1', 64),
            hidden2=self.config.get('convnet_hidden2', 32),
            num_classes=self.num_classes
        )

    def _initialize_alexnet(self) -> nn.Module:
        """Initializes an AlexNet model with correct classifier handling."""
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        num_custom_layers = self.config.get('num_custom_layers', 2)
        train_full_model = (num_custom_layers == 0)

        if train_full_model:
            print("num_custom_layers is 0. All AlexNet parameters will be trainable.")
        else:
            print("Freezing pre-trained AlexNet layers.")

        # Freeze (or unfreeze) the convolutional feature extractor
        for param in model.features.parameters():
            param.requires_grad = train_full_model

        if num_custom_layers > 0:
            # The custom head replaces the whole original classifier and operates on the
            # flattened 256 x 6 x 6 feature map (d_feat = 9216).
            in_features_for_block = 256 * 6 * 6
            custom_classifier = self._create_custom_classifier(in_features_for_block, self.num_classes,
                                                               num_custom_layers)
            model.classifier = nn.Sequential(*custom_classifier)
        else:
            # The last original layer (index 6) has in_features=4096
            last_layer_in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(last_layer_in_features, self.num_classes)

        for param in model.classifier.parameters():
            param.requires_grad = True

        return model


    def _initialize_googlenet(self) -> nn.Module:
        """Initializes a GoogLeNet model, with logic for custom layers vs full fine-tuning."""
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features  # 1024 for GoogLeNet
        num_custom_layers = self.config.get('num_custom_layers', 2)
        train_full_model = (num_custom_layers == 0)

        if train_full_model:
            print("num_custom_layers is 0. All GoogLeNet parameters will be trainable.")
        else:
            print(f"Freezing pre-trained GoogLeNet layers. Only the custom classifier will be trained.")

        for param in model.parameters():
            param.requires_grad = train_full_model

        if num_custom_layers > 0:
            layers = self._create_custom_classifier(num_features, self.num_classes, num_custom_layers)
            model.fc = nn.Sequential(*layers)
        else:
            model.fc = nn.Linear(num_features, self.num_classes)

        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def _initialize_resnet(self) -> nn.Module:
        if self.model_name == 'ResNet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif self.model_name == 'ResNet34':
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif self.model_name == 'ResNet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif self.model_name == 'ResNet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"ResNet model '{self.model_name}' not supported.")

        num_features = model.fc.in_features
        num_custom_layers = self.config.get('num_custom_layers', 2)
        train_full_model = (num_custom_layers == 0)

        if train_full_model:
            print("num_custom_layers is 0. All ResNet parameters will be trainable.")
        else:
            print(f"Freezing pre-trained ResNet layers. Only the custom classifier will be trained.")

        for param in model.parameters():
            param.requires_grad = train_full_model

        if num_custom_layers > 0:
            layers = self._create_custom_classifier(num_features, self.num_classes, num_custom_layers)
            model.fc = nn.Sequential(*layers)
        else:
            model.fc = nn.Linear(num_features, self.num_classes)

        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def _create_custom_classifier(self, in_features, out_features, num_layers):
        layers = []
        if num_layers < 2:
            layers.append(nn.Linear(in_features, out_features))
        else:
            hidden_size = 256
            layers.extend([nn.Linear(in_features, hidden_size), nn.ReLU(), nn.Dropout(0.4)])
            in_size = hidden_size
            for _ in range(num_layers - 2):
                out_size = in_size // 2
                layers.extend([nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(0.4)])
                in_size = out_size
            layers.append(nn.Linear(in_size, out_features))
        return layers

    def _get_trainable_parameters(self) -> List[torch.Tensor]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_weights(self) -> List[np.ndarray]:
        return [param.data.cpu().numpy() for param in self._get_trainable_parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        trainable_params = self._get_trainable_parameters()
        for param, weight_array in zip(trainable_params, weights):
            param.data.copy_(torch.from_numpy(weight_array))

    def get_samples_per_class(self) -> np.ndarray:
        samples = torch.zeros(self.num_classes, device=self.device)
        try:
            train_loader = self._get_dataloader('train', batch_size=32)
            for _, labels in train_loader:
                for label in labels:
                    samples[label.item()] += 1
        except FileNotFoundError:
            print("Warning: Training data not found during sample count.")
        return samples.cpu().numpy()

    def set_calibration_term(self, calibration_val: np.ndarray):
        self.calibration_term = torch.from_numpy(calibration_val).float().to(self.device)

    def _get_dataloader(self, split: str, batch_size: int) -> DataLoader:
        data_path = os.path.join(self.dataset_path, split)
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Dataset directory not found for split '{split}': {data_path}")
        # Every split contains all the class folders (possibly empty, see DatasetSplitter),
        # so class indices are identical on every client and split.
        dataset = ImageFolder(root=data_path, transform=self.transform_pipeline, allow_empty=True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

    def train(self, epochs: int, lr: float, batch_size: int,
              algorithm: str = "FedAvg", global_weights: List[np.ndarray] = None,
              mu: float = 0.0) -> Tuple[float, Dict, float, int]:
        trainable_params = self._get_trainable_parameters()
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
        try:
            train_loader = self._get_dataloader('train', batch_size)
        except FileNotFoundError as e:
            import logging as _log
            _log.getLogger(__name__).error("train() aborted: %s", e)
            return 0.0, {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}, 0.0, 0

        global_params_tensor = None
        if algorithm == 'FedProx' and global_weights is not None:
            global_params_tensor = [torch.from_numpy(w).to(self.device) for w in global_weights]

        final_train_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = self._run_training_epoch(train_loader, optimizer, algorithm, global_params_tensor, mu)
            final_train_loss = epoch_loss
            scheduler.step(epoch_loss)

        _, metric_score, _, dataset_size = self.validate(batch_size, split='train')
        return 0.0, metric_score, final_train_loss, dataset_size

    def _run_training_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                            algorithm: str, global_params: List[torch.Tensor], mu: float) -> float:
        """Train for one epoch. Uses safe iteration to skip batches with missing files."""
        import logging as _log
        self.model.train()
        running_loss = 0.0
        processed = 0
        # Use explicit iterator so we can catch errors from __next__ (file not found during load)
        loader_iter = iter(loader)
        while True:
            try:
                inputs, labels = next(loader_iter)
            except StopIteration:
                break
            except (FileNotFoundError, OSError) as _file_err:
                _log.getLogger(__name__).warning("Skipping batch (file missing during load): %s", _file_err)
                continue
            try:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                if algorithm == 'FedLC':
                    loss = self.criterion(outputs - self.calibration_term.unsqueeze(0), labels)
                else:
                    loss = self.criterion(outputs, labels)

                if algorithm == 'FedProx' and mu > 0 and global_params:
                    # Proximal term of Eq. (7): (mu / 2) * ||w - w^(t)||_2^2
                    prox_term = 0.0
                    local_params = self._get_trainable_parameters()
                    for local_w, global_w in zip(local_params, global_params):
                        prox_term += (local_w - global_w).pow(2).sum()
                    loss += (mu / 2) * prox_term

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                processed += inputs.size(0)
            except (FileNotFoundError, OSError) as _file_err:
                _log.getLogger(__name__).warning("Skipping batch (I/O error during forward): %s", _file_err)
        return running_loss / max(processed, 1)

    def evaluate_global_model(self, batch_size: int) -> Dict[str, float]:
        """Evaluates the current (global) weights on the validation and test subsets.

        Returns the payload sent to the aggregation tier: '<split>_{loss,f1,acc,prec,recall,size}'
        for split in ('val', 'test').
        """
        result = {}
        for prefix, split in (('val', 'valid'), ('test', 'test')):
            loss, metrics, _, size = self.validate(batch_size, split=split)
            result.update({
                f'{prefix}_loss': loss,
                f'{prefix}_f1': metrics['f1_score'],
                f'{prefix}_acc': metrics['accuracy'],
                f'{prefix}_prec': metrics['precision'],
                f'{prefix}_recall': metrics['recall'],
                f'{prefix}_size': size,
            })
        return result

    def validate(self, batch_size: int, split: str = 'valid') -> Tuple[float, Dict, float, int]:
        try:
            data_loader = self._get_dataloader(split, batch_size)
        except FileNotFoundError:
            return 0.0, {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}, 0.0, 0

        self.model.eval()
        all_preds, all_labels = [], []
        running_loss = 0.0

        algorithm = self.config.get('aggregation_algorithm', 'FedAvg')
        import logging as _log
        with torch.no_grad():
            loader_iter = iter(data_loader)
            while True:
                try:
                    images, labels = next(loader_iter)
                except StopIteration:
                    break
                except (FileNotFoundError, OSError) as _file_err:
                    _log.getLogger(__name__).warning(
                        "Skipping batch in validate(%s) — file missing: %s", split, _file_err)
                    continue
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    if algorithm == 'FedLC':
                        loss = self.criterion(outputs - self.calibration_term.unsqueeze(0), labels)
                    else:
                        loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except (FileNotFoundError, OSError) as _file_err:
                    _log.getLogger(__name__).warning(
                        "Skipping batch in validate(%s) — I/O error: %s", split, _file_err)

        if not all_labels:
            return 0.0, {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}, 0.0, 0

        val_loss = running_loss / len(all_labels)
        metrics = {
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        return val_loss, metrics, 0.0, len(all_labels)