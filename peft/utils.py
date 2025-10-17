"""Utilities for LoRA training pipeline."""

import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import json
import wandb
from PIL import Image
import torchvision.transforms as transforms
import math
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length tile_paths lists."""
    tile_paths = [item['tile_paths'] for item in batch]
    batch_without_paths = []
    for item in batch:
        new_item = {k: v for k, v in item.items() if k != 'tile_paths'}
        batch_without_paths.append(new_item)
    collated_batch = default_collate(batch_without_paths)
    collated_batch['tile_paths'] = tile_paths
    return collated_batch


class EAGLEFeaturesDataset(Dataset):
    """Dataset for EAGLE features with labels."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, slide_ids: List[str]):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.slide_ids = slide_ids
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'slide_id': self.slide_ids[idx],
            'tile_paths': []  # No tile paths for pre-computed features
        }


class TileDataset(Dataset):
    """Dataset for slide-level classification using pre-loaded tiles."""
    
    def __init__(self, tile_paths: List[str], labels: List[int], slide_ids: List[str], num_tiles: int = 25):
        self.slide_ids = slide_ids
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.num_tiles = num_tiles
        self.tile_paths = tile_paths
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        logger.info(f"Pre-loading tiles for {len(tile_paths)} slides...")
        self.slide_features = []
        for slide_id, slide_tile_paths in tqdm(zip(slide_ids, tile_paths), total=len(tile_paths), desc="Pre-loading slides"):
            slide_tensors = self._preload_slide_tiles(slide_tile_paths, slide_id)
            self.slide_features.append(slide_tensors)

        memory_gb = len(self.slide_features) * num_tiles * 3 * 224 * 224 * 4 / (1024**3)
        logger.info(f"Pre-loaded {len(self.slide_features)} slides ({memory_gb:.2f} GB)")
    
    def _preload_slide_tiles(self, tile_paths: List[str], slide_id: str) -> torch.Tensor:
        try:
            tile_images = []
            available_tiles = min(len(tile_paths), self.num_tiles)
            
            for i in range(available_tiles):
                try:
                    with Image.open(tile_paths[i]) as tile:
                        tile = tile.convert('RGB')
                        tile = self.transform(tile)
                        tile_images.append(tile)
                except Exception as e:
                    logger.warning(f"Failed to load tile {tile_paths[i]}: {e}")
            
            while len(tile_images) < self.num_tiles:
                tile_images.append(torch.ones(3, 224, 224)) # white padding
            
            return torch.stack(tile_images[:self.num_tiles], dim=0)
        except Exception as e:
            logger.error(f"Error processing slide {slide_id}: {e}")
    
    def __len__(self):
        return len(self.slide_features)
    
    def __getitem__(self, idx):
        return {
            'features': self.slide_features[idx],  # Already pre-loaded and transformed
            'labels': self.labels[idx],
            'slide_id': self.slide_ids[idx],
            'tile_paths': self.tile_paths[idx] if idx < len(self.tile_paths) else []
        }
        

def load_eagle_features(features_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load EAGLE features from HDF5 files."""
    features = []
    slide_ids = []
    
    features_path = Path(features_path)
    
    if features_path.is_file():
        with h5py.File(features_path, 'r') as f:
            for slide_id in f.keys():
                slide_group = f[slide_id]
                if 'features' in slide_group:
                    slide_features = slide_group['features'][:]
                else:
                    slide_features = slide_group[:]
                features.append(slide_features)
                slide_ids.append(slide_id)
                
    elif features_path.is_dir():
        h5_files = [f for f in features_path.glob("*_eagle_features.h5")]
        logger.info(f"Found {len(h5_files)} H5 files in {features_path.name} (excluding summary tables)")
        
        for h5_file in h5_files:
            slide_id = h5_file.stem.replace("_eagle_features", "")
            try:
                with h5py.File(h5_file, 'r') as f:
                    features.append(f['features'][:])
                    slide_ids.append(slide_id)
                    
            except Exception as e:
                logger.warning(f"Failed to load {h5_file}: {e}")
                continue
    
    if not features:
        raise ValueError(f"No features loaded from {features_path}")
    
    features = np.array(features)
    logger.info(f"Loaded {len(features)} EAGLE features from {features_path.name}")
    logger.info(f"Feature shape: {features.shape}")
    
    return features, slide_ids


def prepare_data_loaders(
    features_path: str,
    slide_table_path: str,
    batch_size: int = 32,
    label_column: str = 'CATEGORY',
    patient_column: str = 'PATIENT',
    test_size: float = 0.2,
    val_size: float = 0.2,
    use_class_weights: bool = True,
    min_slides: int = 5,
    num_workers: int = 4,
    random_state: int = 42,
    use_distributed: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder, Dict]:
    """Prepare data loaders for EAGLE features."""
    
    features, slide_ids = load_eagle_features(features_path)
    slide_table = pd.read_csv(slide_table_path)
    logger.info(f"Loaded {len(slide_table)} samples")
    
    label_dist = slide_table[label_column].value_counts()
    valid_classes = label_dist[label_dist >= min_slides].index.tolist()
    excluded_classes = label_dist[label_dist < min_slides].index.tolist()
    
    if excluded_classes:
        logger.info(f"Excluded {len(excluded_classes)} classes with < {min_slides} slides")
    
    slide_table = slide_table[slide_table[label_column].isin(valid_classes)].copy()
    
    label_encoder = LabelEncoder()
    slide_table['encoded_labels'] = label_encoder.fit_transform(slide_table[label_column])
    logger.info(f"Using {len(label_encoder.classes_)} classes with {len(slide_table)} samples")
    
    patient_labels = slide_table.groupby(patient_column)['encoded_labels'].first()
    train_patients, test_patients = train_test_split(
        patient_labels.index, test_size=test_size, random_state=random_state,
        stratify=patient_labels.values
    )
    
    adjusted_val_size = val_size / (1 - test_size)
    
    train_patient_labels = patient_labels[train_patients]
    train_patients, val_patients = train_test_split(
        train_patients, test_size=adjusted_val_size, random_state=random_state,
        stratify=train_patient_labels.values
    )
    
    train_df = slide_table[slide_table[patient_column].isin(train_patients)]
    val_df = slide_table[slide_table[patient_column].isin(val_patients)]
    test_df = slide_table[slide_table[patient_column].isin(test_patients)]
    
    # Log patient-based splits
    logger.info("Patient-based splits with stratification:")
    logger.info(f"  Train: {len(train_df)} slides from {len(train_patients)} patients")
    logger.info(f"  Val: {len(val_df)} slides from {len(val_patients)} patients")
    logger.info(f"  Test: {len(test_df)} slides from {len(test_patients)} patients")
    
    # Create datasets
    def create_dataset_from_df(df):
        slide_to_feature = {sid: i for i, sid in enumerate(slide_ids)}
        dataset_features, dataset_labels, dataset_slide_ids = [], [], []
        
        for _, row in df.iterrows():
            slide_id = os.path.splitext(row['FILENAME'])[0]
            if slide_id in slide_to_feature:
                dataset_features.append(features[slide_to_feature[slide_id]])
                dataset_labels.append(row['encoded_labels'])
                dataset_slide_ids.append(slide_id)
        
        return EAGLEFeaturesDataset(
            features=np.array(dataset_features),
            labels=np.array(dataset_labels),
            slide_ids=dataset_slide_ids
        )
    
    train_dataset = create_dataset_from_df(train_df)
    val_dataset = create_dataset_from_df(val_df)
    test_dataset = create_dataset_from_df(test_df)
    
    train_sampler, val_sampler, test_sampler = None, None, None
    class_weights = None
    
    if use_class_weights and len(train_dataset) > 0:
        train_labels = train_dataset.labels.numpy()
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        sample_weights = class_weights[train_labels]
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False
    else:
        shuffle_train = True
    
    shuffle_val = shuffle_test = False
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=shuffle_train and train_sampler is None, num_workers=num_workers,
        pin_memory=True, collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    info_dict = {
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'feature_dim': features.shape[1],
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'class_weights': class_weights.tolist() if use_class_weights else None
    }
    
    return train_loader, val_loader, test_loader, label_encoder, info_dict

def prepare_slide_data_loaders(
    slide_table_path: str,
    batch_size: int = 8,
    label_column: str = 'CATEGORY',
    patient_column: str = 'PATIENT',
    test_size: float = 0.2,
    val_size: float = 0.2,
    use_class_weights: bool = True,
    min_slides: int = 5,
    num_workers: int = 4,
    num_tiles: int = 25,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder, Dict]:
    """Prepare data loaders for slide-level classification with Virchow2 LORA."""
    
    slide_table = pd.read_csv(slide_table_path)
    
    required_cols = [patient_column, 'FILENAME', label_column, 'TILE_PATH']
    missing_cols = [col for col in required_cols if col not in slide_table.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    slide_level_df = slide_table.groupby('SLIDE_ID').first().reset_index()
    logger.info(f"Loaded {len(slide_table)} tiles from {len(slide_level_df)} slides")
    
    label_dist = slide_level_df[label_column].value_counts()
    valid_classes = label_dist[label_dist >= min_slides].index.tolist()
    excluded_classes = label_dist[label_dist < min_slides].index.tolist()
    
    if excluded_classes:
        logger.info(f"Excluded {len(excluded_classes)} classes with < {min_slides} slides")
    
    slide_level_df = slide_level_df[slide_level_df[label_column].isin(valid_classes)].copy()
    slide_table = slide_table[slide_table[label_column].isin(valid_classes)].copy()
    
    label_encoder = LabelEncoder()
    slide_level_df['encoded_labels'] = label_encoder.fit_transform(slide_level_df[label_column])
    slide_table['encoded_labels'] = slide_table[label_column].map(dict(zip(slide_level_df[label_column], slide_level_df['encoded_labels'])))
    logger.info(f"Using {len(label_encoder.classes_)} classes with {len(slide_level_df)} slides")
    
    patient_labels = slide_level_df.groupby(patient_column)['encoded_labels'].first()
    train_patients, test_patients = train_test_split(
        patient_labels.index, test_size=test_size, random_state=random_state,
        stratify=patient_labels.values
    )
    
    train_patient_labels = patient_labels[train_patients]
    train_patients, val_patients = train_test_split(
        train_patients, test_size=val_size / (1 - test_size), random_state=random_state,
        stratify=train_patient_labels.values
    )
    
    train_slides_df = slide_level_df[slide_level_df[patient_column].isin(train_patients)]
    val_slides_df = slide_level_df[slide_level_df[patient_column].isin(val_patients)]
    test_slides_df = slide_level_df[slide_level_df[patient_column].isin(test_patients)]
    
    train_df = slide_table[slide_table[patient_column].isin(train_patients)]
    val_df = slide_table[slide_table[patient_column].isin(val_patients)]
    test_df = slide_table[slide_table[patient_column].isin(test_patients)]
    
    logger.info(f"Split: Train={len(train_slides_df)}, Val={len(val_slides_df)}, Test={len(test_slides_df)}")
    
    # Create datasets
    def create_slide_dataset_from_df(df):
        dataset_slide_ids = df['SLIDE_ID'].unique().tolist()
        dataset_labels = [df[df['SLIDE_ID'] == sid]['encoded_labels'].iloc[0] for sid in dataset_slide_ids]
        tile_paths = [df[df['SLIDE_ID'] == sid]['TILE_PATH'].tolist() for sid in dataset_slide_ids]
        
        return TileDataset(
            tile_paths=tile_paths,
            labels=dataset_labels,
            slide_ids=dataset_slide_ids,
            num_tiles=num_tiles
        )
    
    train_dataset = create_slide_dataset_from_df(train_df)
    val_dataset = create_slide_dataset_from_df(val_df)
    test_dataset = create_slide_dataset_from_df(test_df)
    
    train_sampler, val_sampler, test_sampler = None, None, None
    class_weights = None
    
    if use_class_weights and len(train_dataset) > 0:
        train_labels = train_dataset.labels.numpy()
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        sample_weights = class_weights[train_labels]
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False
    else:
        shuffle_train = True
    
    shuffle_val = shuffle_test = False
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=shuffle_train and train_sampler is None, num_workers=num_workers,
        pin_memory=True, collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    info_dict = {
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'feature_dim': 2048,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'class_weights': class_weights.tolist() if use_class_weights else None
    }
    
    return train_loader, val_loader, test_loader, label_encoder, info_dict


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int, 
                   class_names: Optional[List[str]] = None, probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {}

    if probs is not None:
        try:
            weighted_auc = roc_auc_score(targets, probs, multi_class="ovr", average="weighted")
            macro_auc = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
            metrics["weighted_auc"] = weighted_auc
            metrics["macro_auc"] = macro_auc
        except Exception as e:
            logger.warning(f"Failed to compute AUC: {e}")

    accuracy = accuracy_score(targets, predictions)
    balanced_acc = balanced_accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted', zero_division=0)
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(targets, predictions, average=None, zero_division=0)
    
    metrics.update({
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    })
    
    # Add per-class metrics
    if class_names:
        for i, class_name in enumerate(class_names):
            if i < len(per_class_precision):
                metrics[f'{class_name}_precision'] = per_class_precision[i]
                metrics[f'{class_name}_recall'] = per_class_recall[i]
                metrics[f'{class_name}_f1'] = per_class_f1[i]
                metrics[f'{class_name}_support'] = per_class_support[i]
    
    return metrics


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False,
    save_lora_only: bool = True
):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'lora_only': save_lora_only and hasattr(model, 'get_lora_parameters'),
        'model_state_dict': model.get_lora_parameters() if (save_lora_only and hasattr(model, 'get_lora_parameters')) else model.state_dict()
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint: {best_path}")
    
    if wandb.run is not None:
        wandb.save(str(checkpoint_path))
        if is_best:
            wandb.save(str(best_path))


def load_model_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Load model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        try:
            import numpy as np
            safe_globals = [np.core.multiarray.scalar, np.ndarray, np.dtype, np.core.multiarray._reconstruct]
            try:
                safe_globals.extend([np.core.numeric.normalize_axis_index, np._NoValue])
            except AttributeError:
                pass
            with torch.serialization.safe_globals(safe_globals):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            logger.warning("Loading checkpoint with weights_only=False for compatibility")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if checkpoint.get('lora_only', False) and hasattr(model, 'load_lora_parameters'):
        model.load_lora_parameters(checkpoint['model_state_dict'])
    else:
        state_dict = checkpoint['model_state_dict']
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        overlap = len(model_keys.intersection(checkpoint_keys))
        
        if overlap == 0:
            logger.info("Detected structure mismatch, adjusting keys...")
            adjusted_state_dict = {}
            
            if any(key.startswith('module.') for key in checkpoint_keys):
                for key, value in state_dict.items():
                    adjusted_state_dict[key[len('module.'):] if key.startswith('module.') else key] = value
            elif any('base_model.model.' in key for key in checkpoint_keys):
                for key, value in state_dict.items():
                    new_key = key.replace('.base_model.model.', '.') if '.base_model.model.' in key else \
                             key[len('base_model.model.'):] if key.startswith('base_model.model.') else key
                    adjusted_state_dict[new_key] = value
            elif any('base_model.model.' in key for key in model_keys):
                for key, value in state_dict.items():
                    new_key = key.replace('backbone.', 'backbone.base_model.model.') if key.startswith('backbone.') else f'base_model.model.{key}'
                    adjusted_state_dict[new_key] = value
            elif any(key.startswith('module.') for key in model_keys):
                adjusted_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
            state_dict = adjusted_state_dict if adjusted_state_dict else state_dict
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed, using non-strict mode")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing {len(missing_keys)} keys")
            if unexpected_keys:
                logger.warning(f"Unexpected {len(unexpected_keys)} keys")
            
            critical_missing = [k for k in missing_keys if 'lora_' not in k and 'classifier' not in k]
            if critical_missing:
                raise RuntimeError(f"Failed to load checkpoint - {len(critical_missing)} critical keys missing")
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint

def setup_wandb(
    project_name: str,
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None
):
    """Initialize Weights & Biases logging."""
    wandb.init(project=project_name, name=run_name, config=config, tags=tags, reinit=True)
    logger.info(f"Initialized W&B logging for {project_name}")