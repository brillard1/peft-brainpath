"""LoRA Trainer with GPU optimizations."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.amp import GradScaler
from tqdm import tqdm
import wandb
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any

from .utils import (
    compute_metrics,
    save_model_checkpoint,
    load_model_checkpoint,
    setup_wandb
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
        patience: int = 10,
        min_delta: float = 1e-4,
        log_every_n_steps: int = 10,
        class_names: Optional[List[str]] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        best_metric: str = "weighted_f1",  # Primary metric for best model selection
        # GPU Optimization parameters
        use_multi_gpu: bool = True,
        mixed_precision: bool = True,
        fp16: bool = True,
        gradient_accumulation_steps: int = 4,
        detailed_timing: bool = True,
        memory_profiling: bool = True,
        enable_tsne: bool = True
    ):
        """
        Initialize LORA trainer with DataParallel GPU optimizations.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            device: Device to use for training
            output_dir: Directory to save outputs
            patience: Early stopping patience
            min_delta: Minimum change for early stopping
            log_every_n_steps: Log metrics every N steps
            class_names: Optional class names for detailed metrics
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            wandb_tags: W&B tags
            best_metric: Metric to use for best model selection ('weighted_f1', 'weighted_auc', 'accuracy')
            use_multi_gpu: Enable multi-GPU DataParallel training
            mixed_precision: Enable mixed precision training
            fp16: Force FP16 precision
            gradient_accumulation_steps: Steps for gradient accumulation
            detailed_timing: Enable detailed timing measurements
            memory_profiling: Enable memory profiling
            enable_tsne: Enable TSNE visualization of test results
        """
        # Store optimization settings
        self.use_multi_gpu = use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.mixed_precision = mixed_precision and fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.detailed_timing = detailed_timing
        self.memory_profiling = memory_profiling
        self.best_metric = best_metric  # Store the metric for best model selection
        self.enable_tsne = enable_tsne
        
        # Determine device setup
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Setup multi-GPU DataParallel if available
        if self.use_multi_gpu:
            self.model = DataParallel(self.model)
            logger.info(f"Wrapped model with DataParallel on {self.num_gpus} GPUs")
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set up criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=1e-3, 
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Mixed precision setup
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Training configuration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_metric = float('-inf')
        self.epochs_without_improvement = 0
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
        self.class_names = class_names
        
        # Best checkpoint tracking
        self.best_checkpoint_path = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_balanced_accuracy': [],
            'val_f1': [],
            'val_auc': [],
            'memory_usage': []
        }
        
        # Initialize wandb if specified
        if wandb_project:
            config = {
                'model_type': type(self.model).__name__,
                'patience': patience,
                'min_delta': min_delta,
                'optimizer': type(self.optimizer).__name__,
                'scheduler': type(self.scheduler).__name__ if scheduler else None,
                'batch_size': train_loader.batch_size,
                'num_train_samples': len(train_loader.dataset),
                'num_val_samples': len(val_loader.dataset),
                'num_test_samples': len(test_loader.dataset) if test_loader else 0,
                'use_multi_gpu': self.use_multi_gpu,
                'num_gpus': self.num_gpus,
                'mixed_precision': self.mixed_precision,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'enable_tsne': self.enable_tsne
            }
            
            setup_wandb(
                project_name=wandb_project,
                run_name=wandb_run_name,
                config=config,
                tags=wandb_tags
            )
        
        logger.info("Initialized LORA Trainer with optimizations")
        logger.info(f"Model: {type(self.model).__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Multi-GPU Training: {self.use_multi_gpu}")
        logger.info(f"Number of GPUs: {self.num_gpus}")
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_tiles = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                outputs = self.model(batch_tiles)
                
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate loss
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                total_loss += accumulated_loss
                num_batches += 1
                self.global_step += 1
                accumulated_loss = 0.0
                
                # Update progress bar
                current_loss = total_loss / num_batches
                postfix = {'loss': f'{current_loss:.4f}'}
                
                progress_bar.set_postfix(postfix)
                
                if self.global_step % self.log_every_n_steps == 0:
                    if wandb.run is not None:
                        log_dict = {
                            'train/step_loss': accumulated_loss * self.gradient_accumulation_steps,
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                            'global_step': self.global_step
                        }

                        wandb.log(log_dict)
        
        avg_loss = total_loss / max(num_batches, 1)
        
        result = {'loss': avg_loss}
        return result
        
    
    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {self.current_epoch + 1} [Val]",
                leave=False
            )
            
            for batch in progress_bar:
                batch_tiles = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                    outputs = self.model(batch_tiles)
                    
                    loss = self.criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probs, dim=1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        avg_loss = total_loss / len(self.val_loader)
        
        logger.info(f"Computing validation metrics for epoch {self.current_epoch + 1}...")
        metrics = compute_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets),
            num_classes=len(self.class_names) if self.class_names else len(np.unique(all_targets)),
            class_names=self.class_names,
            probs=np.array(all_probs) if len(all_probs) > 0 else None
        )
        
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test_epoch(self) -> Dict[str, float]:
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_slide_ids = []
        all_probs = []
        all_features = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.test_loader,
                desc="Testing",
                leave=False
            )
            
            for batch in progress_bar:
                batch_tiles = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                slide_ids = batch['slide_id']
                
                with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                    # Extract feature embeddings before classification for TSNE
                    feature_embeddings = self._extract_feature_embeddings(batch_tiles)
                    
                    outputs = self.model(batch_tiles)
                    
                    loss = self.criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probs, dim=1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_slide_ids.extend(slide_ids)
                all_probs.extend(probs.cpu().numpy())
                all_features.extend(feature_embeddings.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        
        logger.info(f"Computing test metrics...")
        metrics = compute_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets),
            num_classes=len(self.class_names) if self.class_names else len(np.unique(all_targets)),
            class_names=self.class_names,
            probs=np.array(all_probs) if len(all_probs) > 0 else None
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _get_validation_metric(self, val_metrics: Dict[str, float]) -> float:
        return val_metrics[self.best_metric]
    
    def _extract_feature_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(model, '_extract_features'):
            assert len(x.shape) == 5, "Input should be 5 dimensional"
            batch_size, num_tiles, channels, height, width = x.shape

            x = x.view(batch_size * num_tiles, channels, height, width)
            
            processed_tiles = [model.virchow2_transform(tile) for tile in x]
            x = torch.stack(processed_tiles).to(self.device)

            features = model._extract_features(x)
            features = features[:, 0]
            feature_dim = features.shape[-1]
            features = features.view(batch_size, num_tiles, feature_dim)
            features = features.mean(dim=1)
        elif hasattr(model, 'net') and len(model.net) > 2:
            features = x
            for layer in model.net[:-1]:
                features = layer(features)
        elif hasattr(model, 'fc'):
            features = x
            
        return features
    
    def _should_stop_early(self, val_metric: float) -> bool:
        if val_metric > self.best_val_metric + self.min_delta:
            self.best_val_metric = val_metric
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience
    
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        if not is_best:
            return
        
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            self.best_checkpoint_path.unlink()
        
        filename = f"best_epoch_{self.current_epoch + 1}.pth"
        self.best_checkpoint_path = self.checkpoint_dir / filename
        
        save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch + 1,
            metrics=metrics,
            checkpoint_dir=str(self.checkpoint_dir),
            filename=filename,
            is_best=is_best
        )
    
    def train(self, num_epochs: int, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Optional checkpoint path to resume from
            
        Returns:
            Training history and final metrics
        """
        start_epoch = 0
        
        if resume_from_checkpoint:
            checkpoint = load_model_checkpoint(
                checkpoint_path=resume_from_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device
            )
            start_epoch = checkpoint.get('epoch', 0)
            self.current_epoch = start_epoch
            self.best_val_metric = checkpoint.get('metrics', {}).get(self.best_metric, float('-inf'))
            logger.info(f"Resumed training from epoch {start_epoch}")
        
        logger.info(f"Starting training for {num_epochs} epochs")
        training_start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self._get_validation_metric(val_metrics))
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_balanced_accuracy'].append(val_metrics.get('balanced_accuracy', 0))
            self.training_history['val_f1'].append(val_metrics['weighted_f1'])
            self.training_history['val_auc'].append(val_metrics.get('weighted_auc', 0))
            
            # Check if this is the best model
            current_val_metric = self._get_validation_metric(val_metrics)
            is_best = current_val_metric > self.best_val_metric
            
            # Save checkpoint
            if is_best:
                self._save_checkpoint(val_metrics, is_best=is_best)
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Balanced Acc: {val_metrics.get('balanced_accuracy', 0):.4f}, "
                f"Val F1: {val_metrics['weighted_f1']:.4f}, "
                f"Val AUC: {val_metrics.get('weighted_auc', 0):.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Log to wandb
            if wandb.run is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/balanced_accuracy': val_metrics.get('balanced_accuracy', 0),
                    'val/weighted_f1': val_metrics['weighted_f1'],
                    'val/macro_f1': val_metrics['macro_f1'],
                    'val/weighted_precision': val_metrics['weighted_precision'],
                    'val/weighted_recall': val_metrics['weighted_recall'],
                    'val/auc': val_metrics.get('weighted_auc', 0),
                    'epoch_time': epoch_time
                }
                
                # Add per-class metrics if available
                if self.class_names:
                    for class_name in self.class_names:
                        for metric_type in ['precision', 'recall', 'f1']:
                            key = f'{class_name}_{metric_type}'
                            if key in val_metrics:
                                log_dict[f'val/{key}'] = val_metrics[key]
                
                wandb.log(log_dict)
            
            # Early stopping check
            if self._should_stop_early(current_val_metric):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        training_time = time.time() - training_start_time
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Final evaluation on test set
        test_metrics = {}
        if self.test_loader:
            logger.info("Evaluating on test set...")
            test_metrics = self.test_epoch()
            
            logger.info("Test Results:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Log test metrics to wandb
            if wandb.run is not None:
                wandb_test_metrics = {f'test/{k}': v for k, v in test_metrics.items()}
                wandb.log(wandb_test_metrics)
        
        # Prepare final results
        results = {
            'training_history': self.training_history,
            'final_val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_val_metric': self.best_val_metric,
            'total_epochs': self.current_epoch + 1,
            'training_time': training_time
        }
        
        return results
    
    def evaluate(self, data_loader: DataLoader, split_name: str = "eval") -> Dict[str, float]:
        """
        Args:
            data_loader: Data loader to evaluate on
            split_name: Name of the split for logging
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                data_loader,
                desc=f"Evaluating {split_name}",
                leave=False
            )
            
            for batch in progress_bar:
                features = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                    outputs = self.model(features)
                    
                    if isinstance(outputs, dict):
                        loss = self._compute_multi_task_loss(outputs, batch)
                        logits = list(outputs.values())[0]
                        probs = torch.softmax(logits, dim=1)
                        predictions = torch.argmax(probs, dim=1)
                        all_probs.extend(probs.cpu().numpy())
                    else:
                        loss = self.criterion(outputs, labels)
                        probs = torch.softmax(outputs, dim=1)
                        predictions = torch.argmax(probs, dim=1)
                        all_probs.extend(probs.cpu().numpy())
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(data_loader)
        metrics = compute_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets),
            num_classes=len(self.class_names) if self.class_names else len(np.unique(all_targets)),
            class_names=self.class_names,
            probs=np.array(all_probs) if len(all_probs) > 0 else None
        )
        
        metrics['loss'] = avg_loss
        
        logger.info(f"{split_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Args:
            data_loader: Data loader to predict on
        Returns:
            Dictionary with predictions, targets, and slide IDs
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_slide_ids = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                data_loader,
                desc="Generating predictions",
                leave=False
            )
            
            for batch in progress_bar:
                features = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                slide_ids = batch['slide_id']
                
                with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                    outputs = self.model(features)
                    predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_slide_ids.extend(slide_ids)
        return {'predictions': np.array(all_predictions), 'targets': np.array(all_targets), 'slide_ids': all_slide_ids} 