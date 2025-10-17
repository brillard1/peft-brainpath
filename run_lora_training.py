"""Training script for LoRA models."""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import load_config, save_config
from peft.lora_trainer import LoRATrainer
from peft.classifiers import LinearClassifier, MLPClassifier, Virchow2LoRAClassifier
from peft.utils import prepare_data_loaders, prepare_slide_data_loaders
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: dict) -> nn.Module:
    """Create model from config."""
    model_config = config['model']
    model_type = model_config['model_type'].lower()
    
    if model_type == "linear":
        return LinearClassifier(
            input_dim=model_config['input_dim'],
            num_classes=model_config['num_classes']
        )
    elif model_type == "mlp":
        return MLPClassifier(
            input_dim=model_config['input_dim'],
            num_classes=model_config['num_classes'],
            mlp_dim=model_config['hidden_dims'][0],
            dropout=model_config['dropout_rate']
        )
    elif model_type == "lora":
        model_weights_dir = config.get('model_weights_dir', './model_weights')
        return Virchow2LoRAClassifier(
            num_classes=model_config['num_classes'],
            lora_r=model_config['lora_r'],
            lora_alpha=model_config['lora_alpha'],
            lora_dropout=model_config['lora_dropout'],
            target_modules=model_config['target_modules'],
            model_weights_dir=model_weights_dir,
            device=config['device']
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer from config."""
    training_config = config['training']
    optimizer_type = training_config['optimizer'].lower()
    lr = training_config['learning_rate']
    weight_decay = training_config['weight_decay']
    
    if optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """Create learning rate scheduler from config."""
    training_config = config['training']
    scheduler_type = training_config['scheduler'].lower()
    
    if scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',
            patience=training_config['scheduler_patience'],
            factor=training_config['scheduler_factor']
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['num_epochs'],
            eta_min=training_config['learning_rate'] * 0.01
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config['scheduler_patience'],
            gamma=training_config['scheduler_factor']
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def main(args):
    """Main training function."""
    config = load_config(args.config)
    
    set_random_seed(config['random_seed'])
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_config(config, output_dir / "config.yaml")
    
    logger.info(f"Starting training experiment: {config['experiment_name']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {config['device']}")
    
    data_config = config['data']
    if config['model']['model_type'].lower() == "lora":
        logger.info("Loading tile-level data for LoRA training...")
        train_loader, val_loader, test_loader, label_encoder, data_info = prepare_slide_data_loaders(
            slide_table_path=data_config['tile_table_path'],
            batch_size=data_config['batch_size'],
            label_column=data_config['label_column'],
            patient_column=data_config['patient_column'],
            test_size=data_config['test_size'],
            val_size=data_config['val_size'],
            use_class_weights=data_config['use_class_weights'],
            min_slides=data_config['min_slides'],
            num_workers=data_config['num_workers'],
            random_state=config['random_seed']
        )
    else:
        logger.info("Loading pre-computed EAGLE features...")
        train_loader, val_loader, test_loader, label_encoder, data_info = prepare_data_loaders(
            features_path=data_config['features_path'],
            slide_table_path=data_config['slide_table_path'],
            batch_size=data_config['batch_size'],
            label_column=data_config['label_column'],
            patient_column=data_config['patient_column'],
            test_size=data_config['test_size'],
            val_size=data_config['val_size'],
            use_class_weights=data_config['use_class_weights'],
            min_slides=data_config['min_slides'],
            num_workers=data_config['num_workers'],
            random_state=config['random_seed']
        )
    
    config['model']['num_classes'] = data_info['num_classes']
    if 'class_names' not in config or config['class_names'] is None:
        config['class_names'] = data_info['class_names']
    
    logger.info(f"Data info: {data_info}")
    
    logger.info("Creating model...")
    model = create_model(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    if data_config['use_class_weights'] and data_info.get('class_weights'):
        class_weights = torch.tensor(data_info['class_weights'], dtype=torch.float32).to(config['device'])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    wandb_config = config.get('wandb', {})
    trainer = LoRATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device'],
        output_dir=str(output_dir),
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        class_names=config['class_names'],
        wandb_project=wandb_config.get('project'),
        wandb_run_name=wandb_config.get('run_name'),
        wandb_tags=[],
        best_metric=config['training']['best_metric'],
        use_multi_gpu=config.get('use_multi_gpu', False),
        mixed_precision=config.get('mixed_precision', True),
        fp16=config.get('fp16', True),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4)
    )
    
    logger.info("Starting training...")
    results = trainer.train(num_epochs=config['training']['num_epochs'])
    
    logger.info("Training completed!")
    logger.info(f"Best validation {config['training']['best_metric']}: {results['best_val_metric']:.4f}")
    
    if results.get('test_metrics'):
        logger.info("Final test results:")
        for metric, value in results['test_metrics'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
    
    if wandb.run is not None:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA models")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)

