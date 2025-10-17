"""Evaluation script for trained LoRA models."""

import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import logging
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import load_config
from peft.lora_trainer import LoRATrainer
from peft.classifiers import LinearClassifier, MLPClassifier, Virchow2LoRAClassifier
from peft.utils import prepare_data_loaders, prepare_slide_data_loaders, load_model_checkpoint
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best checkpoint in the directory."""
    checkpoint_files = list(checkpoint_dir.glob("best_epoch_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")
    
    if len(checkpoint_files) > 1:
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    return checkpoint_files[-1]


def main(args):
    """Main evaluation function."""
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint_path)
    
    if checkpoint_path.is_dir():
        checkpoint_path = find_best_checkpoint(checkpoint_path)
        logger.info(f"Auto-selected checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    eval_output_dir = output_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = Path(args.config_path) if args.config_path else output_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from: {config_path}")
    config = load_config(str(config_path))
    
    logger.info(f"Starting model evaluation")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Evaluation output: {eval_output_dir}")
    
    try:
        with open(output_dir / "label_encoder.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"Loaded label encoder with {len(label_encoder.classes_)} classes")
    except FileNotFoundError:
        logger.warning("Label encoder not found, will be recreated from data")
        label_encoder = None
    
    logger.info("Preparing data loaders...")
    data_config = config['data']
    
    if config['model']['model_type'].lower() == "lora":
        train_loader, val_loader, test_loader, data_label_encoder, data_info = prepare_slide_data_loaders(
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
        train_loader, val_loader, test_loader, data_label_encoder, data_info = prepare_data_loaders(
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
    
    if label_encoder is None:
        label_encoder = data_label_encoder
        logger.info("Using label encoder from data loading")
    
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
    
    criterion = nn.CrossEntropyLoss()
    
    trainer = LoRATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=config['device'],
        output_dir=str(eval_output_dir),
        class_names=config['class_names'],
        wandb_project=None,
        use_multi_gpu=False,
        mixed_precision=config.get('mixed_precision', True)
    )
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = load_model_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=trainer.model,
        device=config['device']
    )
    
    loaded_epoch = checkpoint_data.get('epoch', 'unknown')
    loaded_metrics = checkpoint_data.get('metrics', {})
    logger.info(f"Loaded checkpoint from epoch {loaded_epoch}")
    if loaded_metrics:
        logger.info("Checkpoint metrics:")
        for metric, value in loaded_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Running test evaluation...")
    test_metrics = trainer.test_epoch()
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test set size: {len(test_loader.dataset)} samples")
    logger.info(f"Number of classes: {len(config['class_names'])}")
    logger.info("\nTest Metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    eval_results = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_epoch': loaded_epoch,
        'checkpoint_metrics': loaded_metrics,
        'test_metrics': test_metrics,
        'data_info': data_info,
        'class_names': config['class_names']
    }
    
    results_path = eval_output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to: {results_path}")
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Check {eval_output_dir} for all outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint file or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Original training output directory")
    parser.add_argument("--config_path", type=str, help="Path to config file (default: output_dir/config.yaml)")
    args = parser.parse_args()
    
    main(args)

