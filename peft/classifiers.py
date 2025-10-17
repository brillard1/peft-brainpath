"""Baseline classifiers and LoRA adapter on Virchow2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Union, List
import logging
from pathlib import Path

try:
    from timm.layers import SwiGLUPacked
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    try:
        from timm.models.layers import SwiGLUPacked
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
    except ImportError:
        SwiGLUPacked = None
        timm = None
        resolve_data_config = None
        create_transform = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinearClassifier(nn.Module):
    """Simple linear classification head for pre-computed features."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPClassifier(nn.Module):
    """Two-layer MLP classifier"""
    def __init__(self, input_dim: int, num_classes: int, mlp_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


from peft import LoraConfig as PeftLoraConfig, get_peft_model


class Virchow2LoRAClassifier(nn.Module):
    """Virchow2 backbone with PEFT LoRA"""

    def __init__(
        self,
        num_classes: int,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        model_weights_dir: str = "./model_weights",
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        if timm is None:
            raise ImportError("timm is required for Virchow2LoRAClassifier")

        device = torch.device(device) if isinstance(device, str) else device
        self.device = device
        backbone, virchow2_transform = self._load_virchow2_backbone(model_weights_dir, device)
        self.virchow2_transform = virchow2_transform

        if hasattr(backbone, "head"): # Remove existing classifier if present
            backbone.reset_classifier(0)

        self.backbone = self._apply_peft_lora(backbone, lora_r, lora_alpha, lora_dropout, target_modules)
        self.feature_dim = getattr(self.backbone, "num_features", None) # 1280

        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _load_virchow2_backbone(self, weights_dir: str, device: torch.device):
        ckpt_name = "Virchow2_pretraining.pth"
        ckpt_path = Path(weights_dir) / ckpt_name

        if not ckpt_path.exists():
            logger.info("Downloading Virchow2 weights...")
            backbone_tmp = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(backbone_tmp.state_dict(), ckpt_path)

        backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=False,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone.load_state_dict(state_dict)
        backbone.eval().to(device)
        config = resolve_data_config(
            backbone.pretrained_cfg, model=backbone
        )
        transform = create_transform(**config)
        return backbone, transform

    def _apply_peft_lora(self, backbone, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: List[str] = None):
        try:
            # Add config for PEFT compatibility
            if not hasattr(backbone, 'config'):
                class ModelConfig:
                    def __init__(self):
                        self.use_return_dict = False
                        self.tie_word_embeddings = False
                        
                    def get(self, key, default=None):
                        return getattr(self, key, default)

                backbone.config = ModelConfig()

            assert target_modules is not None, "target_modules must be specified"

            lora_config = PeftLoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )

            backbone = get_peft_model(backbone, lora_config)
            
            return backbone
            
        except Exception as e:
            logger.warning(f"PEFT LoRA failed: {e}")
            raise e 

    def _extract_features(self, x: torch.Tensor):
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        assert len(x.shape) == 5, "Input should be 5 dimensional"
        batch_size, num_tiles, channels, height, width = x.shape

        x = x.view(batch_size * num_tiles, channels, height, width)

        processed_tiles = [self.virchow2_transform(tile) for tile in x]
        x = torch.stack(processed_tiles).to(self.device)
        features = self.backbone(x)
        features = features[:, 0]  # [batch_size, 1280] (class token) -> class token + 4 register tokens + 256 patch tokens
        feature_dim = features.shape[-1]
        features = features.view(batch_size, num_tiles, feature_dim)
        
        # Mean pooling to get slide-level features
        features = features.mean(dim=1)
        output = self.classifier(features)
        return output

def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    model_type = config.get('model_type', 'linear')
    
    if model_type == 'linear':
        return LinearClassifier(
            input_dim=config['input_dim'],
            num_classes=config['num_classes']
        )
    elif model_type == 'mlp':
        return MLPClassifier(
            input_dim=config['input_dim'],
            num_classes=config['num_classes'],
            mlp_dim=config.get('hidden_dims', [256])[0]
        )
    elif model_type == 'lora':
        return Virchow2LoRAClassifier(
            num_classes=config['num_classes'],
            lora_r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.1),
            target_modules=config.get('target_modules', None),
            model_weights_dir=config.get('model_weights_dir', "./model_weights"),
            device=config.get('device', "cuda")
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")