"""Utility functions for feature extraction pipeline."""

import os
import itertools
import numpy as np
import h5py
import torch
from PIL import Image
import timm
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import logging
from typing import List, Tuple, Dict
import openslide
from openslide import OpenSlide

from models.ctranspath import CTransPath
from models.CHIEF import CHIEF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batched(iterable, batch_size):
    """Batch an iterable into lists of length batch_size."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch

def load_chief_model(model_weights_path: str, device: torch.device):
    model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"CHIEF weights not found: {model_weights_path}")
    
    td = torch.load(model_weights_path, map_location=device)
    if "organ_embedding" in td:
        del td["organ_embedding"]
    model.load_state_dict(td, strict=True)
    model.eval().to(device)
    
    logger.info(f"Loaded CHIEF model from: {model_weights_path}")
    return model

def load_virchow2_model(ckpt_dir: str, device: torch.device):
    checkpoint = "Virchow2_pretraining.pth"
    ckpt_path = os.path.join(ckpt_dir, checkpoint)

    if not os.path.exists(ckpt_path):
        logger.info("Downloading Virchow2 weights...")
        login()
        os.makedirs(ckpt_dir, exist_ok=True)
        temp_model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        torch.save(temp_model.state_dict(), ckpt_path)
    
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=False,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    config = resolve_data_config(
        model.pretrained_cfg, model=model
    )
    transform = create_transform(**config)
    logger.info(f"Loaded Virchow2 model from: {ckpt_path}")
    return model, transform

def load_ctranspath_model(model_weights_path: str, device: torch.device):
    model = CTransPath(
        weights_path=model_weights_path if model_weights_path and os.path.exists(model_weights_path) else None,
        device=device
    )
    logger.info("Loaded CTransPath model")
    return model

def extract_tiles_from_slide(
    slide_path: str, 
    coords: np.ndarray, 
    tile_size: int = 224,
    level: int = 0
) -> List[Image.Image]:
    tiles = []
    slide = OpenSlide(slide_path)
    for coord in coords:
        x, y = int(coord[0]), int(coord[1])
        tile = slide.read_region(
            location=(x, y),
            level=level,
            size=(tile_size, tile_size)
        ).convert('RGB')
        tiles.append(tile)
    slide.close()
    return tiles

def load_ctranspath_features(h5_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(h5_path, 'r') as f:
        feats = torch.tensor(f['feats'][:], dtype=torch.float32)
        coords = torch.tensor(f['coords'][:], dtype=torch.int)
    return feats, coords

def validate_slide_file(slide_path: str) -> bool:
    try:
        slide = OpenSlide(slide_path)
        slide.close()
        return True
    except Exception as e:
        logger.warning(f"Invalid slide file {slide_path}: {e}")
        return False

def get_slide_level_for_mpp(slide_path: str, target_mpp: float = 2.0) -> int:
    """Calculate the appropriate level for a target microns per pixel (mpp) resolution."""
    slide = OpenSlide(slide_path)
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp_x is None:
        print("MPP information not available in slide properties. Using level 0.")
        return 0

    base_mpp = float(mpp_x) # x for reference
    required_downsample = target_mpp / base_mpp
    return slide.get_best_level_for_downsample(required_downsample)