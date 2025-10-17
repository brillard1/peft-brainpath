"""Extract CTransPath features from slides at 2.0 MPP (EAGLE recommendation).
Requires timm==0.5.4 for compatibility with CTransPath weights.
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import argparse
import openslide
from openslide import OpenSlide
from torchvision import transforms

# Import compatibility layer for timm versions
try:
    import timm
    from timm.models.layers.helpers import to_2tuple
    TIMM_VERSION = "0.5.4"  # Expected version for CTransPath
except ImportError:
    from timm.layers.helpers import to_2tuple
    TIMM_VERSION = "newer"

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_extraction.utils import get_slide_level_for_mpp
from models.ctranspath import CTransPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CTransPathFeatureExtractor:
    """CTransPath feature extractor with multi-GPU support."""
    
    def __init__(self, weights_path: str = None, device: str = "cuda", use_multi_gpu: bool = True, gpu_ids: List[int] = None):
        self.use_multi_gpu = use_multi_gpu and torch.cuda.is_available()
        self.gpu_ids = gpu_ids
        self.num_gpus = 0
        
        if torch.cuda.is_available():
            if self.gpu_ids is None:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            
            self.num_gpus = len(self.gpu_ids)
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            logger.info(f"Using GPUs: {self.gpu_ids}")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")
        
        self.model = None
        self.weights_path = weights_path
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    def load_model(self):
        logger.info("Loading CTransPath model...")
        try:
            self.model = CTransPath()
            self.model.head = nn.Identity()  # Remove classification head
            self.model.to(self.device)
            
            if self.weights_path and os.path.exists(self.weights_path):
                logger.info(f"Loading weights from: {self.weights_path}")
                
                # Load checkpoint
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                
                # Load with strict=False to handle any key mismatches
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                logger.info(f"Weight loading summary:")
                logger.info(f"  - Missing keys: {len(missing_keys)}")
                logger.info(f"  - Unexpected keys: {len(unexpected_keys)}")
                
                if len(missing_keys) > 0:
                    logger.warning(f"Missing keys (first 5): {missing_keys[:5]}")
                if len(unexpected_keys) > 0:
                    logger.warning(f"Unexpected keys (first 5): {unexpected_keys[:5]}")
                    
            else:
                logger.warning("No weights provided or weights file not found. Using random initialization.")
            
            # Enable multi-GPU if available and requested
            if self.use_multi_gpu and self.num_gpus > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            
            self.model.eval()
            logger.info("CTransPath model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CTransPath model: {e}")
            raise
    
    def get_optimal_batch_size(self, requested_batch_size: int) -> int:
        """Calculate optimal batch size for multi-GPU setup."""
        if self.use_multi_gpu and self.num_gpus > 1:
            # Ensure batch size is divisible by number of GPUs for even distribution
            if requested_batch_size % self.num_gpus != 0:
                optimal_batch_size = ((requested_batch_size // self.num_gpus) + 1) * self.num_gpus
                logger.info(f"Adjusting batch size from {requested_batch_size} to {optimal_batch_size} "
                           f"for even distribution across {self.num_gpus} GPUs")
                return optimal_batch_size
        
        return requested_batch_size
    
    def extract_patches_from_slide(
        self,
        slide_path: str,
        patch_size: int = 224,
        target_mpp: float = 2.0,
        overlap: float = 0.0,
        tissue_threshold: float = 0.8,
        max_patches: int = 100000,
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """Extract patches from slide at target MPP."""
        patches = []
        coordinates = []
        
        try:
            slide = OpenSlide(slide_path)
            
            # Calculate the appropriate level for target MPP
            level = get_slide_level_for_mpp(slide_path, target_mpp)
            level_dims = slide.level_dimensions[level]
            width, height = level_dims
            downsample_factor = slide.level_downsamples[level]
            
            step_size = int(patch_size * (1 - overlap))
            
            # Total possible patches in the slide
            y_positions = list(range(0, height - patch_size + 1, step_size))
            x_positions = list(range(0, width - patch_size + 1, step_size))
            total_possible_patches = len(y_positions) * len(x_positions)
            
            logger.info(f"Slide dimensions at level {level}: {width}x{height}, downsample: {downsample_factor:.2f}, Total possible patches: {total_possible_patches}, Max limit: {max_patches}")
            
            # Create coordinate pairs for all possible patches
            # Convert level coordinates to level 0 coordinates for read_region
            all_coordinates = []
            for y in y_positions:
                for x in x_positions:
                    # Convert to level 0 coordinates
                    level0_x = int(x * downsample_factor)
                    level0_y = int(y * downsample_factor)
                    all_coordinates.append((level0_x, level0_y, x, y))  # Store both for reference

            # Limit coordinates if max_patches is specified
            if max_patches and len(all_coordinates) > max_patches:
                all_coordinates = all_coordinates[:max_patches]
            
            # Extract patches with progress bar
            for idx, (level0_x, level0_y, level_x, level_y) in enumerate(tqdm(all_coordinates, desc=f"Extracting patches from {os.path.basename(slide_path)}")):
                # Extract patch using level 0 coordinates
                patch = slide.read_region(
                    location=(level0_x, level0_y),
                    level=level,
                    size=(patch_size, patch_size)
                ).convert('RGB')
                
                patches.append(patch)
                coordinates.append([level0_x, level0_y])  # Store level coordinates for reference

            
            slide.close()
            
            # Log extraction summary
            if max_patches and len(patches) >= max_patches:
                logger.info(f"Extracted {len(patches)} patches at {target_mpp}mpp (level {level}) from {slide_path} (reached max limit of {max_patches} out of {total_possible_patches} total possible)")
            else:
                logger.info(f"Extracted {len(patches)} patches at {target_mpp}mpp (level {level}) from {slide_path} (out of {total_possible_patches} total possible)")
            
        except Exception as e:
            logger.error(f"Error extracting patches from {slide_path}: {e}")
            return [], np.array([])
        
        return patches, np.array(coordinates)
    
    def extract_features_from_slide(
        self,
        slide_path: str,
        output_path: str,
        patch_size: int = 224,
        batch_size: int = 32,
        target_mpp: float = 2.0,
        max_patches: int = 100000,
    ) -> bool:
        """Extract CTransPath features from a single slide"""
        
        if self.model is None:
            self.load_model()
        
        # Optimize batch size for multi-GPU
        optimal_batch_size = self.get_optimal_batch_size(batch_size)
        if optimal_batch_size != batch_size:
            logger.info(f"Using optimal batch size: {optimal_batch_size} (requested: {batch_size})")
        
        try:
            # Extract patches
            patches, coordinates = self.extract_patches_from_slide(
                slide_path=slide_path,
                patch_size=patch_size,
                target_mpp=target_mpp,
                max_patches=max_patches,
            )
            
            if len(patches) == 0:
                logger.warning(f"No patches extracted from {slide_path}")
                return False
            
            # Transform patches
            logger.info(f"Transforming {len(patches)} patches...")
            patch_tensors = [self.transform(patch) for patch in patches]
            
            # Extract features in batches
            all_features = []
            num_batches = (len(patch_tensors) + optimal_batch_size - 1) // optimal_batch_size
            
            logger.info(f"Processing {len(patches)} patches in {num_batches} batches (batch_size={optimal_batch_size}) "
                       f"using {self.num_gpus} GPU(s)")
            
            with torch.no_grad():
                for batch_idx, batch_start in enumerate(range(0, len(patch_tensors), optimal_batch_size)):
                    batch_end = min(batch_start + optimal_batch_size, len(patch_tensors))
                    batch_patches = patch_tensors[batch_start:batch_end]
                    
                    batch_tensor = torch.stack(batch_patches).to(self.device)
                    batch_features = self.model(batch_tensor)
                    all_features.append(batch_features.cpu())

            features = torch.cat(all_features, dim=0)
            coordinates_tensor = torch.tensor(coordinates, dtype=torch.long)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('feats', data=features.numpy())
                f.create_dataset('coords', data=coordinates_tensor.numpy())
                
                f.attrs['slide_path'] = slide_path
                f.attrs['patch_size'] = patch_size
                f.attrs['target_mpp'] = target_mpp
                f.attrs['num_patches'] = len(features)
                f.attrs['feature_dim'] = features.shape[-1]
            
            logger.info(f"Saved {features.shape[0]} features to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing slide {slide_path}: {e}")
            return False
    
    def process_slide_table(
        self,
        slide_table_path: str,
        output_dir: str,
        batch_size: int = 32,
        target_mpp: float = 2.0,
        max_patches: int = 100000,
    ):
        """Process all slides in a slide table"""
        
        # Load slide table
        slide_table = pd.read_csv(slide_table_path)
        logger.info(f"Processing {len(slide_table)} slides from {slide_table_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model once
        if self.model is None:
            self.load_model()
        
        # Process each slide
        success_count = 0
        
        for idx, row in slide_table.iterrows():
            slide_id = os.path.splitext(row['FILENAME'])[0]
            slide_path = row.get('SLIDE_PATH', row['FILENAME'])
            
            logger.info(f"Processing slide {idx + 1}/{len(slide_table)}: {slide_id}")
            
            output_path = output_dir / f"{slide_id}_ctranspath_features.h5"
            
            if output_path.exists():
                logger.info(f"Skipping {slide_id} - features already exist")
                success_count += 1
                continue
            
            success = self.extract_features_from_slide(
                slide_path=slide_path,
                output_path=str(output_path),
                batch_size=batch_size,
                target_mpp=target_mpp,
                max_patches=max_patches,
            )
            
            if success:
                success_count += 1
            else:
                logger.warning(f"Failed to process {slide_id}")
        
        logger.info(f"Successfully processed {success_count}/{len(slide_table)} slides")
        logger.info(f"Features saved to: {output_dir}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="CTransPath Feature Extraction")
    parser.add_argument("--slide_table", required=True, help="Path to slide table CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save CTransPath features")
    parser.add_argument("--weights_path", default="./model_weights/ctranspath.pth", help="Path to CTransPath weights file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing (will be auto-adjusted for multi-GPU)")
    parser.add_argument("--target_mpp", type=float, default=2.0, help="Target microns per pixel (default: 2.0 for EAGLE)")
    parser.add_argument("--max_patches", type=int, default=100000, help="Maximum patches per slide")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). Uses all available GPUs if not specified.")
    
    args = parser.parse_args()
    
    # Parse GPU configuration
    gpu_ids = None
    
    if args.gpu_ids:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            logger.info(f"Using specified GPU IDs: {gpu_ids}")
        except ValueError:
            logger.error(f"Invalid GPU IDs format: {args.gpu_ids}. Use comma-separated integers (e.g., '0,1,2')")
            return
    
    # Initialize extractor
    extractor = CTransPathFeatureExtractor(
        weights_path=args.weights_path,
        device=args.device,
        use_multi_gpu=True,
        gpu_ids=gpu_ids
    )
    
    # Process slide table
    logger.info(f"Starting CTransPath feature extraction...")
    logger.info(f"Slide table: {args.slide_table}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Target MPP: {args.target_mpp}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Multi-GPU enabled: {extractor.use_multi_gpu}")
    if extractor.use_multi_gpu and extractor.num_gpus > 1:
        logger.info(f"Using {extractor.num_gpus} GPUs: {extractor.gpu_ids}")
        optimal_batch = extractor.get_optimal_batch_size(args.batch_size)
        if optimal_batch != args.batch_size:
            logger.info(f"Effective batch size will be: {optimal_batch}")
    
    extractor.process_slide_table(
        slide_table_path=args.slide_table,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        target_mpp=args.target_mpp,
        max_patches=args.max_patches,
    )
    
    logger.info("CTransPath feature extraction completed!")


if __name__ == "__main__":
    main() 