"""EAGLE Feature Generation from Pre-extracted Tiles"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
import torch
from tqdm import tqdm
from PIL import Image
import logging
from typing import Dict, List, Optional
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.utils import (
    batched,
    load_virchow2_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EAGLEFeatureExtractor:
    """Generate EAGLE features from pre-extracted tiles using Virchow2."""
    
    def __init__(
        self,
        device: str = "cuda",
        model_weights_dir: str = None,
        batch_size: int = 32
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Set up model paths
        self.model_weights_dir = model_weights_dir
        
        # Initialize models
        self.virchow2_model = None
        self.virchow2_transform = None
        
        logger.info(f"Initialized EAGLE feature extractor")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
    
    def load_models(self):
        self.virchow2_model, self.virchow2_transform = load_virchow2_model(
            self.model_weights_dir, self.device
        )
        logger.info("Virchow2 model loaded successfully")
    
    def process_tiles_for_slide(
        self,
        slide_tiles_info: List[Dict],
        slide_id: str
    ) -> Optional[np.ndarray]:
        try:
            # Load tile images
            tile_images = []
            valid_tiles = []
            
            for tile_info in slide_tiles_info:
                tile_path = tile_info['TILE_PATH']
                
                if not os.path.exists(tile_path):
                    logger.warning(f"Tile not found: {tile_path}")
                    continue
                
                try:
                    tile_image = Image.open(tile_path).convert('RGB')
                    tile_images.append(tile_image)
                    valid_tiles.append(tile_info)
                except Exception as e:
                    logger.warning(f"Failed to load tile {tile_path}: {e}")
                    continue
            
            if not tile_images:
                logger.error(f"No valid tiles found for {slide_id}")
                return None
            
            logger.info(f"Processing {len(tile_images)} tiles for {slide_id}")
            
            # Transform tiles for Virchow2
            try:
                processed_tiles = [self.virchow2_transform(tile) for tile in tile_images]
                batch = torch.stack(processed_tiles).to(self.device)
            except Exception as e:
                logger.error(f"Error during tile transformation for {slide_id}: {e}")
                return None
            
            # Extract features with Virchow2
            tile_embeddings = []
            
            for tile_batch in batched(range(len(batch)), self.batch_size):
                batch_tiles = batch[tile_batch]
                
                if self.device.type == "cuda":
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self.virchow2_model(batch_tiles)
                else:
                    with torch.inference_mode():
                        output = self.virchow2_model(batch_tiles)
                
                # Extract class tokens (first token)
                class_tokens = output[:, 0]  # [batch_size, 1280]
                tile_embeddings.append(class_tokens.cpu())
            
            # Combine all embeddings
            all_embeddings = torch.cat(tile_embeddings, dim=0)
            
            # Average embeddings to create slide-level EAGLE feature
            eagle_embedding = torch.mean(all_embeddings, dim=0)
            
            logger.info(f"Generated EAGLE embedding for {slide_id}: shape {eagle_embedding.shape}")
            
            return eagle_embedding.numpy()
            
        except Exception as e:
            logger.error(f"Error processing tiles for slide {slide_id}: {e}")
            return None
    
    def process_all_slides(
        self,
        tiles_csv_path: str,
        output_dir: str,
        groupby_column: str = "SLIDE_ID"
    ) -> Dict[str, np.ndarray]:
        # Load tiles table
        tiles_df = pd.read_csv(tiles_csv_path)
        logger.info(f"Loaded {len(tiles_df)} tiles from {tiles_csv_path}")
        
        if self.virchow2_model is None:
            self.load_models()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Group tiles by slide
        grouped_tiles = tiles_df.groupby(groupby_column)
        logger.info(f"Processing {len(grouped_tiles)} slides")
        
        eagle_embeddings = {}
        
        # Process each slide
        for slide_id, slide_tiles_df in tqdm(grouped_tiles, desc="Processing slides"):
            # Create individual H5 file path for this slide
            slide_h5_path = os.path.join(output_dir, f"{slide_id}_eagle_features.h5")
            
            # Check if slide features already exist
            if os.path.exists(slide_h5_path):
                try:
                    with h5py.File(slide_h5_path, 'r') as h5_file:
                        if 'features' in h5_file:
                            logger.info(f"Slide {slide_id} already exists - loading existing features")
                            eagle_embeddings[slide_id] = h5_file['features'][:]
                            continue
                except Exception as e:
                    logger.info(f"Could not read existing file for {slide_id}: {e}")
            
            # Convert slide tiles to list of dictionaries
            slide_tiles_info = slide_tiles_df.to_dict('records')
            
            # Process tiles for this slide
            eagle_embedding = self.process_tiles_for_slide(
                slide_tiles_info=slide_tiles_info,
                slide_id=slide_id
            )
            
            if eagle_embedding is not None:
                with h5py.File(slide_h5_path, 'w') as h5_file:
                    h5_file.create_dataset('features', data=eagle_embedding)
                    
                    # Save metadata from first tile (since all tiles from same slide, not true for multi-scale)
                    first_tile = slide_tiles_info[0]
                    h5_file.attrs['slide_id'] = slide_id
                    h5_file.attrs['patient_id'] = first_tile.get('PATIENT', 'unknown')
                    h5_file.attrs['diagnosis'] = first_tile.get('DIAGNOSIS', 'unknown')
                    h5_file.attrs['category'] = first_tile.get('CATEGORY', 'unknown')
                    h5_file.attrs['filename'] = first_tile.get('FILENAME', 'unknown')
                    h5_file.attrs['num_tiles'] = len(slide_tiles_info)
                    h5_file.attrs['tile_size'] = first_tile.get('TILE_SIZE', 224)
                    h5_file.attrs['target_mpp'] = first_tile.get('TARGET_MPP', 2.0)
                
                eagle_embeddings[slide_id] = eagle_embedding
                logger.info(f"Saved EAGLE features for {slide_id} to: {slide_h5_path}")
            else:
                logger.warning(f"Failed to process slide: {slide_id}")
        
        logger.info(f"Successfully processed {len(eagle_embeddings)} slides")
        
        return eagle_embeddings
    
    def generate_summary_report(
        self,
        output_dir: str,
        tiles_csv_path: str,
        output_report_path: str
    ):
        tiles_df = pd.read_csv(tiles_csv_path)
        eagle_embeddings = {}
        
        h5_files = glob.glob(os.path.join(output_dir, "*_eagle_features.h5"))
        
        # Load features from each individual H5 file
        for h5_path in h5_files:
            try:
                slide_id = os.path.basename(h5_path).replace('_eagle_features.h5', '')
                with h5py.File(h5_path, 'r') as h5_file:
                    if 'features' in h5_file:
                        eagle_embeddings[slide_id] = h5_file['features'][:]
            except Exception as e:
                logger.warning(f"Could not load features from {h5_path}: {e}")
        
        stats = {
            'total_tiles': len(tiles_df),
            'total_slides': len(eagle_embeddings),
            'unique_slides': tiles_df['SLIDE_ID'].nunique(),
            'tiles_per_slide': tiles_df.groupby('SLIDE_ID').size().describe().to_dict(),
            'embedding_dimension': list(eagle_embeddings.values())[0].shape[0] if eagle_embeddings else 0,
            'categories': tiles_df['CATEGORY'].value_counts().to_dict() if 'CATEGORY' in tiles_df.columns else {},
            'diagnoses': tiles_df['DIAGNOSIS'].value_counts().to_dict() if 'DIAGNOSIS' in tiles_df.columns else {},
            'h5_files_found': len(h5_files)
        }
        
        # Write report
        with open(output_report_path, 'w') as f:
            f.write("EAGLE Feature Extraction Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Input tiles CSV: {tiles_csv_path}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"H5 files found: {stats['h5_files_found']}\n\n")
            
            f.write(f"Total tiles processed: {stats['total_tiles']}\n")
            f.write(f"Total slides processed: {stats['total_slides']}\n")
            f.write(f"Unique slides in tiles CSV: {stats['unique_slides']}\n")
            f.write(f"EAGLE embedding dimension: {stats['embedding_dimension']}\n\n")
            
            f.write("Tiles per slide statistics:\n")
            for key, value in stats['tiles_per_slide'].items():
                f.write(f"  {key}: {value:.2f}\n")
            f.write("\n")
            
            if stats['categories']:
                f.write("Category distribution:\n")
                for category, count in stats['categories'].items():
                    f.write(f"  {category}: {count}\n")
                f.write("\n")
            
            if stats['diagnoses']:
                f.write("Diagnosis distribution:\n")
                for diagnosis, count in stats['diagnoses'].items():
                    f.write(f"  {diagnosis}: {count}\n")
        
        logger.info(f"Summary report saved to: {output_report_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate EAGLE features from pre-extracted tiles")
    parser.add_argument("--tiles_table", required=True, help="Path to tiles table")
    parser.add_argument("--output_dir", required=True, help="Directory to save EAGLE feature H5 files")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--model_weights_dir", default="./model_weights", help="Directory containing Virchow2 weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing tiles")
    parser.add_argument("--groupby_column", default="SLIDE_ID", help="Column to group tiles by")
    parser.add_argument("--generate_report", action="store_true", help="Generate summary report")
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = EAGLEFeatureExtractor(
        device=args.device,
        model_weights_dir=args.model_weights_dir,
        batch_size=args.batch_size
    )
    
    logger.info("Loading Virchow2 model...")
    extractor.load_models()
    
    logger.info(f"Processing tiles CSV: {args.tiles_table}")
    
    embeddings = extractor.process_all_slides(
        tiles_csv_path=args.tiles_table,
        output_dir=args.output_dir,
        groupby_column=args.groupby_column
    )
    
    logger.info(f"EAGLE feature extraction completed!")
    logger.info(f"Processed {len(embeddings)} slides")
    logger.info(f"Results saved to: {args.output_dir}")
    
    if args.generate_report:
        report_path = os.path.join(args.output_dir, 'eagle_features_summary_report.txt')
        extractor.generate_summary_report(
            output_dir=args.output_dir,
            tiles_csv_path=args.tiles_table,
            output_report_path=report_path
        )

if __name__ == "__main__":
    main()