import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import glob
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.utils import (
    load_chief_model,
    load_ctranspath_features,
    validate_slide_file,
    get_slide_level_for_mpp,
    extract_tiles_from_slide
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopKTileExtractor:
    def __init__(
        self,
        device: str = "cuda",
        model_weights_dir: str = None,
        ctranspath_features_dir: str = None,
        tiles_output_dir: str = None,
        top_k_tiles: int = 25,
        tile_size: int = 224,
        target_mpp: float = 2.0,
    ):
        self.device = torch.device(device)
        self.top_k_tiles = top_k_tiles
        self.tile_size = tile_size
        self.target_mpp = target_mpp
        
        self.chief_weights_path = str(Path(model_weights_dir) / "CHIEF_pretraining.pth")
        self.ctranspath_features_dir = ctranspath_features_dir
        self.tiles_output_dir = tiles_output_dir
        
        # Initialize model
        self.chief_model = None
        
        logger.info(f"Initialized Top-k Tile Extractor")
        logger.info(f"Device: {self.device}")
        logger.info(f"Top-k tiles: {self.top_k_tiles}")
        logger.info(f"Target MPP: {self.target_mpp}")
    
    def load_models(self):
        if self.chief_weights_path:
            self.chief_model = load_chief_model(self.chief_weights_path, self.device)
            logger.info("CHIEF model loaded successfully")
        else:
            raise FileNotFoundError(f"CHIEF weights not found: {self.chief_weights_path}")
    
    def get_ctranspath_features(
        self,
        slide_id: str,
        ctranspath_features_path: Optional[str] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            logger.debug(f"Loading CTransPath features from: {ctranspath_features_path}")
            features, coordinates = load_ctranspath_features(ctranspath_features_path)
            return features, coordinates
        except Exception as e:
            logger.warning(f"Failed to load existing CTransPath features for {slide_id}: {e}")
    
        return None
    
    def check_tiles_exist(self, slide_id: str) -> bool:
        slide_tiles_dir = os.path.join(self.tiles_output_dir, slide_id)
        if not os.path.exists(slide_tiles_dir):
            return False
        
        pattern = os.path.join(slide_tiles_dir, "*.png")
        existing_tiles = glob.glob(pattern)
        num_existing = len(existing_tiles)
        expected_tiles = self.top_k_tiles
        return num_existing == expected_tiles
    
    def extract_topk_tiles_for_slide(
        self,
        slide_path: str,
        slide_id: str,
        ctranspath_features_path: Optional[str] = None,
        diagnosis: Optional[str] = None,
        patient_id: Optional[str] = None,
        label_column: str = "CATEGORY",
        label_value: Optional[str] = None
    ) -> Optional[List[Dict]]:
        try:
            if not validate_slide_file(slide_path):
                logger.error(f"Invalid slide file: {slide_path}")
                return None
            
            # Get CTransPath features
            result = self.get_ctranspath_features(
                slide_id, ctranspath_features_path
            )
            
            if result is None:
                logger.error(f"Failed to obtain CTransPath features for {slide_id}")
                return None

            features, coordinates = result
            
            if len(features) == 0:
                logger.warning(f"No features found for {slide_id}")
                return None
            
            # Apply CHIEF attention to select top tiles
            features_gpu = features.to(self.device)
            
            with torch.no_grad():
                result = self.chief_model(features_gpu)
                attention_raw = result["attention_raw"].squeeze(0).cpu()
            
            # Select top-k tiles
            k = min(self.top_k_tiles, attention_raw.shape[0])
            topk_values, topk_indices = torch.topk(attention_raw, k)

            selected_coordinates = coordinates[topk_indices].numpy()
            selected_attention_scores = topk_values.numpy()

            pyramid_level = get_slide_level_for_mpp(slide_path, self.target_mpp)

            tile_info_list = []
            if self.tiles_output_dir:
                slide_tiles_dir = self.tiles_output_dir + "/" + slide_id
                os.makedirs(slide_tiles_dir, exist_ok=True)
                
                tile_images = extract_tiles_from_slide(
                    slide_path=slide_path,
                    coords=selected_coordinates,
                    tile_size=self.tile_size,
                    level=pyramid_level
                )

                    if not tile_images:
                        logger.error(f"Failed to extract tiles for {slide_id}")
                        return None
                    
                    for i, (tile_image, coord, attention_score) in enumerate(zip(tile_images, selected_coordinates, selected_attention_scores)):
                        tile_filename = f"{slide_id}_tile_{i:03d}.png"
                        tile_path = slide_tiles_dir + "/" + tile_filename
                        
                        if isinstance(tile_image, Image.Image):
                            tile_image.save(tile_path, format='PNG')
                        else:
                            if isinstance(tile_image, np.ndarray):
                                tile_image = Image.fromarray(tile_image)
                            tile_image.save(tile_path, format='PNG')
                        
                        tile_info = {
                            'SLIDE_ID': slide_id,
                            'PATIENT': patient_id or 'unknown',
                            'FILENAME': f"{slide_id}.ndpi",
                            'SLIDE_PATH': slide_path,
                            'TILE_INDEX': i,
                            'TILE_PATH': str(tile_path),
                            'COORD_X': int(coord[0]),
                            'COORD_Y': int(coord[1]),
                            'ATTENTION_SCORE': float(attention_score),
                            'PYRAMID_LEVEL': pyramid_level,
                            'TILE_SIZE': self.tile_size,
                            'TARGET_MPP': self.target_mpp
                        }
                        
                        tile_info[label_column] = label_value or 'unknown'
                        tile_info['DIAGNOSIS'] = diagnosis or 'unknown'
                        tile_info_list.append(tile_info)

            else:
                logger.info(f"No path to save tiles to disk")
        
            return tile_info_list

        except Exception as e:
            logger.error(f"Error processing slide {slide_id}: {e}")
            return None

    def process_slide_table(
        self,
        slide_table_path: str,
        output_csv_path: str,
        ctranspath_base_dir: Optional[str] = None,
        label_column: str = "CATEGORY"
    ) -> pd.DataFrame:

        slide_table = pd.read_csv(slide_table_path)
        logger.info(f"Processing {len(slide_table)} slides from {slide_table_path}")
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Process slides
        all_tile_info = []
        skipped_slides = 0
        
        for idx, row in tqdm(slide_table.iterrows(), total=len(slide_table), desc="Processing slides"):
            slide_id = os.path.splitext(row['FILENAME'])[0]
            slide_path = row.get('SLIDE_PATH', row['FILENAME'])
            patient_id = str(row.get('PATIENT', ''))
            
            label_value = row.get(label_column, '')
            diagnosis = row.get('DIAGNOSIS', '')
            
            # Check if tiles already exist and skip if requested
            if self.check_tiles_exist(slide_id):
                logger.info(f"Skipping {slide_id} - tiles already exist")
                skipped_slides += 1
                continue
            
            # Construct CTransPath features path
            ctranspath_path = None
            if ctranspath_base_dir:
                ctranspath_filename = row['FILENAME'].replace('.ndpi', '_ctranspath_features.h5')
                ctranspath_path = os.path.join(ctranspath_base_dir, ctranspath_filename)
            
            # Process slide
            tile_info_list = self.extract_topk_tiles_for_slide(
                slide_path=slide_path,
                slide_id=slide_id,
                ctranspath_features_path=ctranspath_path,
                diagnosis=diagnosis,
                patient_id=patient_id,
                label_column=label_column,
                label_value=label_value
            )
            
            if tile_info_list:
                all_tile_info.extend(tile_info_list)
            else:
                logger.warning(f"Failed to process slide: {slide_id}")

        # Create DataFrame and save
        tiles_df = pd.DataFrame(all_tile_info)
        tiles_df.to_csv(output_csv_path, index=False)
        processed_slides = len(slide_table) - skipped_slides
        logger.info(f"Total slides in table: {len(slide_table)}")
        logger.info(f"Skipped slides (already have tiles): {skipped_slides}")
        logger.info(f"Processed slides: {processed_slides}")
        logger.info(f"Generated {len(tiles_df)} top-k tiles")
        logger.info(f"Results saved to: {output_csv_path}")

        return tiles_df

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Top-k Tile Extraction Pipeline")
    parser.add_argument("--slide_table", required=True, help="Path to slide table CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file with tile paths")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--model_weights_dir", default="./model_weights", help="Directory containing model weights")
    parser.add_argument("--ctranspath_features_dir", default="./ctranspath_features", help="Directory containing CTransPath features")
    parser.add_argument("--tiles_output_dir", default="./chief_tiles", help="Directory to save extracted tiles (optional)")
    parser.add_argument("--top_k_tiles", type=int, default=25, help="Number of top tiles to select")
    parser.add_argument("--tile_size", type=int, default=224, help="Size of tiles to extract")
    parser.add_argument("--target_mpp", type=float, default=2.0, help="Target microns per pixel")
    parser.add_argument("--label_column", default="CATEGORY", help="Column name for labels in slide table")

    args = parser.parse_args()

    # Initialize extractor
    extractor = TopKTileExtractor(
        device=args.device,
        model_weights_dir=args.model_weights_dir,
        ctranspath_features_dir=args.ctranspath_features_dir,
        tiles_output_dir=args.tiles_output_dir,
        top_k_tiles=args.top_k_tiles,
        tile_size=args.tile_size,
        target_mpp=args.target_mpp,
    )

    extractor.load_models()

    logger.info(f"Processing slide table: {args.slide_table}")
    tiles_df = extractor.process_slide_table(
        slide_table_path=args.slide_table,
        output_csv_path=args.output_csv,
        ctranspath_base_dir=args.ctranspath_features_dir,
        label_column=args.label_column
    )

    logger.info(f"Generated {len(tiles_df)} tile records")

if __name__ == "__main__":
    main()