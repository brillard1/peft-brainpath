import os
import sys
import argparse
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.ebrain_dataset import EBRAINDataset
from dataset.slide_table_generator import SlideTableGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Initializing EBRAIN dataset from: {args.ebrain_root}")
    ebrain_dataset = EBRAINDataset(
        ebrain_root=args.ebrain_root,
        annotation_file=args.annotation_file
    )
    
    stats = ebrain_dataset.get_statistics()
    logger.info("Dataset Statistics:")
    logger.info(f"  Total slides: {stats['total_slides']}")
    logger.info(f"  Total patients: {stats['total_patients']}")
    logger.info(f"  Total diagnoses: {stats['diagnoses_count']}")
    logger.info(f"  Total categories: {stats['categories_count']}")
    
    # Initialize Slide Table Generator
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing slide table generator, output: {output_dir}")
    generator = SlideTableGenerator(
        ebrain_dataset=ebrain_dataset,
        output_dir=str(output_dir)
    )
    
    if args.table_types == 'all' or 'all' in args.table_types:
        logger.info("Generating all types of slide tables...")
        all_tables = generator.generate_all_tables()
        
        total_tables = sum(len(tables) for tables in all_tables.values())
        logger.info(f"Generated {total_tables} slide tables in total")
        
        for table_type, tables in all_tables.items():
            logger.info(f"  {table_type}: {len(tables)} tables")
    else:
        if 'individual' in args.table_types:
            logger.info("Generating individual diagnosis tables...")
            individual_tables = generator.generate_individual_diagnosis_tables()
            logger.info(f"Generated {len(individual_tables)} individual diagnosis tables")
        
        if 'categories' in args.table_types:
            logger.info("Generating category tables...")
            category_tables = generator.generate_category_tables()
            logger.info(f"Generated {len(category_tables)} category tables")
        
        if 'grades' in args.table_types:
            logger.info("Generating grade-based tables...")
            grade_tables = generator.generate_grade_based_tables()
            logger.info(f"Generated {len(grade_tables)} grade-based tables")
        
        if 'binary' in args.table_types:
            logger.info("Generating binary classification tables...")
            binary_tables = generator.generate_binary_classification_tables()
            logger.info(f"Generated {len(binary_tables)} binary classification tables")
    
    logger.info(f"All slide tables saved to: {output_dir}")
    logger.info("Check 'slide_tables_summary.csv' for overview of all generated tables")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate slide tables from EBRAIN dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "ebrain_root",
        nargs='?',
        help="Path to EBRAIN dataset root directory"
    )
    parser.add_argument(
        "output_dir", 
        nargs='?',
        help="Output directory for slide tables"
    )
    parser.add_argument(
        "--annotation-file",
        default="annotation.csv",
        help="Name of annotation file (default: annotation.csv)"
    )
    parser.add_argument(
        "--table-types",
        nargs='+',
        choices=['all', 'individual', 'categories', 'grades', 'binary'],
        default=['all'],
        help="Types of slide tables to generate (default: all)"
    )
    parser.add_argument(
        "--generate-splits",
        nargs='+',
        help="Generate train/val/test splits for specified diagnoses or categories"
    )
    parser.add_argument(
        "--list-targets",
        metavar="EBRAIN_ROOT",
        help="List available diagnoses and categories, then exit"
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.ebrain_root or not args.output_dir:
        parser.error("ebrain_root and output_dir are required (unless using --list-targets)")
    
    if not os.path.exists(args.ebrain_root):
        parser.error(f"EBRAIN dataset directory not found: {args.ebrain_root}")
    
    try:
        main(args)
        logger.info("Slide table generation completed successfully!")
    except Exception as e:
        logger.error(f"Failed to generate slide tables: {e}")
        sys.exit(1) 