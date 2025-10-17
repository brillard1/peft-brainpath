"""
Slide Table Generator for EBRAIN Dataset

Generates EAGLE-compatible slide tables from EBRAIN dataset
for feature extraction pipeline.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .ebrain_dataset import EBRAINDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlideTableGenerator:
    """
    Generates slide tables compatible with EAGLE pipeline from EBRAIN dataset.
    """
    
    def __init__(self, ebrain_dataset: EBRAINDataset, output_dir: str):
        """
        Initialize slide table generator.
        
        Args:
            ebrain_dataset: EBRAINDataset instance
            output_dir: Directory to save slide tables
        """
        self.ebrain_dataset = ebrain_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Slide table generator initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def generate_individual_diagnosis_tables(self) -> Dict[str, str]:
        """
        Generate slide tables for each individual diagnosis.
        
        Returns:
            Dictionary mapping diagnosis names to slide table file paths
        """
        diagnosis_tables = {}
        
        for diagnosis in self.ebrain_dataset.get_diagnosis_list():
            # Create safe filename
            safe_diagnosis = diagnosis.replace("/", "_").replace(",", "_").replace(" ", "_")
            output_file = self.output_dir / f"slide_table_{safe_diagnosis}.csv"
            
            # Export slide table
            self.ebrain_dataset.export_slide_table(
                output_path=str(output_file),
                diagnosis_or_category=diagnosis
            )
            
            diagnosis_tables[diagnosis] = str(output_file)
        
        logger.info(f"Generated {len(diagnosis_tables)} individual diagnosis tables")
        return diagnosis_tables
    
    def generate_category_tables(self) -> Dict[str, str]:
        """
        Generate slide tables for broader categories.
        
        Returns:
            Dictionary mapping category names to slide table file paths
        """
        category_tables = {}
        
        for category in self.ebrain_dataset.get_category_list():
            output_file = self.output_dir / f"slide_table_{category}.csv"
            
            # Get slides for this category
            category_slides = self.ebrain_dataset.get_slides_by_category(category)
            
            if len(category_slides) > 0:
                # Export slide table
                self.ebrain_dataset.export_slide_table(
                    output_path=str(output_file),
                    diagnosis_or_category=category
                )
                category_tables[category] = str(output_file)
            else:
                logger.warning(f"No slides found for category: {category}")
        
        logger.info(f"Generated {len(category_tables)} category tables")
        return category_tables
    
    def generate_split_tables(
        self, 
        diagnosis_or_category: str,
        split_ratios: tuple = (0.7, 0.15, 0.15),
        split_by_patient: bool = True
    ) -> Dict[str, str]:
        """
        Generate train/val/test split tables for a diagnosis or category.
        
        Args:
            diagnosis_or_category: Diagnosis name or category name
            split_ratios: (train, val, test) ratios
            split_by_patient: Whether to split by patient
            
        Returns:
            Dictionary mapping split names to file paths
        """
        # Create safe filename
        safe_name = diagnosis_or_category.replace("/", "_").replace(",", "_").replace(" ", "_")
        
        split_tables = {}
        
        for split in ['train', 'val', 'test']:
            output_file = self.output_dir / f"slide_table_{safe_name}_{split}.csv"
            
            # Export split table
            self.ebrain_dataset.export_slide_table(
                output_path=str(output_file),
                diagnosis_or_category=diagnosis_or_category,
                split=split
            )
            
            split_tables[split] = str(output_file)
        
        logger.info(f"Generated split tables for {diagnosis_or_category}")
        return split_tables
    
    def generate_grade_based_tables(self) -> Dict[str, str]:
        """
        Generate slide tables based on tumor grades.
        
        Returns:
            Dictionary mapping grade categories to slide table file paths
        """
        grade_tables = {}
        
        # Define grade mappings
        grade_mappings = {
            'Low_Grade': ['I', 'II'],
            'High_Grade': ['III', 'IV'],
            'Grade_I': ['I'],
            'Grade_II': ['II'],
            'Grade_III': ['III'],
            'Grade_IV': ['IV']
        }
        
        for grade_category, grades in grade_mappings.items():
            # Filter slides by grade directly from the slides_by_diagnosis data
            grade_slides = []
            for diagnosis, slides in self.ebrain_dataset.slides_by_diagnosis.items():

                filtered_slides = slides[slides['grade'].astype(str).isin(grades)].copy()
                if len(filtered_slides) > 0:
                    grade_slides.append(filtered_slides)
            
            if grade_slides:
                combined_slides = pd.concat(grade_slides, ignore_index=True)
                
                # Create slide table
                slide_table = pd.DataFrame({
                    'PATIENT': combined_slides['pat_id'].astype(str),
                    'FILENAME': combined_slides['filename'],
                    'DIAGNOSIS': combined_slides['diagnosis'],
                    'GRADE': combined_slides['grade'],
                    'SLIDE_PATH': combined_slides['slide_path'].astype(str)
                })
                
                output_file = self.output_dir / f"slide_table_{grade_category}.csv"
                slide_table.to_csv(output_file, index=False)
                
                grade_tables[grade_category] = str(output_file)
                logger.info(f"Generated {grade_category} table with {len(slide_table)} slides")
        
        return grade_tables
    
    def generate_binary_classification_tables(self) -> Dict[str, str]:
        """
        Generate slide tables for common binary classification tasks.
        
        Returns:
            Dictionary mapping task names to slide table file paths
        """
        binary_tables = {}
        
        # Define binary classification tasks
        binary_tasks = {
            'Glioma_vs_Meningioma': {
                'positive': ['Astrocytoma', 'Glioblastoma', 'Oligodendroglioma', 'Low_Grade_Glioma', 'Other_Glioma'],
                'negative': ['Meningioma']
            },
            'Malignant_vs_Benign': {
                'positive': ['Glioblastoma', 'Astrocytoma', 'Lymphoma', 'Soft_Tissue', 'Metastasis', 'Embryonal_Tumour'],
                'negative': ['Meningioma', 'Nerve_Tumour', 'Pituitary', 'Other']
            },
            'Pediatric_vs_Adult': {
                'positive': ['Medulloblastoma', 'Low_Grade_Glioma'],
                'negative': ['Astrocytoma', 'Glioblastoma', 'Meningioma']  # Adult tumors
            }
        }
        
        for task_name, task_def in binary_tasks.items():
            task_slides = []
            
            # Collect positive class slides
            for category in task_def['positive']:
                category_slides = self.ebrain_dataset.get_slides_by_category(category)
                if len(category_slides) > 0:
                    category_slides = category_slides.copy()
                    category_slides['binary_label'] = 1
                    task_slides.append(category_slides)
            
            # Collect negative class slides
            for category in task_def['negative']:
                category_slides = self.ebrain_dataset.get_slides_by_category(category)
                if len(category_slides) > 0:
                    category_slides = category_slides.copy()
                    category_slides['binary_label'] = 0
                    task_slides.append(category_slides)
            
            if task_slides:
                combined_slides = pd.concat(task_slides, ignore_index=True)
                
                # Create slide table
                slide_table = pd.DataFrame({
                    'PATIENT': combined_slides['pat_id'].astype(str),
                    'FILENAME': combined_slides['filename'],
                    'DIAGNOSIS': combined_slides['diagnosis'],
                    'BINARY_LABEL': combined_slides['binary_label'],
                    'SLIDE_PATH': combined_slides['slide_path'].astype(str)
                })
                
                output_file = self.output_dir / f"slide_table_{task_name}.csv"
                slide_table.to_csv(output_file, index=False)
                
                binary_tables[task_name] = str(output_file)
                logger.info(f"Generated {task_name} table with {len(slide_table)} slides")
        
        return binary_tables
    
    def generate_all_tables(self) -> Dict[str, Dict[str, str]]:
        """
        Generate all types of slide tables.
        
        Returns:
            Dictionary with all generated tables organized by type
        """
        all_tables = {}
        
        # Generate individual diagnosis tables
        all_tables['individual_diagnoses'] = self.generate_individual_diagnosis_tables()
        
        # Generate category tables
        all_tables['categories'] = self.generate_category_tables()
        
        # Generate grade-based tables
        all_tables['grades'] = self.generate_grade_based_tables()
        
        # Generate binary classification tables
        all_tables['binary_tasks'] = self.generate_binary_classification_tables()
        
        # Generate a comprehensive summary
        self._generate_summary_table(all_tables)
        
        logger.info("Generated all slide tables")
        return all_tables
    
    def _generate_summary_table(self, all_tables: Dict[str, Dict[str, str]]):
        """Generate a summary table with statistics for all generated tables."""
        summary_data = []
        
        for table_type, tables in all_tables.items():
            for table_name, table_path in tables.items():
                # Read table and get statistics
                df = pd.read_csv(table_path)
                
                summary_data.append({
                    'Table_Type': table_type,
                    'Table_Name': table_name,
                    'Num_Slides': len(df),
                    'Num_Patients': df['PATIENT'].nunique() if 'PATIENT' in df.columns else 'N/A',
                    'File_Path': table_path
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "slide_tables_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Generated summary table: {summary_path}")
    
    def get_table_statistics(self, table_path: str) -> Dict:
        """
        Get statistics for a specific slide table.
        
        Args:
            table_path: Path to slide table CSV file
            
        Returns:
            Dictionary with table statistics
        """
        df = pd.read_csv(table_path)
        
        stats = {
            'num_slides': len(df),
            'num_patients': df['PATIENT'].nunique() if 'PATIENT' in df.columns else None,
            'diagnoses': df['DIAGNOSIS'].value_counts().to_dict() if 'DIAGNOSIS' in df.columns else None,
            'file_path': table_path
        }
        
        return stats 