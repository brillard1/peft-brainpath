"""
Configures the EBRAIN (Digital Brain Tumour Atlas) dataset organization
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EBRAINDataset:
    def __init__(self, ebrain_root: str, annotation_file: str = "annotation.csv"):
        self.ebrain_root = Path(ebrain_root)
        self.version_path = self.ebrain_root / "v1.0"
        annotation_path = self.version_path / annotation_file
        self.annoation_path = annotation_path if annotation_path.exists() else None
        
        if self.annotation_path is None:
            raise FileNotFoundError(f"Annotation file '{annotation_file}' not found in {self.version_path}")
        
        logger.info(f"Using annotation file: {self.annotation_path}")
        
        # Load annotation data
        self.annotations = self._load_annotations()
        self.category_mappings = self._define_category_mappings()
        self.slides_by_diagnosis = self._organize_slides_by_diagnosis()
        
        logger.info(f"Loaded EBRAIN dataset with {len(self.annotations)} slides")
        logger.info(f"Found {len(self.slides_by_diagnosis)} diagnosis categories with existing slides")
    
    def _load_annotations(self) -> pd.DataFrame:
        """Load and validate annotation file."""
        annotations = pd.read_csv(self.annotation_path)
        
        # Validate required columns
        required_cols = ['uuid', 'pat_id', 'diagnosis', 'control']
        missing_cols = [col for col in required_cols if col not in annotations.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        annotations['diagnosis'] = annotations['diagnosis'].fillna('Control') # no diagnosis -> Control (normal)
        annotations['filename'] = annotations['uuid'] + '.ndpi'
        
        return annotations
    
    def _define_category_mappings(self) -> Dict[str, List[str]]:
        """
        Define broader category mappings for diagnosis grouping.
        """
        return {
            'Astrocytoma': [
                'Anaplastic astrocytoma, IDH-mutant',
                'Anaplastic astrocytoma, IDH-wildtype',
                'Diffuse astrocytoma, IDH-mutant',
                'Diffuse astrocytoma, IDH-wildtype',
                'Diffuse midline glioma, H3 K27M-mutant',
                'Gemistocytic astrocytoma, IDH-mutant'
            ],
            
            'Glioblastoma': [
                'Giant cell glioblastoma',
                'Glioblastoma, IDH-mutant',
                'Glioblastoma, IDH-wildtype',
                'Gliosarcoma'
            ],
            
            'Oligodendroglioma': [
                'Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeleted',
                'Oligodendroglioma, IDH-mutant and 1p/19q codeleted'
            ],
            
            'Low_Grade_Glioma': [
                'Angiocentric glioma',
                'Desmoplastic infantile astrocytoma and ganglioglioma',
                'Pilocytic astrocytoma',
                'Pilomyxoid astrocytoma',
                'Pleomorphic xanthoastrocytoma'
            ],
            
            'Other_Glioma': [
                'Anaplastic pleomorphic xanthoastrocytoma',
                'Astroblastoma',
                'Chordoid glioma of the third ventricle'
            ],
            
            'Meningioma': [
                'Anaplastic meningioma',
                'Angiomatous meningioma',
                'Atypical meningioma',
                'Chordoid meningioma',
                'Clear cell meningioma',
                'Fibrous meningioma',
                'Lymphoplasmacyte-rich meningioma',
                'Meningothelial meningioma',
                'Metaplastic meningioma',
                'Microcystic meningioma',
                'Papillary meningioma',
                'Psammomatous meningioma',
                'Rhabdoid meningioma',
                'Secretory meningioma',
                'Transitional meningioma'
            ],
            
            'Medulloblastoma': [
                'Medulloblastoma, SHH-activated and TP53-mutant',
                'Medulloblastoma, SHH-activated and TP53-wildtype',
                'Medulloblastoma, WNT-activated',
                'Medulloblastoma, non-WNT/non-SHH'
            ],
            
            'Ependymoma': [
                'Anaplastic ependymoma',
                'Ependymoma',
                'Ependymoma, RELA fusion-positive',
                'Myxopapillary ependymoma',
                'Papillary ependymoma',
                'Subependymoma',
                'Tanycytic ependymoma'
            ],
            
            'Glioneuronal': [
                'Anaplastic ganglioglioma',
                'Diffuse leptomeningeal glioneuronal tumour',
                'Dysembryoplastic neuroepithelial tumour',
                'Dysplastic cerebellar gangliocytoma',
                'Gangliocytoma',
                'Ganglioglioma',
                'Papillary glioneuronal tumour',
                'Rosette-forming glioneuronal tumour'
            ],
            
            'Neural_Tumour': [
                'Central neurocytoma',
                'Cerebellar liponeurocytoma',
                'Extraventricular neurocytoma',
                'Ganglioneuroma'
            ],
            
            'Pineal': [
                'Pineal parenchymal tumour of intermediate differentiation',
                'Pineocytoma'
            ],
            
            'Pituitary': [
                'Pituitary adenoma'
            ],
            
            'Nerve_Tumour': [
                'Cellular schwannoma',
                'Hybrid nerve sheath tumours',
                'Malignant peripheral nerve sheath tumour',
                'Melanotic schwannoma',
                'Neurofibroma',
                'Perineurioma',
                'Plexiform neurofibroma',
                'Schwannoma'
            ],
            
            'Lymphoma': [
                'Diffuse large B-cell lymphoma of the CNS',
                'EBV-positive diffuse large B-cell lymphoma, NOS',
                'Follicular lymphoma',
                'Immunodeficiency-associated CNS lymphoma',
                'Intravascular large B-cell lymphoma',
                'Low-grade B-cell lymphomas of the CNS',
                'MALT lymphoma of the dura',
                'T-cell and NK/T-cell lymphomas of the CNS'
            ],
            
            'Choroid_Plexus_Tumour': [
                'Atypical choroid plexus papilloma',
                'Choroid plexus carcinoma',
                'Choroid plexus papilloma'
            ],
            
            'Soft_Tissue': [
                'Angiosarcoma',
                'Chondrosarcoma',
                'Chordoma',
                'Epitheloid MPNST',
                'Ewing sarcoma',
                'Fibrosarcoma',
                'Haemangiopericytoma',
                'Leiomyoma',
                'Leiomyosarcoma',
                'Lipoma',
                'Liposarcoma',
                'Osteochondroma',
                'Osteoma',
                'Osteosarcoma',
                'Rhabdomyosarcoma',
                'Undifferentiated pleomorphic sarcoma'
            ],
            
            'Embryonal_Tumour': [
                'Atypical teratoid/rhabdoid tumour',
                'CNS ganglioneuroblastoma',
                'Embryonal tumour with multilayered rosettes, C19MC-altered',
                'Olfactory neuroblastoma',
                'Pineoblastoma'
            ],
            
            'Metastasis': [
                'Choriocarcinoma',
                'Embryonal carcinoma',
                'Metastatic tumours'
            ],
            
            'Melanocytic': [
                'Meningeal melanocytoma',
                'Meningeal melanoma'
            ],
            
            'Hematological': [
                'Erdheim-Chester disease',
                'Langerhans cell histiocytosis'
            ],
            
            'Other': [
                'Adamantinomatous craniopharyngioma',
                'Crystal-storing histiocytosis',
                'Germinoma',
                'Granular cell tumour of the sellar region',
                'Haemangioblastoma',
                'Haemangioma',
                'Immature teratoma',
                'Inflammatory myofibroblastic tumour',
                'Juvenile xanthogranuloma',
                'Mature teratoma',
                'Mixed germ cell tumour',
                'Papillary craniopharyngioma',
                'Papillary tumour of the pineal region',
                'Paraganglioma',
                'Pituicytoma',
                'Spindle cell oncocytoma',
                'Subependymal giant cell astrocytoma',
                'Teratoma with malignant transformation'
            ],

            'Control': [
                'Control'
            ]
        }
    
    def _diagnosis_to_folder_name(self, diagnosis: str) -> str:
        if diagnosis == "Embryonal tumour with multilayered rosettes, C19MC-altered":
            return "Embryonal tumour with multilayered rosette, C19MC-altered"

        return diagnosis.replace("/", "-")
    
    def _organize_slides_by_diagnosis(self) -> Dict[str, pd.DataFrame]:
        slides_by_diagnosis = {}
        
        for diagnosis in self.annotations['diagnosis'].unique():
            diagnosis_slides = self.annotations[
                self.annotations['diagnosis'] == diagnosis
            ].copy()
            
            folder_name = self._diagnosis_to_folder_name(diagnosis)
            
            diagnosis_slides['slide_path'] = diagnosis_slides.apply(
                lambda row: self.version_path / folder_name / row['filename'],
                axis=1
            )
            
            existing_slides = diagnosis_slides[
                diagnosis_slides['slide_path'].apply(lambda x: x.exists())
            ].copy()
            
            if len(existing_slides) > 0:
                slides_by_diagnosis[diagnosis] = existing_slides
                logger.info(f"{diagnosis}: {len(existing_slides)} slides")
            else:
                logger.warning(f"{diagnosis}: No existing slides found")
        
        return slides_by_diagnosis
    
    def get_slides_by_category(self, category: str) -> pd.DataFrame:
        if category not in self.category_mappings:
            raise ValueError(f"Unknown category: {category}")
        
        category_slides = []
        for diagnosis in self.category_mappings[category]:
            if diagnosis in self.slides_by_diagnosis:
                slides = self.slides_by_diagnosis[diagnosis].copy()
                slides['category'] = category
                category_slides.append(slides)
        
        if category_slides:
            return pd.concat(category_slides, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_diagnosis_list(self) -> List[str]:
        return list(self.slides_by_diagnosis.keys())
    
    def get_category_list(self) -> List[str]:
        return list(self.category_mappings.keys())
    
    def get_statistics(self) -> Dict:
        stats = {
            'total_slides': len(self.annotations),
            'total_patients': self.annotations['pat_id'].nunique(),
            'diagnoses_count': len(self.slides_by_diagnosis),
            'categories_count': len(self.category_mappings),
            'slides_per_diagnosis': {
                diagnosis: len(slides) 
                for diagnosis, slides in self.slides_by_diagnosis.items()
            },
            'slides_per_category': {
                category: len(self.get_slides_by_category(category))
                for category in self.category_mappings.keys()
            }
        }
        return stats
    
    def create_train_val_test_split(
        self, 
        diagnosis_or_category: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_by_patient: bool = True,
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits.
        Args:
            diagnosis_or_category: Diagnosis name or category name
            split_ratios: (train, val, test) ratios
            split_by_patient: If True, split by patient to avoid data leakage
            random_state: Random seed for reproducibility
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        # Get slides for the specified diagnosis or category
        if diagnosis_or_category in self.category_mappings:
            slides = self.slides_by_diagnosis[diagnosis_or_category].copy()
        elif diagnosis_or_category in self.slides_by_diagnosis:
            slides = self.get_slides_by_category(diagnosis_or_category)
        else:
            raise ValueError(f"Unknown diagnosis or category: {diagnosis_or_category}")
        
        if len(slides) == 0:
            raise ValueError(f"No slides found for: {diagnosis_or_category}")
        
        np.random.seed(random_state)
        
        if split_by_patient:
            # Split by patient to avoid data leakage
            unique_patients = slides['pat_id'].unique()
            np.random.shuffle(unique_patients)
            
            n_train = int(len(unique_patients) * split_ratios[0])
            n_val = int(len(unique_patients) * split_ratios[1])
            
            train_patients = unique_patients[:n_train]
            val_patients = unique_patients[n_train:n_train + n_val]
            test_patients = unique_patients[n_train + n_val:]
            
            train_slides = slides[slides['pat_id'].isin(train_patients)]
            val_slides = slides[slides['pat_id'].isin(val_patients)]
            test_slides = slides[slides['pat_id'].isin(test_patients)]
        else:
            # Split by slides directly
            slides_shuffled = slides.sample(frac=1, random_state=random_state)
            
            n_train = int(len(slides) * split_ratios[0])
            n_val = int(len(slides) * split_ratios[1])
            
            train_slides = slides_shuffled[:n_train]
            val_slides = slides_shuffled[n_train:n_train + n_val]
            test_slides = slides_shuffled[n_train + n_val:]
        
        logger.info(f"Split for {diagnosis_or_category}:")
        logger.info(f"  Train: {len(train_slides)} slides")
        logger.info(f"  Val: {len(val_slides)} slides")
        logger.info(f"  Test: {len(test_slides)} slides")
        
        return {
            'train': train_slides,
            'val': val_slides,
            'test': test_slides
        }
    
    def export_slide_table(
        self, 
        output_path: str, 
        diagnosis_or_category: str,
        split: Optional[str] = None
    ):
        splits = self.create_train_val_test_split(diagnosis_or_category)
        slides = splits[split]
        
        # Create EAGLE-compatible format
        slide_table = pd.DataFrame({
            'PATIENT': slides['pat_id'].astype(str),
            'FILENAME': slides['filename'],
            'DIAGNOSIS': slides['diagnosis'],
            'SLIDE_PATH': slides['slide_path'].astype(str)
        })
        
        slide_table.to_csv(output_path, index=False)
        logger.info(f"Exported slide table to: {output_path}")
        logger.info(f"  {len(slide_table)} slides included") 