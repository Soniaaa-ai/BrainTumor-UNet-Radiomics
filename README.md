# BrainTumor-UNet-Radiomics

Code repository accompanying the manuscript:

Patient-Level Cross-Validated Brain Tumor Segmentation with U-Net: A Radiomics-Based Framework for Anticipating Segmentation Failure

This repository contains the research code used in the manuscript and is provided for research transparency and peer-review purposes.

## Repository Contents

| File | Description |
|---|---|
| brain_tumor_3fold_fast.py | U-Net segmentation with patient-level 3-fold cross-validation |
| radiomics_lgg_final.py | Radiomic feature extraction and failure-prediction analysis |
| table7_correlation_and_classifier.py | Statistical analysis and classifier-related results |
| ICC_final_results.csv | Radiomic feature stability results |
| radiomics_features_lgg.csv | Extracted radiomic features |
| all_folds_combined.csv | Combined cross-validation results |

## Dataset

The experiments use the publicly available LGG MRI Segmentation Dataset, originally provided through The Cancer Imaging Archive (TCIA) and distributed through Kaggle.

The dataset itself is not included in this repository.

## Code Overview

The repository includes code for:

- Patient-level cross-validation
- U-Net-based brain tumor segmentation
- Radiomic feature extraction
- Radiomic feature stability analysis
- Segmentation-failure prediction
- Statistical and correlation analyses

## Reproducibility

The main analysis scripts are provided directly in this repository.

```bash
python brain_tumor_3fold_fast.py
python radiomics_lgg_final.py
python table7_correlation_and_classifier.py
