[README (1).md](https://github.com/user-attachments/files/32413505/README.1.md)
# BrainTumor-UNet-Radiomics# Code for: Patient-Level Cross-Validated Brain Tumor Segmentation with U-Net

Pipeline order:

1. **`brain_tumor_3fold_fast.py`** — trains the U-Net segmentation model
   with 3-fold patient-level cross-validation and evaluates it on each
   held-out test fold. Produces `all_folds_combined.csv`.
2. **`radiomics_lgg_final.py`** — extracts radiomic features (first-order,
   shape, GLCM) from every scored tumor slice and runs the 16-feature
   XGBoost failure-prediction classifier. Produces `radiomics_features_lgg.csv`.
3. **`table7_correlation_and_classifier.py`** — generates the final
   Table VII (tumor-characteristic correlations, slice- and patient-level)
   and Table VIII (16- vs. 18-feature classifier comparison, Fig. 3).

## Data note

The dataset download (`kagglehub: mateuszbuda/lgg-mri-segmentation`)
extracts with the full `kaggle_3m` image/mask tree duplicated inside an
extra nested folder, so a plain recursive file search finds every slice
twice. `find_all_pairs()` in `brain_tumor_3fold_fast.py` dedupes by
filename to correct for this; `radiomics_lgg_final.py` and
`table7_correlation_and_classifier.py` also defensively drop duplicate
rows (on `Image`, `Patient`, `Fold`) from the CSVs they read, so the
pipeline produces correct, non-duplicated sample sizes even if re-run
independently.
