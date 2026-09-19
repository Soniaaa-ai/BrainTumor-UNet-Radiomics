# ============================================================
# RADIOMICS + XGBoost on the NEW (LGG, patient-level CV) dataset
# Reuses per-image Dice already computed in the 3-fold CV
# (all_folds_combined.csv) as ground truth for success/failure.
#
# Uses the SAME 3 patient-level folds as the segmentation CV,
# so the failure-prediction classifier is evaluated with the
# same no-leakage rigor Dr. Wang asked for.
#
# CPU only - no GPU needed. Runtime: ~30-45 minutes.
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys, subprocess

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["kagglehub", "xgboost", "scikit-image"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        pip_install(pkg)

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import xgboost as xgb
import kagglehub

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/brain_tumor_3fold"
COMBINED_SCORE_PATH = os.path.join(BASE_DIR, "all_folds_combined.csv")
OUT_DIR = os.path.join(BASE_DIR, "radiomics_lgg")
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.exists(COMBINED_SCORE_PATH), "Run the 3-fold CV segmentation script first!"
score_df = pd.read_csv(COMBINED_SCORE_PATH)

# NOTE: the segmentation CV script's file-discovery step used to pick up
# every image/mask pair twice (see the fix in find_all_pairs() in
# brain_tumor_3fold_fast.py), so each scored slice could appear twice in
# all_folds_combined.csv with identical values. Dedupe defensively here
# too, so this script produces correct counts even if run against an
# older, un-deduplicated combined-scores file.
before = len(score_df)
score_df = score_df.drop_duplicates(subset=['Image','Patient','Fold']).reset_index(drop=True)
if len(score_df) != before:
    print(f"Removed {before - len(score_df)} duplicate score rows "
          f"({before} -> {len(score_df)}).")
print(f"Loaded {len(score_df)} tumor-containing test-slice scores from segmentation CV")

DATASET_PATH = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

def get_patient_id(filepath):
    base = os.path.basename(filepath)
    parts = base.replace('.tif','').replace('.png','').split('_')
    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '_'.join(parts[:3])
    return base

# ---- Locate the actual image/mask file for each scored slice ----
def find_all_pairs(base_path):
    images = sorted(glob(os.path.join(base_path, "**", "*.tif"), recursive=True))
    images = [f for f in images if '_mask' not in f]
    masks  = [f.replace('.tif', '_mask.tif') for f in images]
    paired = [(img, msk) for img, msk in zip(images, masks) if os.path.exists(msk)]
    return {os.path.basename(p[0]): p for p in paired}

pairs_lookup = find_all_pairs(DATASET_PATH)
print(f"Found {len(pairs_lookup)} total image-mask pairs in dataset")

# ============================================================
# STEP 1: Extract radiomic features (first-order, shape, GLCM)
# from the ground-truth mask of every scored tumor-slice
# ============================================================
def extract_radiomics(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    mask_bin = (mask > 127).astype(np.uint8)
    if mask_bin.sum() < 8:
        return None

    feats = {}
    roi = image[mask_bin == 1].astype(np.float64)
    feats["firstorder_Mean"]     = roi.mean()
    feats["firstorder_Variance"] = roi.var()
    feats["firstorder_Skewness"] = skew(roi) if len(roi) > 2 else 0.0
    feats["firstorder_Kurtosis"] = kurtosis(roi) if len(roi) > 2 else 0.0
    feats["firstorder_Min"]      = roi.min()
    feats["firstorder_Max"]      = roi.max()
    feats["firstorder_Energy"]   = float(np.sum(roi ** 2))

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area < 1 or perimeter < 1:
        return None
    sphericity  = (4 * np.pi * area) / (perimeter ** 2 + 1e-9)
    compactness = (perimeter ** 2) / (area + 1e-9)
    x, y, bw, bh = cv2.boundingRect(cnt)
    bbox_area = bw * bh
    extent = area / (bbox_area + 1e-9)

    feats["shape_Area"] = area
    feats["shape_Perimeter"] = perimeter
    feats["shape_Sphericity"] = sphericity
    feats["shape_Compactness"] = compactness
    feats["shape_Extent"] = extent
    feats["shape_BoundingBoxArea"] = bbox_area

    ys, xs = np.where(mask_bin == 1)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    roi_img = image[y0:y1+1, x0:x1+1]
    roi_q = np.clip((roi_img.astype(np.float64) / 256.0 * 32).astype(np.uint8), 0, 31)
    try:
        glcm = graycomatrix(roi_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                             levels=32, symmetric=True, normed=True)
        feats["glcm_Contrast"]    = float(graycoprops(glcm, "contrast").mean())
        feats["glcm_Homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
        feats["glcm_Energy"]      = float(graycoprops(glcm, "energy").mean())
        feats["glcm_Correlation"] = float(np.nan_to_num(graycoprops(glcm, "correlation")).mean())
        glcm_mean = glcm.mean(axis=(2, 3))
        glcm_norm = glcm_mean / (glcm_mean.sum() + 1e-12)
        feats["glcm_Entropy"] = float(-np.sum(glcm_norm * np.log2(glcm_norm + 1e-12)))
    except Exception:
        feats["glcm_Contrast"] = feats["glcm_Homogeneity"] = feats["glcm_Energy"] = 0.0
        feats["glcm_Correlation"] = feats["glcm_Entropy"] = 0.0

    return feats

print("\nExtracting radiomic features for every scored tumor slice...")
rows = []
for _, row in tqdm(score_df.iterrows(), total=len(score_df)):
    name = row["Image"]
    if name not in pairs_lookup:
        continue
    img_path, mask_path = pairs_lookup[name]
    feats = extract_radiomics(img_path, mask_path)
    if feats is None:
        continue
    feats["Image"]   = name
    feats["Patient"] = row["Patient"]
    feats["Fold"]    = row["Fold"]          # SAME fold assignment as segmentation CV
    feats["Dice"]    = row["F1"]
    rows.append(feats)

feat_df = pd.DataFrame(rows)
feat_df.to_csv(os.path.join(OUT_DIR, "radiomics_features_lgg.csv"), index=False)
print(f"\nExtracted {feat_df.shape[1]-4} features for {len(feat_df)} slices "
      f"across {feat_df['Patient'].nunique()} patients")

# ============================================================
# STEP 2: Define failure and standardize features (z-score,
# fit on train folds only per CV split -- avoids leakage)
# ============================================================
feat_df["failure"] = (feat_df["Dice"] < 0.5).astype(int)
print(f"\nClass balance: {feat_df['failure'].sum()} failure slices out of {len(feat_df)} "
      f"({100*feat_df['failure'].mean():.1f}%)")

feature_cols = [c for c in feat_df.columns
                if c not in ["Image", "Patient", "Fold", "Dice", "failure"]]

# ============================================================
# STEP 3: 3-fold patient-level CV for the XGBoost classifier,
# using the SAME fold assignment as the segmentation model
# (Fold column already carried over) -- no leakage, and directly
# comparable/consistent with the segmentation CV methodology.
# ============================================================
all_preds, all_true, all_proba = [], [], []
fold_results = []

for fold_idx in sorted(feat_df["Fold"].unique()):
    test_mask  = feat_df["Fold"] == fold_idx
    train_mask = ~test_mask

    X_train_raw = feat_df.loc[train_mask, feature_cols].astype(float).fillna(0)
    X_test_raw  = feat_df.loc[test_mask,  feature_cols].astype(float).fillna(0)
    y_train = feat_df.loc[train_mask, "failure"]
    y_test  = feat_df.loc[test_mask,  "failure"]

    # Standardize features: fit scaler on TRAIN fold only (no leakage)
    mu, sigma = X_train_raw.mean(), X_train_raw.std().replace(0, 1)
    X_train = (X_train_raw - mu) / sigma
    X_test  = (X_test_raw  - mu) / sigma

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
        scale_pos_weight=pos_weight,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    all_preds.extend(y_pred); all_true.extend(y_test); all_proba.extend(y_proba)

    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = float("nan")
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    fold_results.append({"Fold": fold_idx, "N_test": len(y_test),
                          "N_failures": int(y_test.sum()),
                          "AUC": auc, "Accuracy": acc,
                          "Precision": prec, "Recall": rec, "F1": f1})
    print(f"\nFold {fold_idx}: n={len(y_test)}, failures={int(y_test.sum())}, "
          f"AUC={auc:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_csv(os.path.join(OUT_DIR, "xgboost_per_fold_results.csv"), index=False)

# ---- Pooled (across all 3 held-out folds) performance ----
all_true  = np.array(all_true)
all_preds = np.array(all_preds)
all_proba = np.array(all_proba)

pooled_auc  = roc_auc_score(all_true, all_proba)
pooled_acc  = accuracy_score(all_true, all_preds)
pooled_prec = precision_score(all_true, all_preds, zero_division=0)
pooled_rec  = recall_score(all_true, all_preds, zero_division=0)
pooled_f1   = f1_score(all_true, all_preds, zero_division=0)
cm = confusion_matrix(all_true, all_preds)

print(f"\n{'='*60}")
print("POOLED 3-FOLD CV RESULTS (every slice evaluated exactly once, held out)")
print(f"{'='*60}")
print(f"AUC       : {pooled_auc:.4f}")
print(f"Accuracy  : {pooled_acc:.4f}")
print(f"Precision : {pooled_prec:.4f}")
print(f"Recall    : {pooled_rec:.4f}")
print(f"F1        : {pooled_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"\nPer-fold AUC: {fold_results_df['AUC'].mean():.4f} \u00b1 {fold_results_df['AUC'].std():.4f}")

# ---- Feature importance: refit on ALL data for a final importance ranking ----
X_all_raw = feat_df[feature_cols].astype(float).fillna(0)
mu_all, sigma_all = X_all_raw.mean(), X_all_raw.std().replace(0, 1)
X_all = (X_all_raw - mu_all) / sigma_all
y_all = feat_df["failure"]
final_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    scale_pos_weight=(y_all==0).sum()/max((y_all==1).sum(),1))
final_model.fit(X_all, y_all)
importance = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features (importance from full-data fit, for interpretation only):")
print(importance.head(10))
importance.to_csv(os.path.join(OUT_DIR, "feature_importance_lgg.csv"))

summary = pd.DataFrame([{
    "Pooled_AUC": pooled_auc, "Pooled_Accuracy": pooled_acc,
    "Pooled_Precision": pooled_prec, "Pooled_Recall": pooled_rec, "Pooled_F1": pooled_f1,
    "PerFold_AUC_mean": fold_results_df["AUC"].mean(), "PerFold_AUC_std": fold_results_df["AUC"].std(),
    "N_total": len(feat_df), "N_failures": int(feat_df["failure"].sum()),
    "N_patients": feat_df["Patient"].nunique(),
}])
summary.to_csv(os.path.join(OUT_DIR, "xgboost_summary_lgg.csv"), index=False)

print(f"\nAll results saved to: {OUT_DIR}")

try:
    from google.colab import files
    files.download(os.path.join(OUT_DIR, "xgboost_summary_lgg.csv"))
    files.download(os.path.join(OUT_DIR, "xgboost_per_fold_results.csv"))
    files.download(os.path.join(OUT_DIR, "feature_importance_lgg.csv"))
    files.download(os.path.join(OUT_DIR, "radiomics_features_lgg.csv"))
except Exception as e:
    print("Files saved to Google Drive:", e)
