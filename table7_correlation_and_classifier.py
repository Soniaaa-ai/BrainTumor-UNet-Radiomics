# ============================================================
# Table VII (tumor-characteristic correlations, slice- and
# patient-level) and Table VIII (XGBoost failure-prediction
# classifier, 16 stable features vs. all 18) for the LGG
# patient-level CV study.
#
# Reads:
#   - brain_tumor_3fold/all_folds_combined.csv     (segmentation CV scores)
#   - final radiomics/radiomics_features_lgg.csv   (radiomic features)
# produced by brain_tumor_3fold_fast.py and radiomics_lgg_final.py.
#
# Both source CSVs are defensively de-duplicated on
# (Image, Patient, Fold) before use -- see the note in
# brain_tumor_3fold_fast.py's find_all_pairs() about why
# duplicates could appear upstream.
#
# CPU only. Runtime: a few minutes.
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import xgboost as xgb

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

RADIOMICS_PATH = "/content/drive/MyDrive/final radiomics/radiomics_features_lgg.csv"
SCORES_PATH    = "/content/drive/MyDrive/brain_tumor_3fold/all_folds_combined.csv"

# ============================================================
# STEP 0: Load and de-duplicate both source files, then merge
# ============================================================
radiomics_df = pd.read_csv(RADIOMICS_PATH)
radiomics_df = radiomics_df.drop_duplicates(subset=['Image', 'Patient', 'Fold']).reset_index(drop=True)

scores_df = pd.read_csv(SCORES_PATH)
scores_df = scores_df.drop_duplicates(subset=['Image', 'Patient', 'Fold']).reset_index(drop=True)

merged = pd.merge(radiomics_df, scores_df, on=['Image', 'Patient', 'Fold'], how='inner')
print(f"Clean merged rows: {len(merged)}")
print(f"Unique patients: {merged['Patient'].nunique()}")

# ============================================================
# STEP 1: TABLE VII -- correlation between tumor characteristics
# and segmentation Dice, at slice level AND patient level
# ============================================================
factors = {
    'tumor_size (shape_Area)':                 'shape_Area',
    'signal_energy (firstorder_Energy)':       'firstorder_Energy',
    'intensity_contrast (firstorder_Variance)': 'firstorder_Variance',
    'shape_sphericity':                        'shape_Sphericity',
    'shape_extent':                            'shape_Extent',
}

print("\n" + "=" * 70)
print(f"TABLE VII -- SLICE-LEVEL (n={len(merged)})")
print("=" * 70)
slice_results = {}
for label, col in factors.items():
    r, p = pearsonr(merged[col], merged['Dice'])
    slice_results[label] = (r, p)
    print(f"  {label:42}: r={r:+.3f}, p={p:.3e}")

patient_level = merged.groupby('Patient')[list(factors.values()) + ['Dice']].mean()

print("\n" + "=" * 70)
print(f"TABLE VII -- PATIENT-LEVEL ROBUSTNESS CHECK (n={len(patient_level)})")
print("=" * 70)
for label, col in factors.items():
    r, p = pearsonr(patient_level[col], patient_level['Dice'])
    sig = "significant" if p < 0.05 else "NOT significant"
    print(f"  {label:42}: r={r:+.3f}, p={p:.4g}  ({sig})")

# ============================================================
# STEP 2: TABLE VIII -- XGBoost failure-prediction classifier
# Only true radiomic features are used as inputs -- NEVER any
# segmentation-performance column (F1/Jaccard/Recall/Precision/
# F2/HD95/Dice), since Dice directly defines the "failure" label
# and the others are deterministic functions of it: including
# them would leak the target into the inputs.
# ============================================================
feat_df = merged.copy()
feat_df["failure"] = (feat_df["Dice"] < 0.5).astype(int)
print(f"\nClass balance: {feat_df['failure'].sum()} failure slices out of {len(feat_df)} "
      f"({100 * feat_df['failure'].mean():.1f}%)")

ALL_18_FEATURES = [
    'firstorder_Mean', 'firstorder_Variance', 'firstorder_Skewness', 'firstorder_Kurtosis',
    'firstorder_Min', 'firstorder_Max', 'firstorder_Energy',
    'shape_Area', 'shape_Perimeter', 'shape_Sphericity', 'shape_Compactness',
    'shape_Extent', 'shape_BoundingBoxArea',
    'glcm_Contrast', 'glcm_Homogeneity', 'glcm_Energy', 'glcm_Correlation', 'glcm_Entropy',
]
# The 2 features dropped for low ICC stability (< 0.75) in Table IX
UNSTABLE_FEATURES = ['firstorder_Skewness', 'firstorder_Kurtosis']
STABLE_16_FEATURES = [c for c in ALL_18_FEATURES if c not in UNSTABLE_FEATURES]


def run_classifier_cv(feat_df, feature_cols, label=""):
    all_preds, all_true, all_proba = [], [], []
    fold_aucs = []

    for fold_idx in sorted(feat_df["Fold"].unique()):
        test_mask = feat_df["Fold"] == fold_idx
        train_mask = ~test_mask

        X_train_raw = feat_df.loc[train_mask, feature_cols].astype(float).fillna(0)
        X_test_raw = feat_df.loc[test_mask, feature_cols].astype(float).fillna(0)
        y_train = feat_df.loc[train_mask, "failure"]
        y_test = feat_df.loc[test_mask, "failure"]

        mu, sigma = X_train_raw.mean(), X_train_raw.std().replace(0, 1)
        X_train = (X_train_raw - mu) / sigma
        X_test = (X_test_raw - mu) / sigma

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=pos_weight,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        all_preds.extend(y_pred); all_true.extend(y_test); all_proba.extend(y_proba)

        auc = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan")
        fold_aucs.append(auc)
        print(f"  [{label}] Fold {fold_idx}: n={len(y_test)}, failures={int(y_test.sum())}, AUC={auc:.4f}")

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_proba = np.array(all_proba)

    pooled_auc = roc_auc_score(all_true, all_proba)
    pooled_acc = accuracy_score(all_true, all_preds)
    pooled_prec = precision_score(all_true, all_preds, zero_division=0)
    pooled_rec = recall_score(all_true, all_preds, zero_division=0)
    pooled_f1 = f1_score(all_true, all_preds, zero_division=0)
    cm = confusion_matrix(all_true, all_preds)

    print(f"\n[{label}] POOLED: AUC={pooled_auc:.4f}, Acc={pooled_acc:.4f}, "
          f"Prec={pooled_prec:.4f}, Rec={pooled_rec:.4f}, F1={pooled_f1:.4f}")
    print(f"[{label}] Per-fold AUC: {np.mean(fold_aucs):.4f} \u00b1 {np.std(fold_aucs):.4f}")
    print(f"[{label}] Confusion Matrix:\n{cm}")

    return {
        "pooled_auc": pooled_auc, "pooled_acc": pooled_acc, "pooled_prec": pooled_prec,
        "pooled_rec": pooled_rec, "pooled_f1": pooled_f1,
        "perfold_auc_mean": np.mean(fold_aucs), "perfold_auc_std": np.std(fold_aucs),
        "n_total": len(feat_df), "n_failures": int(feat_df["failure"].sum()),
    }


print("\n" + "=" * 70)
print("TABLE VIII -- 16 STABLE FEATURES (ICC-filtered, primary result)")
print("=" * 70)
results_16 = run_classifier_cv(feat_df, STABLE_16_FEATURES, label="16-feature")

print("\n" + "=" * 70)
print("SUPPLEMENTARY CHECK -- ALL 18 FEATURES (sanity check, Sec. V-D)")
print("=" * 70)
results_18 = run_classifier_cv(feat_df, ALL_18_FEATURES, label="18-feature")

print("\n" + "=" * 70)
print("16-feature vs 18-feature pooled AUC comparison")
print("=" * 70)
print(f"  16 features: {results_16['pooled_auc']:.4f}")
print(f"  18 features: {results_18['pooled_auc']:.4f}")

# ============================================================
# STEP 3: Feature importance (Fig. 3) -- refit on ALL data,
# 16-feature model, for interpretation only
# ============================================================
X_all_raw = feat_df[STABLE_16_FEATURES].astype(float).fillna(0)
mu_all, sigma_all = X_all_raw.mean(), X_all_raw.std().replace(0, 1)
X_all = (X_all_raw - mu_all) / sigma_all
y_all = feat_df["failure"]
final_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    scale_pos_weight=(y_all == 0).sum() / max((y_all == 1).sum(), 1))
final_model.fit(X_all, y_all)
importance = pd.Series(final_model.feature_importances_, index=STABLE_16_FEATURES).sort_values(ascending=False)
print("\nTop 10 features (importance from full-data fit, for Fig. 3):")
print(importance.head(10))

# ---- Fig. 3 ----
import matplotlib.pyplot as plt
top10 = importance.head(10)
plt.figure(figsize=(8, 5))
plt.barh(top10.index[::-1], top10.values[::-1])
plt.xlabel("Feature Importance (gain)")
plt.title("XGBoost Feature Importance for Predicting Segmentation Failure")
plt.tight_layout()
plt.savefig("fig3_feature_importance.png", dpi=300)
plt.show()

# ---- Save everything ----
OUT_DIR = "/content/drive/MyDrive/brain_tumor_3fold/table7_table8_final"
import os
os.makedirs(OUT_DIR, exist_ok=True)

pd.DataFrame([
    {"Factor": k, "Slice_r": v[0], "Slice_p": v[1],
     "Patient_r": pearsonr(patient_level[factors[k]], patient_level['Dice'])[0],
     "Patient_p": pearsonr(patient_level[factors[k]], patient_level['Dice'])[1]}
    for k, v in slice_results.items()
]).to_csv(os.path.join(OUT_DIR, "table7_final.csv"), index=False)

pd.DataFrame([
    {"Model": "16-feature (primary)", **results_16},
    {"Model": "18-feature (sanity check)", **results_18},
]).to_csv(os.path.join(OUT_DIR, "table8_final.csv"), index=False)

importance.to_csv(os.path.join(OUT_DIR, "feature_importance_final.csv"))

print(f"\nAll final tables saved to: {OUT_DIR}")

try:
    from google.colab import files
    files.download(os.path.join(OUT_DIR, "table7_final.csv"))
    files.download(os.path.join(OUT_DIR, "table8_final.csv"))
    files.download("fig3_feature_importance.png")
except Exception as e:
    print("Files saved to Google Drive:", e)
