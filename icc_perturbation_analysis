# ============================================================
# ICC (Intraclass Correlation) feature-stability analysis.
#
# For every extracted radiomic feature, this script checks how
# stable its value is under three small, realistic perturbations
# of the tumor mask -- erosion, dilation, and a 1px shift -- and
# computes ICC(3,1) across the four conditions (original + 3
# perturbations) for each feature. Features with ICC >= 0.75 are
# treated as stable enough to use; ICC >= 0.90 as excellent.
#
# NOTE: erosion (not dilation) is the perturbation that can shrink
# a small tumor's mask below the minimum-pixel threshold and
# eliminate it entirely. Dilation only grows the mask, so it never
# eliminates a slice; the small number of slices dropped from the
# panel (1,372 -> 1,368) are dropped because of erosion.
#
# Reads the already-deduplicated radiomics_features_lgg.csv
# produced by radiomics_lgg_final.py. Defensively deduplicates it
# again here too, so this script gives correct results even if
# pointed at an older, un-deduplicated copy of that file.
#
# CPU only. Estimated runtime: ~15-20 minutes.
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys, subprocess

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["kagglehub", "scikit-image", "pingouin"]:
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
import pingouin as pg
import kagglehub

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/brain_tumor_3fold"
OUT_DIR = os.path.join(BASE_DIR, "icc_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_PATH = os.path.join(BASE_DIR, "radiomics_lgg", "radiomics_features_lgg.csv")
if not os.path.exists(FEAT_PATH):
    FEAT_PATH = "/content/drive/MyDrive/final radiomics/radiomics_features_lgg.csv"
assert os.path.exists(FEAT_PATH), "Run radiomics_lgg_final.py first!"

orig_feat_df = pd.read_csv(FEAT_PATH)
before = len(orig_feat_df)
orig_feat_df = orig_feat_df.drop_duplicates(subset=['Image', 'Patient', 'Fold']).reset_index(drop=True)
if len(orig_feat_df) != before:
    print(f"Removed {before - len(orig_feat_df)} duplicate rows ({before} -> {len(orig_feat_df)}).")
print(f"Loaded {len(orig_feat_df)} original feature rows")

DATASET_PATH = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

def find_all_pairs(base_path):
    # Dict keyed by basename -- self-deduplicating against the
    # duplicate nested folder in the downloaded dataset (see the
    # note in brain_tumor_3fold_fast.py).
    images = sorted(glob(os.path.join(base_path, "**", "*.tif"), recursive=True))
    images = [f for f in images if '_mask' not in f]
    masks  = [f.replace('.tif', '_mask.tif') for f in images]
    paired = [(img, msk) for img, msk in zip(images, masks) if os.path.exists(msk)]
    return {os.path.basename(p[0]): p for p in paired}

pairs_lookup = find_all_pairs(DATASET_PATH)
feature_cols = [c for c in orig_feat_df.columns
                if c not in ["Image", "Patient", "Fold", "Dice", "failure"]]

def extract_radiomics(image, mask_bin):
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
    feats["shape_Area"] = area
    feats["shape_Perimeter"] = perimeter
    feats["shape_Sphericity"]  = (4 * np.pi * area) / (perimeter ** 2 + 1e-9)
    feats["shape_Compactness"] = (perimeter ** 2) / (area + 1e-9)
    x, y, bw, bh = cv2.boundingRect(cnt)
    feats["shape_Extent"] = area / (bw * bh + 1e-9)
    feats["shape_BoundingBoxArea"] = bw * bh

    ys, xs = np.where(mask_bin == 1)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    roi_img = image[y0:y1 + 1, x0:x1 + 1]
    roi_q = np.clip((roi_img.astype(np.float64) / 256.0 * 32).astype(np.uint8), 0, 31)
    try:
        glcm = graycomatrix(roi_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                             levels=32, symmetric=True, normed=True)
        feats["glcm_Contrast"]    = float(graycoprops(glcm, "contrast").mean())
        feats["glcm_Homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
        feats["glcm_Energy"]      = float(graycoprops(glcm, "energy").mean())
        feats["glcm_Correlation"] = float(np.nan_to_num(graycoprops(glcm, "correlation")).mean())
        gm = glcm.mean(axis=(2, 3)); gn = gm / (gm.sum() + 1e-12)
        feats["glcm_Entropy"] = float(-np.sum(gn * np.log2(gn + 1e-12)))
    except Exception:
        feats["glcm_Contrast"] = feats["glcm_Homogeneity"] = feats["glcm_Energy"] = 0.0
        feats["glcm_Correlation"] = feats["glcm_Entropy"] = 0.0
    return feats

# ---- Perturbations ----
def perturb_erode(mask, k=1):
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)

def perturb_dilate(mask, k=1):
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def perturb_shift(mask, dx=1, dy=1):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                           flags=cv2.INTER_NEAREST, borderValue=0)

PERTURBATIONS = {
    "erode":  lambda m: perturb_erode(m, 1),
    "dilate": lambda m: perturb_dilate(m, 1),
    "shift":  lambda m: perturb_shift(m, 1, 1),
}

print(f"\nExtracting perturbed features for {len(orig_feat_df)} slices "
      f"x {len(PERTURBATIONS)} perturbations...")

rows_by_perturbation = {name: [] for name in PERTURBATIONS}
n_failed = {name: 0 for name in PERTURBATIONS}

for _, row in tqdm(orig_feat_df.iterrows(), total=len(orig_feat_df)):
    name = row["Image"]
    if name not in pairs_lookup:
        continue
    img_path, mask_path = pairs_lookup[name]
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        continue
    mask_bin = (mask > 127).astype(np.uint8)

    for pname, pfunc in PERTURBATIONS.items():
        perturbed_mask = pfunc(mask_bin)
        feats = extract_radiomics(image, perturbed_mask)
        if feats is None:
            n_failed[pname] += 1
            continue
        feats["Image"] = name
        rows_by_perturbation[pname].append(feats)

print("\nPerturbation success/failure counts:")
for pname in PERTURBATIONS:
    n_ok = len(rows_by_perturbation[pname])
    print(f"  {pname}: {n_ok} succeeded, {n_failed[pname]} failed (mask eliminated or invalid)")

for pname in PERTURBATIONS:
    pert_df = pd.DataFrame(rows_by_perturbation[pname])
    pert_df.to_csv(os.path.join(OUT_DIR, f"perturbed_features_{pname}.csv"), index=False)
print(f"\nRaw perturbed features saved to {OUT_DIR}/perturbed_features_*.csv")
print("(If ICC computation below has any issue, these files let us recompute without re-extracting.)")

# ---- ICC(3,1) across original + 3 perturbations, per feature ----
icc_results = []

image_sets = [set(pd.DataFrame(rows_by_perturbation[p])["Image"]) for p in PERTURBATIONS]
common_images = set(orig_feat_df["Image"])
for s in image_sets:
    common_images &= s
print("\nCommon images (valid under all 3 perturbations):", len(common_images))

for feat in feature_cols:
    long_rows = []
    orig_lookup = dict(zip(orig_feat_df["Image"], orig_feat_df[feat]))

    pert_lookup = {}
    for pname in PERTURBATIONS:
        df = pd.DataFrame(rows_by_perturbation[pname])
        pert_lookup[pname] = dict(zip(df["Image"], df[feat]))

    for img in common_images:
        long_rows.append({"Image": img, "rater": "original", "value": orig_lookup[img]})
        for pname in PERTURBATIONS:
            long_rows.append({
                "Image": img,
                "rater": pname,
                "value": pert_lookup[pname][img],
            })

    long_df = pd.DataFrame(long_rows)

    try:
        icc_table = pg.intraclass_corr(data=long_df, targets="Image", raters="rater", ratings="value")
        icc_row = icc_table[icc_table["Type"] == "ICC(C,1)"].iloc[0]
        icc_value = float(icc_row["ICC"])
        icc_results.append({"Feature": feat, "ICC": icc_value, "N_images": len(common_images)})
        print(feat, "ICC =", icc_value)
    except Exception as e:
        print("FAILED:", feat, repr(e))

icc_df = pd.DataFrame(icc_results)
print("\n========== FINAL ICC RESULTS ==========")
print(icc_df.to_string(index=False))

icc_df["Stable_0.75"] = (icc_df["ICC"] >= 0.75)
icc_df["Stable_0.90"] = (icc_df["ICC"] >= 0.90)

print("\n========== SUMMARY ==========")
print("Valid ICC:", icc_df["ICC"].notna().sum(), "/", len(icc_df))
print("ICC >=0.75:", icc_df["Stable_0.75"].sum())
print("ICC >=0.90:", icc_df["Stable_0.90"].sum())

save_path = os.path.join(OUT_DIR, "ICC_final_results.csv")
icc_df.to_csv(save_path, index=False)
print("Saved:", save_path)

try:
    from google.colab import files
    files.download(save_path)
except Exception as e:
    print("Saved to Drive:", e)
