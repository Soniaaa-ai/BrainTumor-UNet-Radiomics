# ============================================================
# Brain Tumor Segmentation - U-Net with 3-FOLD PATIENT-LEVEL CV
# (Faster version: 3 folds instead of 5, tumor-only validation
#  metric to avoid empty-slice noise, reduced epoch budget)
#
# Dataset: mateuszbuda/lgg-mri-segmentation (110 patients, TCGA)
#
# KEY FIX vs previous version:
#   Validation Dice is now computed ONLY on tumor-containing
#   slices. Previously it included ~64% empty-mask slices,
#   where even tiny false-positive noise crushes that slice's
#   Dice toward 0 -- this made the training curve look stuck
#   even while the model was actually learning normally on
#   real tumor cases.
#
# Fully resumable: if disconnected mid-fold, just re-run this
# cell -- it resumes training from the last checkpoint, then
# continues to the next fold automatically.
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys, subprocess

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import kagglehub
except ImportError:
    pip_install("kagglehub")
    import kagglehub

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/brain_tumor_3fold"
os.makedirs(BASE_DIR, exist_ok=True)

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
    MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (ModelCheckpoint, CSVLogger,
    ReduceLROnPlateau, EarlyStopping)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import CustomObjectScope

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

print("\nDownloading LGG MRI Segmentation dataset (TCGA)...")
DATASET_PATH = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
print("Dataset path:", DATASET_PATH)

# ---- Config (faster settings) ----
H, W     = 256, 256
BATCH    = 8
LR       = 1e-3
EPOCHS   = 80           # reduced: prior run showed plateau by ~epoch 80-100
PATIENCE = 20           # reduced accordingly
SEED     = 42
N_FOLDS  = 3            # reduced from 5
np.random.seed(SEED)
tf.random.set_seed(SEED)

smooth = 1e-6
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

CUSTOM_OBJS = {"dice_coef": dice_coef, "dice_loss": dice_loss, "bce_dice_loss": bce_dice_loss}

def get_patient_id(filepath):
    base = os.path.basename(filepath)
    parts = base.replace('.tif','').replace('.png','').split('_')
    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '_'.join(parts[:3])
    return base

def find_all_pairs(base_path):
    # NOTE: kagglehub's extracted archive for this dataset contains the
    # whole "kaggle_3m" tree TWICE, nested inside an extra top-level
    # "lgg-mri-segmentation" folder (a duplicate copy of the same files
    # on disk). A plain recursive glob therefore finds every image/mask
    # pair twice. We dedupe by basename (the TCGA filename is unique
    # across the dataset) so each real slice is only counted once.
    images = sorted(glob(os.path.join(base_path, "**", "*.tif"), recursive=True))
    images = [f for f in images if '_mask' not in f]
    masks  = [f.replace('.tif', '_mask.tif') for f in images]
    paired = [(img, msk) for img, msk in zip(images, masks) if os.path.exists(msk)]
    if not paired:
        images = sorted(glob(os.path.join(base_path, "**", "*.png"), recursive=True))
        images = [f for f in images if '_mask' not in f and 'mask' not in f.lower()]
        paired = [(img, img.replace('.png','_mask.png')) for img in images
                  if os.path.exists(img.replace('.png','_mask.png'))]

    seen = {}
    for img, msk in paired:
        seen[os.path.basename(img)] = (img, msk)
    paired = list(seen.values())

    return [p[0] for p in paired], [p[1] for p in paired]

def has_tumor(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return False
    return (m > 127).sum() > 0

all_images, all_masks = find_all_pairs(DATASET_PATH)
all_patients = [get_patient_id(f) for f in all_images]
unique_patients = sorted(set(all_patients))
print(f"\nTotal: {len(all_images)} slices from {len(unique_patients)} unique patients")

gkf = GroupKFold(n_splits=N_FOLDS)
fold_assignments = list(gkf.split(all_images, groups=all_patients))
print(f"Built {N_FOLDS} patient-level folds.")
for i, (_, test_idx) in enumerate(fold_assignments):
    fold_patients = set(all_patients[j] for j in test_idx)
    print(f"  Fold {i}: {len(test_idx)} slices from {len(fold_patients)} patients")

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None: return np.zeros((H, W, 3), dtype=np.float32)
    x = cv2.resize(x, (W, H))
    return (x / 255.0).astype(np.float32)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if x is None: return np.zeros((H, W, 1), dtype=np.float32)
    x = cv2.resize(x, (W, H))
    x = (x / 255.0).astype(np.float32)
    return np.expand_dims(x, axis=-1)

def tf_parse(x, y):
    x, y = tf.numpy_function(lambda a, b: (read_image(a), read_mask(b)),
                              [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3]); y.set_shape([H, W, 1])
    return x, y

def augment_fn(x, y):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x); y = tf.image.flip_left_right(y)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x); y = tf.image.flip_up_down(y)
    x = tf.clip_by_value(tf.image.random_brightness(x, 0.1), 0.0, 1.0)
    return x, y

def tf_dataset(X, Y, batch=BATCH, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(512)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def conv_block(x, filters, dropout=0.0):
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    if dropout > 0: x = Dropout(dropout)(x)
    return x

def encoder_block(x, filters, dropout=0.0):
    s = conv_block(x, filters, dropout)
    return s, MaxPool2D(2)(s)

def decoder_block(x, skip, filters):
    x = Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    return conv_block(x, filters)

def build_unet(input_shape=(H, W, 3)):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256, dropout=0.1)
    s4, p4 = encoder_block(p3, 512, dropout=0.1)
    b = conv_block(p4, 1024, dropout=0.2)
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    return Model(inputs, outputs, name="U-Net")

try:
    from scipy.spatial.distance import directed_hausdorff
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def hausdorff95(pred, gt):
    if not HAS_SCIPY: return 0.0
    pp, gp = np.argwhere(pred > 0), np.argwhere(gt > 0)
    if len(pp) == 0 or len(gp) == 0: return 0.0
    return max(directed_hausdorff(pp, gp)[0], directed_hausdorff(gp, pp)[0])

# ============================================================
# MAIN LOOP
# ============================================================
for fold_idx in range(N_FOLDS):
    fold_dir = os.path.join(BASE_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    model_path = os.path.join(fold_dir, "model.h5")
    csv_path   = os.path.join(fold_dir, "log.csv")
    done_flag  = os.path.join(fold_dir, "DONE.txt")
    score_path = os.path.join(fold_dir, "score_tumor_only.csv")

    print(f"\n{'='*60}\nFOLD {fold_idx+1}/{N_FOLDS}\n{'='*60}")

    if os.path.exists(done_flag) and os.path.exists(score_path):
        print(f"Fold {fold_idx} already complete. Skipping.")
        continue

    train_val_idx, test_idx = fold_assignments[fold_idx]
    train_val_images  = [all_images[i] for i in train_val_idx]
    train_val_masks   = [all_masks[i]  for i in train_val_idx]
    train_val_patients = [all_patients[i] for i in train_val_idx]
    test_images = [all_images[i] for i in test_idx]
    test_masks  = [all_masks[i]  for i in test_idx]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=SEED)
    tr_idx, val_idx = next(gss.split(train_val_images, groups=train_val_patients))
    train_images = [train_val_images[i] for i in tr_idx]
    train_masks  = [train_val_masks[i]  for i in tr_idx]
    val_images_all = [train_val_images[i] for i in val_idx]
    val_masks_all   = [train_val_masks[i]  for i in val_idx]

    # KEY FIX: filter validation to tumor-only slices for a clean,
    # representative training signal (not swamped by empty-mask noise)
    val_images, val_masks = [], []
    print("Filtering validation set to tumor-containing slices...")
    for vi, vm in zip(val_images_all, val_masks_all):
        if has_tumor(vm):
            val_images.append(vi); val_masks.append(vm)
    print(f"Validation: {len(val_images_all)} total slices -> {len(val_images)} tumor-containing slices used")

    print(f"Train: {len(train_images)} slices (all, incl. empty, for specificity), "
          f"Val: {len(val_images)} tumor slices, "
          f"Test: {len(test_images)} slices from {len(set(get_patient_id(f) for f in test_images))} patients")

    if not os.path.exists(done_flag):
        train_ds = tf_dataset(train_images, train_masks, augment=True)
        val_ds   = tf_dataset(val_images,   val_masks,   augment=False)

        if os.path.exists(model_path):
            print("Resuming fold from existing checkpoint...")
            with CustomObjectScope(CUSTOM_OBJS):
                model = tf.keras.models.load_model(model_path)
        else:
            print("Starting fresh training for this fold...")
            model = build_unet()

        model.compile(loss=bce_dice_loss, optimizer=Adam(LR), metrics=[dice_coef])

        initial_epoch = 0
        if os.path.exists(csv_path):
            try:
                prev = pd.read_csv(csv_path)
                initial_epoch = int(prev["epoch"].max()) + 1
                print(f"Resuming from epoch {initial_epoch}")
            except Exception:
                initial_epoch = 0

        callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True,
                            monitor="val_dice_coef", mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                              min_lr=1e-7, verbose=1),
            CSVLogger(csv_path, append=os.path.exists(csv_path)),
            EarlyStopping(monitor="val_dice_coef", mode="max", patience=PATIENCE,
                          restore_best_weights=True, verbose=1),
        ]

        model.fit(train_ds, epochs=EPOCHS, initial_epoch=initial_epoch,
                  validation_data=val_ds, callbacks=callbacks)

        with open(done_flag, "w") as f:
            f.write("done")

        with CustomObjectScope(CUSTOM_OBJS):
            model = tf.keras.models.load_model(model_path)
    else:
        print("Training already done for this fold, loading model for evaluation...")
        with CustomObjectScope(CUSTOM_OBJS):
            model = tf.keras.models.load_model(model_path)

    print(f"\nEvaluating fold {fold_idx} on TEST set (tumor-only slices)...")
    rows = []
    for x_path, y_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
        name    = os.path.basename(x_path)
        patient = get_patient_id(x_path)

        image = cv2.imread(x_path, cv2.IMREAD_COLOR)
        mask  = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        image = cv2.resize(image, (W, H))
        mask  = cv2.resize(mask, (W, H))
        mask_bin = (mask / 255.0 > 0.5).astype(np.int32)
        if mask_bin.sum() == 0:
            continue

        x_inp = np.expand_dims((image / 255.0).astype(np.float32), 0)
        y_pred = model.predict(x_inp, verbose=0)[0]
        y_pred_bin = (np.squeeze(y_pred) >= 0.5).astype(np.int32)

        fm, fp = mask_bin.flatten(), y_pred_bin.flatten()
        f1   = f1_score(fm, fp, average="binary", zero_division=0)
        jac  = jaccard_score(fm, fp, average="binary", zero_division=0)
        rec  = recall_score(fm, fp, average="binary", zero_division=0)
        prec = precision_score(fm, fp, average="binary", zero_division=0)
        hd95 = hausdorff95(y_pred_bin, mask_bin)
        f2   = (5*prec*rec)/(4*prec+rec+1e-15)

        rows.append([name, patient, fold_idx, f1, jac, rec, prec, f2, hd95])

    fold_df = pd.DataFrame(rows, columns=["Image","Patient","Fold","F1","Jaccard",
                                            "Recall","Precision","F2","HD95"])
    fold_df.to_csv(score_path, index=False)
    print(f"Fold {fold_idx} DONE. Tumor-slice Dice: "
          f"{fold_df['F1'].mean():.4f} \u00b1 {fold_df['F1'].std():.4f} (n={len(fold_df)})")

# ============================================================
# AGGREGATE
# ============================================================
all_fold_scores = []
completed = 0
for fold_idx in range(N_FOLDS):
    score_path = os.path.join(BASE_DIR, f"fold_{fold_idx}", "score_tumor_only.csv")
    if os.path.exists(score_path):
        all_fold_scores.append(pd.read_csv(score_path))
        completed += 1

print(f"\n{'='*60}\n{completed}/{N_FOLDS} folds completed so far.\n{'='*60}")

if completed == N_FOLDS:
    full_df = pd.concat(all_fold_scores, ignore_index=True)
    full_df.to_csv(os.path.join(BASE_DIR, "all_folds_combined.csv"), index=False)

    print("\n*** ALL FOLDS COMPLETE - FINAL AGGREGATED RESULTS ***\n")
    print(f"Total tumor-containing test slices: {len(full_df)}")
    print(f"Total unique patients evaluated: {full_df['Patient'].nunique()} / {len(unique_patients)}")

    print("\n--- Slice-level pooled statistics ---")
    for col in ["F1","Jaccard","Recall","Precision","F2","HD95"]:
        print(f"  {col}: {full_df[col].mean():.4f} \u00b1 {full_df[col].std():.4f}")

    print("\n--- Patient-level aggregation ---")
    patient_means = full_df.groupby('Patient')[["F1","Jaccard","Recall","Precision","F2","HD95"]].mean()
    for col in ["F1","Jaccard","Recall","Precision","F2","HD95"]:
        print(f"  {col}: {patient_means[col].mean():.4f} \u00b1 {patient_means[col].std():.4f}")

    print("\n--- Per-fold Dice ---")
    fold_means = full_df.groupby('Fold')['F1'].mean()
    for f, v in fold_means.items():
        print(f"  Fold {f}: {v:.4f}")
    print(f"  Across-fold mean \u00b1 SD: {fold_means.mean():.4f} \u00b1 {fold_means.std():.4f}")

    np.random.seed(42)
    vals = patient_means['F1'].values
    boot = [np.random.choice(vals, size=len(vals), replace=True).mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n95% Bootstrap CI (patient-level Dice): [{ci_lo:.4f}, {ci_hi:.4f}]")

    try:
        from google.colab import files
        files.download(os.path.join(BASE_DIR, "all_folds_combined.csv"))
    except Exception as e:
        print("File saved in Google Drive:", e)
else:
    print(f"\n{N_FOLDS - completed} fold(s) remaining. Re-run this same cell to continue.")
