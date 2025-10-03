#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGSEM Stroke Classification (CT Imaging) — Paper-Matched Reproduction
---------------------------------------------------------------------
Matches the manuscript settings:
- Preprocessing: Grayscale, wavelet denoising (DWT threshold), resize 512x512, normalize [0,1]
- Augmentation: H/V flip, rotations (±45°), zoom (1.5x), shifts (−30/+25 px), translation (~17, 6 px)
- Split: Stratified 65/15/20 on the *augmented* dataset (to mirror the paper)
- Base models: CNN, MLP, RF, DT, GNB, LR, XGB
- Ensemble: Majority vote across the 7 base learners (>=4 votes)
- Metrics: Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa, AUC, Log Loss, FNR, FPR, FDR
- XAI: Grad-CAM visualizations for the CNN component

Adjust ROOT_NORMAL_DIR / ROOT_STROKE_DIR to your data paths.
"""

import os, sys, math, glob, json, random
from typing import Tuple, List, Dict

import numpy as np
import cv2
import pywt
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_auc_score, classification_report, roc_curve, auc, log_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Set your dataset folders here
ROOT_NORMAL_DIR = "/content/drive/MyDrive/Brain Stroke/Normal_CT"
ROOT_STROKE_DIR = "/content/drive/MyDrive/Brain Stroke/Stroke_CT"

IMAGE_SIZE = 512

# Paper-reported augmented totals (Normal=10,857; Stroke=6,650; total≈17,507)
TARGET_NORMAL_COUNT = 10857
TARGET_STROKE_COUNT = 6650

SAVE_AUG_IMAGES = False
SAVE_AUG_DIR = "./augmented_export"

# Splits: 65/15/20 (on augmented dataset)
SPLIT_TRAIN = 0.65
SPLIT_VAL = 0.15
SPLIT_TEST = 0.20

# CNN training (Table-matched): 100 epochs, Adam 1e-4, 3 conv + 256 FC + Dropout 0.25
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 7

# -----------------------------
# Utilities
# -----------------------------
def find_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

def load_gray_resize(path: str, size: int=IMAGE_SIZE) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def wavelet_denoise(img_gray: np.ndarray, wavelet='db1', level=2, thresh_mode='soft') -> np.ndarray:
    coeffs = pywt.wavedec2(img_gray, wavelet=wavelet, level=level)
    cA, cD = coeffs[0], coeffs[1:]
    sigma = np.median(np.abs(cD[-1][0])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(img_gray.size))
    cD_thresh = []
    for (cH, cV, cD_) in cD:
        cH = pywt.threshold(cH, uthresh, thresh_mode)
        cV = pywt.threshold(cV, uthresh, thresh_mode)
        cD_ = pywt.threshold(cD_, uthresh, thresh_mode)
        cD_thresh.append((cH, cV, cD_))
    denoised = pywt.waverec2([cA] + cD_thresh, wavelet)
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    return denoised

def compute_psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
    if mse == 0: return 99.0
    PIXEL_MAX = 255.0
    return 10 * math.log10((PIXEL_MAX ** 2) / mse)

def normalize01(img_gray: np.ndarray) -> np.ndarray:
    return (img_gray / 255.0).astype(np.float32)

def zoom_1p5(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = int(h * 1.5), int(w * 1.5)
    z = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    y0 = (nh - h)//2; x0 = (nw - w)//2
    z = z[y0:y0+h, x0:x0+w]
    return z

def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def augment_one(img: np.ndarray) -> List[np.ndarray]:
    # Deterministic set to mirror the paper table
    aug = []
    aug.append(cv2.flip(img, 1))         # horizontal
    aug.append(cv2.flip(img, 0))         # vertical
    aug.append(rotate(img, 45))
    aug.append(rotate(img, -45))
    aug.append(zoom_1p5(img))
    aug.append(shift(img, -30, 0))       # left 30
    aug.append(shift(img, 25, 0))        # right 25
    aug.append(shift(img, 17, 6))        # ~ (16.5, 6.2)
    return aug

def prepare_dataset(normal_paths: List[str], stroke_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    print("Loading & preprocessing base images...")
    for p in tqdm(normal_paths, desc="Normal"):
        g = load_gray_resize(p)
        d = wavelet_denoise(g)
        X.append(normalize01(d)[..., np.newaxis]); y.append(0)
    for p in tqdm(stroke_paths, desc="Stroke"):
        g = load_gray_resize(p)
        d = wavelet_denoise(g)
        X.append(normalize01(d)[..., np.newaxis]); y.append(1)
    X = np.stack(X, axis=0); y = np.array(y, dtype=np.int32)
    return X, y

def expand_to_exact_counts(X: np.ndarray, y: np.ndarray,
                           target_norm: int, target_strk: int,
                           save_images: bool=False, save_root: str=SAVE_AUG_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Augment each class until we reach the exact paper totals per class."""
    os.makedirs(save_root, exist_ok=True) if save_images else None

    Xn = X[y == 0]; Ys = X[y == 1]
    n_norm, n_strk = len(Xn), len(Ys)

    def expand_class(Xc: np.ndarray, label: int, target_count: int, prefix: str):
        outX, outY = [], []
        idx = 0; cur = len(Xc)
        while cur < target_count:
            img = (Xc[idx % len(Xc)][..., 0] * 255.0).astype(np.uint8)
            for vi, v in enumerate(augment_one(img)):
                v = (v/255.0).astype(np.float32)[..., np.newaxis]
                outX.append(v); outY.append(label); cur += 1
                if save_images:
                    sub = os.path.join(save_root, f"{prefix}_lab{label}")
                    os.makedirs(sub, exist_ok=True)
                    cv2.imwrite(os.path.join(sub, f"{idx:06d}_{vi}.png"), (v[...,0]*255).astype(np.uint8))
                if cur >= target_count: break
            idx += 1
        return outX, outY

    print(f"Augmenting Normal from {n_norm} -> {target_norm} and Stroke from {n_strk} -> {target_strk}")
    addN_X, addN_y = ([], [])
    if n_norm < target_norm:
        addN_X, addN_y = expand_class(Xn, 0, target_norm, "normal")

    addS_X, addS_y = ([], [])
    if n_strk < target_strk:
        addS_X, addS_y = expand_class(Ys, 1, target_strk, "stroke")

    X_aug, y_aug = X, y
    if addN_X:
        X_aug = np.concatenate([X_aug, np.stack(addN_X)], axis=0)
        y_aug = np.concatenate([y_aug, np.array(addN_y, dtype=np.int32)], axis=0)
    if addS_X:
        X_aug = np.concatenate([X_aug, np.stack(addS_X)], axis=0)
        y_aug = np.concatenate([y_aug, np.array(addS_y, dtype=np.int32)], axis=0)

    return X_aug, y_aug

def build_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)) -> tf.keras.Model:
    model = models.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_all(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, label: str="") -> Dict[str, float]:
    # Primary metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    kap  = cohen_kappa_score(y_true, y_pred)

    # Extended metrics per paper
    try:
        aucv = roc_auc_score(y_true, y_prob)
    except Exception:
        aucv = float('nan')
    try:
        ll   = log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7))
    except Exception:
        ll   = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    fdr = fp / (tp + fp + 1e-12)

    out = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "mcc": mcc, "kappa": kap, "auc": aucv, "log_loss": ll,
        "fnr": fnr, "fpr": fpr, "fdr": fdr
    }
    print(f"\n== {label} ==")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", np.array([[tn, fp],[fn, tp]]))
    print({k: (round(v,4) if isinstance(v,float) else v) for k,v in out.items()})
    return out

def grad_cam(model: tf.keras.Model, img: np.ndarray, layer_name: str=None) -> np.ndarray:
    x = img[np.newaxis, ...]  # (1,H,W,1) in [0,1]
    if layer_name is None:
        layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                layer_name = layer.name; break
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0); heatmap /= (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    return heatmap

def main():
    # 1) Load file paths
    normal_files = find_images(ROOT_NORMAL_DIR)
    stroke_files = find_images(ROOT_STROKE_DIR)
    if len(normal_files) == 0 or len(stroke_files) == 0:
        print("ERROR: No images found. Check ROOT_NORMAL_DIR and ROOT_STROKE_DIR.")
        sys.exit(1)
    print(f"Found {len(normal_files)} normal and {len(stroke_files)} stroke images.")

    # 2) Preprocess (grayscale, denoise, normalize)
    X_base, y_base = prepare_dataset(normal_files, stroke_files)

    # Optional PSNR demo
    try:
        g0 = load_gray_resize(normal_files[0])
        d0 = wavelet_denoise(g0)
        print(f"Example PSNR (normal[0]): {compute_psnr(g0, d0):.2f} dB")
    except Exception as e:
        print("PSNR demo failed:", e)

    # 3) Augment to exact paper totals per class
    X_aug, y_aug = expand_to_exact_counts(
        X_base, y_base,
        target_norm=TARGET_NORMAL_COUNT,
        target_strk=TARGET_STROKE_COUNT,
        save_images=SAVE_AUG_IMAGES,
        save_root=SAVE_AUG_DIR
    )
    print(f"Augmented dataset shape: {X_aug.shape}, labels: {y_aug.shape}")
    idx = np.arange(len(X_aug)); np.random.shuffle(idx)
    X_aug, y_aug = X_aug[idx], y_aug[idx]

    # 4) Stratified 65/15/20 split 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_aug, y_aug, test_size=(1.0 - SPLIT_TRAIN), random_state=SEED, stratify=y_aug
    )
    rel = SPLIT_VAL / (SPLIT_VAL + SPLIT_TEST)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - rel), random_state=SEED, stratify=y_temp
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 5) CNN
    cnn = build_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    ckpt_path = "cnn_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    ]
    cnn.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
    y_prob_cnn = cnn.predict(X_test, verbose=0).ravel()
    y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
    metrics_cnn = evaluate_all(y_test, y_prob_cnn, y_pred_cnn, label="CNN")

    # 6) Flatten for classical models
    Npix = IMAGE_SIZE * IMAGE_SIZE
    Xtr_flat = X_train.reshape((len(X_train), Npix))
    Xva_flat = X_val.reshape((len(X_val), Npix))
    Xte_flat = X_test.reshape((len(X_test), Npix))

    # Standardize for LR/MLP (and can benefit XGB)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_std = scaler.fit_transform(Xtr_flat)
    Xva_std = scaler.transform(Xva_flat)
    Xte_std = scaler.transform(Xte_flat)

    # 7) Base learners
    # MLP (128,64,32), max_iter=50 to match "50 epochs" in paper
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                        solver='adam', learning_rate_init=1e-3,
                        max_iter=50, early_stopping=True, random_state=SEED, verbose=False)
    mlp.fit(Xtr_std, y_train)
    y_prob_mlp = mlp.predict_proba(Xte_std)[:,1]
    y_pred_mlp = (y_prob_mlp >= 0.5).astype(int)
    metrics_mlp = evaluate_all(y_test, y_prob_mlp, y_pred_mlp, label="MLP")

    # Random Forest (100 trees, gini)
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=SEED, n_jobs=-1)
    rf.fit(Xtr_flat, y_train)
    y_prob_rf = rf.predict_proba(Xte_flat)[:,1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    metrics_rf = evaluate_all(y_test, y_prob_rf, y_pred_rf, label="RF")

    # Decision Tree (max_depth=5, gini)
    dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=SEED)
    dt.fit(Xtr_flat, y_train)
    y_prob_dt = dt.predict_proba(Xte_flat)[:,1]
    y_pred_dt = (y_prob_dt >= 0.5).astype(int)
    metrics_dt = evaluate_all(y_test, y_prob_dt, y_pred_dt, label="DT")

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(Xtr_flat, y_train)
    y_prob_gnb = gnb.predict_proba(Xte_flat)[:,1]
    y_pred_gnb = (y_prob_gnb >= 0.5).astype(int)
    metrics_gnb = evaluate_all(y_test, y_prob_gnb, y_pred_gnb, label="GNB")

    # Logistic Regression (L2, C=1.0, liblinear)
    lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=SEED, max_iter=200)
    lr.fit(Xtr_std, y_train)
    y_prob_lr = lr.predict_proba(Xte_std)[:,1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    metrics_lr = evaluate_all(y_test, y_prob_lr, y_pred_lr, label="LR")

    # XGBoost (500 estimators, early stop=10)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        max_depth=6, subsample=0.9, colsample_bytree=0.9,
        tree_method='hist', random_state=SEED, n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_clf.fit(Xtr_flat, y_train, eval_set=[(Xva_flat, y_val)], verbose=False,
                early_stopping_rounds=10)
    y_prob_xgb = xgb_clf.predict_proba(Xte_flat)[:,1]
    y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)
    metrics_xgb = evaluate_all(y_test, y_prob_xgb, y_pred_xgb, label="XGB")

    # 8) Majority Vote Ensemble (7 models)
    preds = np.vstack([y_pred_cnn, y_pred_mlp, y_pred_rf, y_pred_dt, y_pred_gnb, y_pred_lr, y_pred_xgb]).T
    votes = np.sum(preds, axis=1)
    y_pred_ens = (votes >= 4).astype(int)  # majority of 7

    probs = np.vstack([y_prob_cnn, y_prob_mlp, y_prob_rf, y_prob_dt, y_prob_gnb, y_prob_lr, y_prob_xgb]).T
    y_prob_ens = np.mean(probs, axis=1)    # for AUC only

    metrics_ens = evaluate_all(y_test, y_prob_ens, y_pred_ens, label="NGSEM (Majority Vote)")

    # --- Voting contribution analysis (agreement with ensemble) ---
    names = ["CNN","MLP","RF","DT","GNB","LR","XGB"]
    base_preds = [y_pred_cnn, y_pred_mlp, y_pred_rf, y_pred_dt, y_pred_gnb, y_pred_lr, y_pred_xgb]
    agree_rates = {}
    for nm, bp in zip(names, base_preds):
        agree = np.mean(bp == y_pred_ens)
        agree_rates[nm] = float(agree)
    with open("voting_contribution.json", "w") as f:
        json.dump(agree_rates, f, indent=2)
    print("\nVoting contribution (agreement with ensemble):", {k: round(v,4) for k,v in agree_rates.items()})

    # 9) ROC Curve (ensemble)
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob_ens)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'NGSEM (area = {roc_auc:.3f})')
        plt.plot([0,1],[0,1],'--', lw=1, color='gray')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve — NGSEM'); plt.legend(loc='lower right'); plt.tight_layout()
        plt.savefig('roc_ngsem.png', dpi=200); plt.close()
    except Exception as e:
        print("ROC plotting failed:", e)

    # 10) Grad-CAM samples (CNN component)
    try:
        os.makedirs("gradcam_samples", exist_ok=True)
        sample_idxs = np.random.choice(len(X_test), size=min(6, len(X_test)), replace=False)
        for si in sample_idxs:
            img = X_test[si][...,0]
            heat = grad_cam(cnn, X_test[si])
            heatmap = (heat * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            base = (img*255).astype(np.uint8)
            base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(base_rgb, 0.5, heatmap_color, 0.5, 0)
            cv2.imwrite(f"gradcam_samples/sample_{si}_overlay.png", overlay)
    except Exception as e:
        print("Grad-CAM generation failed:", e)

    # 11) Save summary
    summary = {
        "cnn": metrics_cnn, "mlp": metrics_mlp, "rf": metrics_rf, "dt": metrics_dt,
        "gnb": metrics_gnb, "lr": metrics_lr, "xgb": metrics_xgb, "ensemble": metrics_ens,
        "voting_contribution": agree_rates
    }
    with open("metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: metrics_summary.json, voting_contribution.json, roc_ngsem.png, gradcam_samples/*.png, cnn_best.keras")

if __name__ == "__main__":
    main()
