#!/usr/bin/env python3
"""
train_open_dataset_model.py
===========================
Trains a deep Indian Classical Music Svara & Swarasthana recognition engine
on authentic open-source datasets:
  - MTG UPF CompMusic Carnatic Varnam Dataset 1.1 (28 master recordings, 12,140 note instances)
  - Pre-trained MTG RISE InceptionTime 9.18M-parameter encoder (trained on Carnatic Music Rhythm)
  
Extracts multi-scale contextual embeddings (prec, curr, succ) + pitch trajectory features,
and trains a production-grade multi-head svara classifier.
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATASET_PKL = "/Users/adityanayak/mir_datasets/carnatic_dataset_processed.pkl"
CHECKPOINT_PTH = "/Users/adityanayak/mir_datasets/RISE/checkpoints/encoder.pth"
MODEL_SAVE_PATH = "/Users/adityanayak/Documents/GitHub/Guide/Guide/trained_svara_open_dataset.joblib"
INCEPTION_PY = "/Users/adityanayak/mir_datasets/RISE/src/svaras/model/inception.py"

def load_inception_encoder(device):
    import importlib.util
    spec = importlib.util.spec_from_file_location("inception", INCEPTION_PY)
    inc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inc)
    
    enc = inc.Encoder(embed_dim=48, depth=5, num_features=2).to(device)
    sd = torch.load(CHECKPOINT_PTH, map_location="cpu")
    enc.load_state_dict(sd)
    enc.eval()
    return enc

def prep_tensor(curve_list, L):
    B = len(curve_list)
    x = np.zeros((B, 2, L), dtype=np.float32)
    for i, c in enumerate(curve_list):
        n = min(len(c), L)
        vals = c[:n]
        valid_mask = ~np.isnan(vals)
        norm_p = (vals - (-4200.0)) / (4200.0 - (-4200.0))
        norm_p[~valid_mask] = 0.0
        x[i, 0, :n] = norm_p
        x[i, 1, :n] = (~valid_mask).astype(np.float32)
        if n < L:
            x[i, 1, n:] = 1.0
    return torch.from_numpy(x)

def extract_trajectory_features(s):
    curr = s['curr']
    prec = s['prec']
    succ = s['succ']
    valid_c = curr[~np.isnan(curr)]
    if len(valid_c) == 0:
        valid_c = np.array([0.0])
        
    p10, p25, p50, p75, p90 = np.percentile(valid_c, [10, 25, 50, 75, 90])
    mean_c = float(np.mean(valid_c))
    std_c = float(np.std(valid_c))
    dur = float(s['duration'])
    
    octave = np.floor(p50 / 1200.0)
    pitch_class = p50 % 1200.0
    
    n_pts = len(valid_c)
    if n_pts >= 3:
        t_norm = np.linspace(0, 1, n_pts)
        slope = float(np.polyfit(t_norm, valid_c, 1)[0])
        init_c = float(np.mean(valid_c[:max(1, int(n_pts * 0.25))]))
        mid_c = float(np.mean(valid_c[int(n_pts * 0.25):max(int(n_pts * 0.25) + 1, int(n_pts * 0.75))]))
        final_c = float(np.mean(valid_c[int(n_pts * 0.75):]))
    else:
        slope = 0.0
        init_c = mid_c = final_c = p50
        
    valid_p = prec[~np.isnan(prec)]
    prec_p50 = float(np.median(valid_p)) if len(valid_p) > 0 else p50
    prec_delta = p50 - prec_p50
    
    valid_s = succ[~np.isnan(succ)]
    succ_p50 = float(np.median(valid_s)) if len(valid_s) > 0 else p50
    succ_delta = succ_p50 - p50
    
    if n_pts >= 2:
        x_old = np.linspace(0, 1, n_pts)
        x_new = np.linspace(0, 1, 16)
        contour = np.interp(x_new, x_old, valid_c) - p50
    else:
        contour = np.zeros(16)
        
    return [
        p50, mean_c, std_c, p10, p25, p75, p90, p90 - p10,
        octave, pitch_class, dur, slope,
        init_c, mid_c, final_c,
        prec_p50, prec_delta, succ_p50, succ_delta
    ] + list(contour)

def main():
    print("=" * 60)
    print("Training Svara Engine on Open Source CompMusic & RISE Datasets")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware accelerator: {device}")
    
    with open(DATASET_PKL, "rb") as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} annotated note samples from 28 master recordings.")
    
    # 1. Load Pretrained Inception Encoder
    print("Loading pre-trained MTG RISE InceptionTime encoder (9.18M parameters)...")
    enc = load_inception_encoder(device)
    
    # 2. Extract Inception Embeddings in batches
    print("Extracting contextual embeddings (prec, curr, succ)...")
    t0 = time.time()
    batch_size = 256
    curr_embs, prec_embs, succ_embs = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            t_curr = prep_tensor([s['curr'] for s in batch], L=128).to(device)
            t_prec = prep_tensor([s['prec'] for s in batch], L=64).to(device)
            t_succ = prep_tensor([s['succ'] for s in batch], L=64).to(device)
            
            e_curr = torch.mean(enc(t_curr), dim=-1).cpu().numpy()
            e_prec = torch.mean(enc(t_prec), dim=-1).cpu().numpy()
            e_succ = torch.mean(enc(t_succ), dim=-1).cpu().numpy()
            
            curr_embs.append(e_curr)
            prec_embs.append(e_prec)
            succ_embs.append(e_succ)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(samples))}/{len(samples)} samples...")
                
    curr_embs = np.concatenate(curr_embs, axis=0)
    prec_embs = np.concatenate(prec_embs, axis=0)
    succ_embs = np.concatenate(succ_embs, axis=0)
    print(f"Embedding extraction complete in {time.time() - t0:.2f}s! Shape: {curr_embs.shape}")
    
    # 3. Extract Acoustic & Trajectory Features
    print("Extracting melodic trajectory features...")
    traj_feats = np.array([extract_trajectory_features(s) for s in samples], dtype=np.float32)
    print(f"Trajectory features shape: {traj_feats.shape}")
    
    # Combine full feature matrix: 48 (curr) + 48 (prec) + 48 (succ) + 35 (traj) = 179 features
    X = np.hstack([curr_embs, prec_embs, succ_embs, traj_feats])
    print(f"Combined Feature Matrix X: {X.shape}")
    
    # Labels
    base_labels = ['S', 'R', 'G', 'M', 'P', 'D', 'N']
    y_base = np.array([base_labels.index(s['svara'][0]) for s in samples], dtype=np.int64)
    
    # Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_base, test_size=0.2, random_state=42, stratify=y_base
    )
    print(f"Train samples: {X_train.shape[0]}, Held-out test samples: {X_test.shape[0]}")
    
    # 4. Train High-Performance Ensemble Classifier
    print("Training Gradient Boosted Decision Tree Ensemble with Inception context features...")
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=41,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )
    t_train = time.time()
    clf.fit(X_train, y_train)
    print(f"Training completed in {time.time() - t_train:.2f}s.")
    
    # 5. Evaluate on Held-Out Test Set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===========================================================")
    print(f"=== HELD-OUT TEST ACCURACY: {acc * 100:.2f}% ===")
    print(f"===========================================================\n")
    print(classification_report(y_test, y_pred, target_names=base_labels))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"{'Pred:':6s}" + "".join([f"{lbl:>6s}" for lbl in base_labels]))
    for i, row in enumerate(cm):
        print(f"{base_labels[i]:<6s}" + "".join([f"{val:6d}" for val in row]))
        
    # Save the trained model artifact
    save_payload = {
        "classifier": clf,
        "base_labels": base_labels,
        "feature_dim": X.shape[1],
        "test_accuracy": float(acc),
        "trained_on": "MTG UPF CompMusic Carnatic Varnam 1.1 + RISE Pretrained Encoder",
        "total_samples": len(samples),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(save_payload, MODEL_SAVE_PATH)
    print(f"\nTrained model successfully saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
