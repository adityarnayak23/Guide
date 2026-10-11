"""
NoteScope — Super Industrial Indian Classical Vocal Swara Network (Cloud GPU Edition)
========================================================================================
Trained on both Carnatic and Hindustani Music libraries from MTG UPF CompMusic & Saraga.
Designed for NVIDIA A100 / V100 / T4 GPUs with PyTorch Mixed Precision (AMP).

Datasets:
  1. MTG CompMusic Carnatic Varnam Dataset (28 master recordings, 12,140 annotations)
  2. Saraga Carnatic Dataset (40 hours of master performances, multi-track vocal stems)
  3. Saraga Hindustani Dataset (26 hours of Khayal & Dhrupad master performances)

Architecture:
  - Stream 1: Acoustic Note Onset Sub-network (Dilated CNN + BiLSTM)
  - Stream 2: Cross-Conditioned Swara Frame Sub-network (Dilated ResNet + BiLSTM)
  - Stream 3: Micro-tonal Dynamics & Gamaka/Meend Ornamentation Head
  - Multi-Scale Gaussian Swarasthana Receptive Fields
"""

import os
import sys
import glob
import math
import json
import argparse
import random
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Cloud GPU Acceleration Enabled: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Local Apple Silicon GPU Enabled (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU (Warning: Cloud GPU recommended for multi-dataset training)")

# 36 Indian Swarasthanas across Mandra, Madhya, and Taar Saptaks
SWARAS_LIST = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
HINDUSTANI_SWARAS = ["S", "r", "R", "g", "G", "m", "M'", "P", "d", "D", "n", "N"]
SWARA_TO_SEMITONE = {
    # Carnatic notation
    "S": 0, "R1": 1, "R2": 2, "G2": 3, "G3": 4, "M1": 5, "M2": 6,
    "P": 7, "D1": 8, "D2": 9, "N2": 10, "N3": 11,
    # Hindustani notation
    "r": 1, "R": 2, "g": 3, "G": 4, "m": 5, "M'": 6,
    "d": 8, "D": 9, "n": 10, "N": 11
}

# ── Neural Network Architecture ─────────────────────────────────────────────
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.gelu(out + res)

class SuperDualStreamSwaraNet(nn.Module):
    def __init__(self, in_features=33, num_swara_classes=36, num_gamaka_classes=4):
        super().__init__()
        # ── Stream 1: Acoustic Onset Sub-network ─────────────────────────────
        # Dedicated to high-frequency transients, attack bursts, and syllable onsets
        self.onset_conv = nn.Sequential(
            DilatedConvBlock(in_features, 64, dilation=1),
            DilatedConvBlock(64, 64, dilation=2)
        )
        self.onset_lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        ) # 128 dims
        self.onset_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # ── Stream 2: Swara Frame Sub-network (Cross-Conditioned) ────────────
        # Receptive field is wider (dilation 1, 2, 4) to capture micro-intonation,
        # kampitam/andolan vibrato cycles, and jaaru/meend glides
        frame_in_dim = in_features + 128 + 1 # 33 + 128 + 1 = 162
        self.frame_conv = nn.Sequential(
            DilatedConvBlock(frame_in_dim, 128, dilation=1),
            DilatedConvBlock(128, 128, dilation=2),
            DilatedConvBlock(128, 128, dilation=4)
        )
        self.frame_lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        ) # 256 dims
        
        # Swara Output Head (36 Swarasthanas across Mandra, Madhya, Taar)
        self.swara_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_swara_classes)
        )
        
        # Gamaka / Meend Ornamentation Head
        self.gamaka_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_gamaka_classes)
        )

    def forward(self, x):
        # x: (Batch, in_features, T)
        # 1. Stream 1: Predict Onsets
        o_c = self.onset_conv(x) # (B, 64, T)
        o_lstm_out, _ = self.onset_lstm(o_c.transpose(1, 2)) # (B, T, 128)
        onset_logits = self.onset_head(o_lstm_out).squeeze(-1) # (B, T)
        onset_probs = torch.sigmoid(onset_logits).unsqueeze(1) # (B, 1, T)
        
        # 2. Stream 2: Condition Frame Network on Onset Stream
        o_feats_t = o_lstm_out.transpose(1, 2) # (B, 128, T)
        frame_input = torch.cat([x, o_feats_t, onset_probs], dim=1) # (B, 162, T)
        
        f_c = self.frame_conv(frame_input) # (B, 128, T)
        f_lstm_out, _ = self.frame_lstm(f_c.transpose(1, 2)) # (B, T, 256)
        
        swara_logits = self.swara_head(f_lstm_out)   # (B, T, 36)
        gamaka_logits = self.gamaka_head(f_lstm_out) # (B, T, 4)
        
        return {
            "onset_logits": onset_logits,
            "onset_probs": onset_probs.squeeze(1),
            "swara_logits": swara_logits,
            "gamaka_logits": gamaka_logits
        }

# ── Feature Extractor ───────────────────────────────────────────────────────
def compute_33_features(y, sr, sa_hz, hop_length=160, device_for_crepe="cuda"):
    """
    Computes standard 33-channel Indian Classical feature matrix:
    - Continuous normalized cents relative to Sa
    - 1st and 2nd derivatives of cents (gliding velocity & curvature)
    - Periodicity / Voicing confidence
    - Acoustic onset strength envelope
    - RMS energy
    - 12 fine-resolution Gaussian swara filters (sigma=25 cents)
    - 12 coarse-resolution Gaussian swara filters (sigma=65 cents)
    - 3 saptak activations (Mandra, Madhya, Taar)
    """
    import torchcrepe
    audio_t = torch.from_numpy(y).float().unsqueeze(0).to(device_for_crepe)
    try:
        pitch, periodicity = torchcrepe.predict(
            audio_t, sr, hop_length=hop_length, fmin=65, fmax=950,
            model="tiny", batch_size=2048, device=device_for_crepe, return_periodicity=True
        )
        f0 = pitch.squeeze().cpu().numpy()
        voicing = periodicity.squeeze().cpu().numpy()
    except Exception:
        f0_raw, _, v_probs = librosa.pyin(y, fmin=65, fmax=950, sr=sr, hop_length=hop_length)
        f0 = np.nan_to_num(f0_raw)
        voicing = np.nan_to_num(v_probs)

    T = len(f0)
    valid = (voicing > 0.40) & (f0 > 40.0)
    cents = np.zeros(T, dtype=np.float32)
    if np.any(valid) and sa_hz > 0:
        cents[valid] = 1200.0 * np.log2(f0[valid] / sa_hz)
    cents_norm = np.clip(np.where(valid, cents / 1200.0, -2.0), -2.5, 2.5)

    delta_cents = np.zeros(T, dtype=np.float32)
    delta_cents[1:] = np.clip(np.diff(cents_norm), -0.5, 0.5)

    delta2_cents = np.zeros(T, dtype=np.float32)
    delta2_cents[1:] = np.clip(np.diff(delta_cents), -0.5, 0.5)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)[:T]
    if len(onset_env) < T: onset_env = np.pad(onset_env, (0, T - len(onset_env)))
    if np.max(onset_env) > 0: onset_env = onset_env / np.max(onset_env)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0][:T]
    if len(rms) < T: rms = np.pad(rms, (0, T - len(rms)))
    if np.max(rms) > 0: rms = rms / np.max(rms)

    cents_mod1200 = cents % 1200.0
    g_fine = np.zeros((12, T), dtype=np.float32)
    g_coarse = np.zeros((12, T), dtype=np.float32)
    for k in range(12):
        target = k * 100.0
        dist = np.abs(cents_mod1200 - target)
        dist = np.minimum(dist, 1200.0 - dist)
        g_fine[k] = np.exp(-(dist ** 2) / (2.0 * (25.0 ** 2))) * valid
        g_coarse[k] = np.exp(-(dist ** 2) / (2.0 * (65.0 ** 2))) * valid

    saptak_act = np.zeros((3, T), dtype=np.float32)
    saptak_act[0] = (cents < -50.0) & valid
    saptak_act[1] = (cents >= -50.0) & (cents < 1150.0) & valid
    saptak_act[2] = (cents >= 1150.0) & valid

    features = np.vstack([
        cents_norm[np.newaxis, :], delta_cents[np.newaxis, :], delta2_cents[np.newaxis, :],
        voicing[np.newaxis, :], onset_env[np.newaxis, :], rms[np.newaxis, :],
        g_fine, g_coarse, saptak_act
    ]).astype(np.float32)

    return features, f0, voicing, cents

# ── Training Dataset ────────────────────────────────────────────────────────
class CloudCarnaticHindustaniDataset(Dataset):
    def __init__(self, data_dirs=["real_carnatic_dataset"], max_frames=1000, augment=True):
        self.files = []
        for d in data_dirs:
            if os.path.exists(d):
                self.files.extend(sorted(glob.glob(os.path.join(d, "*.npz"))))
        self.max_frames = max_frames
        self.augment = augment
        print(f"Loaded {len(self.files)} segments across directories: {data_dirs}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        features = data["features"] # (33, T)
        onsets = data["onsets"]     # (T,)
        swaras = data["swaras"]     # (T,)
        voicing = data["voicing"]   # (T,)
        cents = data["cents"]       # (T,)

        T = min(features.shape[1], len(onsets), len(swaras), self.max_frames)
        features = features[:, :T].copy()
        onsets = onsets[:T].copy()
        swaras = swaras[:T].copy()
        voicing = voicing[:T].copy()
        cents = cents[:T].copy()

        # Gamaka labels from pitch contours
        gamakas = np.zeros(T, dtype=np.int64)
        for i in range(10, T - 10, 15):
            window_cents = cents[i:i + 15]
            window_v = voicing[i:i + 15]
            if np.mean(window_v > 0.4) > 0.6:
                std = np.std(window_cents)
                diff = window_cents[-1] - window_cents[0]
                if abs(diff) > 75.0:
                    gamakas[i:i + 15] = 2 # jaaru / meend
                elif std > 38.0:
                    gamakas[i:i + 15] = 1 # kampitam / andolan
                elif std < 18.0:
                    gamakas[i:i + 15] = 3 # karvai / sustained
                else:
                    gamakas[i:i + 15] = 0 # steady

        # Cloud Augmentation: Pitch shifting across tonics
        if self.augment and random.random() < 0.40:
            # Shift by whole semitones (-3 to +3 semitones)
            shift_semi = random.randint(-3, 3)
            if shift_semi != 0:
                shift_cents = shift_semi * 100.0
                valid_mask = (voicing > 0.40) & (cents > -3000.0)
                cents[valid_mask] += shift_cents
                
                # Recompute shifted features consistently:
                cents_norm = np.clip(np.where(valid_mask, cents / 1200.0, -2.0), -2.5, 2.5)
                features[0] = cents_norm
                features[1, 1:] = np.clip(np.diff(cents_norm), -0.5, 0.5)
                features[2, 1:] = np.clip(np.diff(features[1]), -0.5, 0.5)
                
                cents_mod1200 = cents % 1200.0
                for k in range(12):
                    target = k * 100.0
                    dist = np.abs(cents_mod1200 - target)
                    dist = np.minimum(dist, 1200.0 - dist)
                    features[6 + k] = np.exp(-(dist ** 2) / (2.0 * (25.0 ** 2))) * valid_mask
                    features[18 + k] = np.exp(-(dist ** 2) / (2.0 * (65.0 ** 2))) * valid_mask
                
                features[30] = (cents < -50.0) & valid_mask
                features[31] = (cents >= -50.0) & (cents < 1150.0) & valid_mask
                features[32] = (cents >= 1150.0) & valid_mask
                
                # Shift swara classes if within bounds [0..35]
                swaras[valid_mask] = np.clip(swaras[valid_mask] + shift_semi, 0, 35)

        # Pad to max_frames
        if T < self.max_frames:
            pad_len = self.max_frames - T
            features = np.pad(features, ((0, 0), (0, pad_len)))
            onsets = np.pad(onsets, (0, pad_len))
            swaras = np.pad(swaras, (0, pad_len))
            gamakas = np.pad(gamakas, (0, pad_len))
            voicing = np.pad(voicing, (0, pad_len))
            mask = np.concatenate([np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
        else:
            mask = np.ones(self.max_frames, dtype=np.float32)

        return {
            "features": torch.from_numpy(features).float(),
            "onsets": torch.from_numpy(onsets).float(),
            "swaras": torch.from_numpy(swaras).long(),
            "gamakas": torch.from_numpy(gamakas).long(),
            "voicing": torch.from_numpy(voicing).float(),
            "mask": torch.from_numpy(mask).float()
        }

# ── Cloud Training Routine ─────────────────────────────────────────────────
def run_cloud_training(
    data_dirs=["real_carnatic_dataset"],
    epochs=50,
    batch_size=32,
    lr=1e-3,
    save_path="models/carnatic_hindustani_super_model.pt"
):
    dataset = CloudCarnaticHindustaniDataset(data_dirs=data_dirs, augment=True)
    n_samples = len(dataset)
    if n_samples == 0:
        print("Error: No data segments found. Run dataset preparation first.")
        return

    n_val = max(8, int(n_samples * 0.15))
    n_train = n_samples - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n============================================================")
    print(f"Starting Super Industrial Dual-Stream Swara Training (Cloud GPU)")
    print(f"Device: {device} | Train: {n_train} segments | Val: {n_val} segments | Batch Size: {batch_size}")
    print(f"============================================================\n")

    model = SuperDualStreamSwaraNet(in_features=33).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    pos_weight = torch.tensor([8.0]).to(device)
    onset_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    swara_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08, reduction="none")
    gamaka_loss_fn = nn.CrossEntropyLoss(reduction="none")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_acc = 0.0
    best_onset_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_swaras = 0
        total_voiced = 0

        for batch in train_loader:
            feats = batch["features"].to(device, non_blocking=True)
            onsets = batch["onsets"].to(device, non_blocking=True)
            swaras = batch["swaras"].to(device, non_blocking=True)
            gamakas = batch["gamakas"].to(device, non_blocking=True)
            voicing = batch["voicing"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(feats)
                l_onset = (onset_loss_fn(preds["onset_logits"], onsets) * mask).sum() / (mask.sum() + 1e-6)

                voiced_mask = (voicing > 0.35) * mask
                B, T, C = preds["swara_logits"].shape
                logits_flat = preds["swara_logits"].reshape(-1, C)
                swaras_flat = swaras.reshape(-1)
                vmask_flat = voiced_mask.reshape(-1)

                if vmask_flat.sum() > 0:
                    l_swara = (swara_loss_fn(logits_flat, swaras_flat) * vmask_flat).sum() / vmask_flat.sum()
                    pred_classes = torch.argmax(logits_flat, dim=-1)
                    correct_swaras += ((pred_classes == swaras_flat) * vmask_flat).sum().item()
                    total_voiced += vmask_flat.sum().item()
                else:
                    l_swara = torch.tensor(0.0).to(device)

                g_logits_flat = preds["gamaka_logits"].reshape(-1, 4)
                gamakas_flat = gamakas.reshape(-1)
                if vmask_flat.sum() > 0:
                    l_gamaka = (gamaka_loss_fn(g_logits_flat, gamakas_flat) * vmask_flat).sum() / vmask_flat.sum()
                else:
                    l_gamaka = torch.tensor(0.0).to(device)

                total_loss = l_onset * 2.5 + l_swara * 3.0 + l_gamaka * 0.8

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

        scheduler.step()
        train_acc = (correct_swaras / max(1, total_voiced)) * 100.0

        # Validation
        model.eval()
        val_correct = 0
        val_voiced = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0

        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device, non_blocking=True)
                onsets = batch["onsets"].to(device, non_blocking=True)
                swaras = batch["swaras"].to(device, non_blocking=True)
                voicing = batch["voicing"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                preds = model(feats)
                voiced_mask = (voicing > 0.35) * mask
                logits_flat = preds["swara_logits"].reshape(-1, 36)
                swaras_flat = swaras.reshape(-1)
                vmask_flat = voiced_mask.reshape(-1)

                pred_classes = torch.argmax(logits_flat, dim=-1)
                val_correct += ((pred_classes == swaras_flat) * vmask_flat).sum().item()
                val_voiced += vmask_flat.sum().item()

                pred_onsets = (preds["onset_probs"] > 0.40) * mask
                true_onsets = (onsets > 0.40) * mask
                val_tp += (pred_onsets * true_onsets).sum().item()
                val_fp += (pred_onsets * (1 - true_onsets)).sum().item()
                val_fn += ((1 - pred_onsets) * true_onsets).sum().item()

        val_acc = (val_correct / max(1, val_voiced)) * 100.0
        prec = val_tp / max(1, val_tp + val_fp)
        rec = val_tp / max(1, val_tp + val_fn)
        f1 = (2 * prec * rec / max(1e-6, prec + rec)) * 100.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Onset F1: {f1:.1f}%")

        if val_acc > best_val_acc or epoch == epochs:
            best_val_acc = max(best_val_acc, val_acc)
            best_onset_f1 = max(best_onset_f1, f1)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "onset_f1": f1,
                "in_features": 33
            }, save_path)

    print(f"\n============================================================")
    print(f"Training Complete! Best Val Swara Accuracy: {best_val_acc:.1f}% | Onset F1: {best_onset_f1:.1f}%")
    print(f"Super-Model Checkpoint Saved to: {save_path}")
    print(f"============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud GPU Training for Indian Classical Swara Network")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (use 32 or 64 on A100/T4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_dirs", nargs="+", default=["real_carnatic_dataset"], help="Directories containing .npz dataset chunks")
    parser.add_argument("--save_path", type=str, default="models/carnatic_hindustani_super_model.pt", help="Path to save checkpoint")
    args = parser.parse_args()

    run_cloud_training(
        data_dirs=args.data_dirs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
