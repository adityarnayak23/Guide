"""
NoteScope — Dual-Stream Onsets and Frames Swara Sequence Network
Trained directly on authentic studio recordings of Carnatic vocal masters (MTG UPF CompMusic Dataset).

Architecture:
- Stream 1: Acoustic Onset Sub-network (Dilated CNN + BiLSTM) predicting note attack probability P(onset | t).
- Stream 2: Swara Frame Sub-network (Dilated ResNet + BiLSTM) conditioned on acoustic features + onset stream,
            predicting 36 Indian Swarasthanas across Mandra, Madhya, and Taar Saptaks.
- Auxiliary Gamaka Head: Multi-task regularizer classifying micro-tonal dynamics (steady, kampitam, jaaru, karvai).
"""

import os
import glob
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Dual-Stream Onsets & Frames Training Device: {device}")

SWARAS_LIST = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
GAMAKAS_LIST = ["steady", "kampitam", "jaaru", "karvai"]

# ── Dataset Loader ─────────────────────────────────────────────────────────
class RealCarnaticDataset(Dataset):
    def __init__(self, data_dir="real_carnatic_dataset", max_frames=1000):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        print(f"Loaded {len(self.files)} real music segments from '{data_dir}'.")
        
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
        features = features[:, :T]
        onsets = onsets[:T]
        swaras = swaras[:T]
        voicing = voicing[:T]
        cents = cents[:T]
        
        # Determine gamaka labels from pitch contours
        gamakas = np.zeros(T, dtype=np.int64)
        for i in range(10, T - 10, 15):
            window_cents = cents[i:i + 15]
            window_v = voicing[i:i + 15]
            if np.mean(window_v > 0.4) > 0.6:
                std = np.std(window_cents)
                diff = window_cents[-1] - window_cents[0]
                if abs(diff) > 75.0:
                    gamakas[i:i + 15] = 2 # jaaru (glide)
                elif std > 38.0:
                    gamakas[i:i + 15] = 1 # kampitam (oscillation)
                elif std < 18.0:
                    gamakas[i:i + 15] = 3 # karvai (sustained hold)
                else:
                    gamakas[i:i + 15] = 0 # steady
                    
        # Pad to max_frames if shorter
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
            "features": torch.from_numpy(features).float(),  # (33, max_frames)
            "onsets": torch.from_numpy(onsets).float(),      # (max_frames,)
            "swaras": torch.from_numpy(swaras).long(),        # (max_frames,)
            "gamakas": torch.from_numpy(gamakas).long(),      # (max_frames,)
            "voicing": torch.from_numpy(voicing).float(),    # (max_frames,)
            "mask": torch.from_numpy(mask).float()           # (max_frames,)
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
        self.res = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.gelu(out + res)

class DualStreamOnsetsAndFramesSwaraNet(nn.Module):
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
        ) # 64 * 2 = 128 dims
        self.onset_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # ── Stream 2: Swara Frame Sub-network (Conditioned on Onset Stream) ─
        # Follows Google Magenta Onsets & Frames paradigm:
        # Frame network takes raw acoustic features + onset representations + onset probability
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
        ) # 128 * 2 = 256 dims
        
        # Swara Output Head (36 Swarasthanas)
        self.swara_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_swara_classes)
        )
        
        # Gamaka Output Head (Auxiliary Regularizer)
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

class HighCapacitySwaraNet(nn.Module):
    def __init__(self, in_features=33, num_swaras=36, num_gamakas=4):
        super().__init__()
        self.onset_conv = nn.Sequential(DilatedConvBlock(in_features, 64, 1), DilatedConvBlock(64, 128, 2))
        self.onset_lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.onset_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

        frame_in_dim = in_features + 256 + 1
        self.frame_conv = nn.Sequential(
            DilatedConvBlock(frame_in_dim, 128, 1),
            DilatedConvBlock(128, 256, 2),
            DilatedConvBlock(256, 256, 4)
        )
        self.frame_lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.swara_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256, num_swaras))
        self.gamaka_head = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Linear(64, num_gamakas))

    def forward(self, x):
        o_c = self.onset_conv(x)
        o_lstm, _ = self.onset_lstm(o_c.transpose(1, 2))
        onset_logits = self.onset_head(o_lstm).squeeze(-1)
        onset_probs = torch.sigmoid(onset_logits).unsqueeze(1)
        f_in = torch.cat([x, o_lstm.transpose(1, 2), onset_probs], dim=1)
        f_c = self.frame_conv(f_in)
        f_lstm, _ = self.frame_lstm(f_c.transpose(1, 2))
        return {
            'onset_logits': onset_logits,
            'onset_probs': onset_probs.squeeze(1),
            'swara_logits': self.swara_head(f_lstm),
            'gamaka_logits': self.gamaka_head(f_lstm)
        }

# ── Training Routine ───────────────────────────────────────────────────────
def train_model(data_dir="real_carnatic_dataset", epochs=30, batch_size=16, lr=1e-3, save_path="models/carnatic_swara_model.pt"):
    dataset = RealCarnaticDataset(data_dir=data_dir)
    n_samples = len(dataset)
    if n_samples == 0:
        print("No samples found in real_carnatic_dataset. Please verify directory.")
        return None
        
    n_val = max(4, int(n_samples * 0.15))
    n_train = n_samples - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    print(f"Training on {n_train} segments, validating on {n_val} segments (Batch size: {batch_size}).")
    
    model = DualStreamOnsetsAndFramesSwaraNet(in_features=33).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Positive weight for onset sparsity (~7% onset frames)
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
        train_onset_loss = 0.0
        train_swara_loss = 0.0
        correct_swaras = 0
        total_voiced = 0
        
        for batch in train_loader:
            feats = batch["features"].to(device) # (B, 33, T)
            onsets = batch["onsets"].to(device)   # (B, T)
            swaras = batch["swaras"].to(device)   # (B, T)
            gamakas = batch["gamakas"].to(device) # (B, T)
            voicing = batch["voicing"].to(device) # (B, T)
            mask = batch["mask"].to(device)       # (B, T)
            
            optimizer.zero_grad()
            preds = model(feats)
            
            # 1. Onset Loss (weighted BCE)
            l_onset = (onset_loss_fn(preds["onset_logits"], onsets) * mask).sum() / (mask.sum() + 1e-6)
            
            # 2. Swara Loss (computed on voiced frames only)
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
                
            # 3. Gamaka Loss (Auxiliary)
            g_logits_flat = preds["gamaka_logits"].reshape(-1, 4)
            gamakas_flat = gamakas.reshape(-1)
            if vmask_flat.sum() > 0:
                l_gamaka = (gamaka_loss_fn(g_logits_flat, gamakas_flat) * vmask_flat).sum() / vmask_flat.sum()
            else:
                l_gamaka = torch.tensor(0.0).to(device)
                
            # Combined Loss: Stream 1 (Onsets) + Stream 2 (Frames) + Auxiliary (Gamakas)
            total_loss = l_onset * 2.5 + l_swara * 3.0 + l_gamaka * 0.8
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_onset_loss += l_onset.item()
            train_swara_loss += l_swara.item()
            
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
                feats = batch["features"].to(device)
                onsets = batch["onsets"].to(device)
                swaras = batch["swaras"].to(device)
                voicing = batch["voicing"].to(device)
                mask = batch["mask"].to(device)
                
                preds = model(feats)
                
                voiced_mask = (voicing > 0.35) * mask
                logits_flat = preds["swara_logits"].reshape(-1, 36)
                swaras_flat = swaras.reshape(-1)
                vmask_flat = voiced_mask.reshape(-1)
                
                pred_classes = torch.argmax(logits_flat, dim=-1)
                val_correct += ((pred_classes == swaras_flat) * vmask_flat).sum().item()
                val_voiced += vmask_flat.sum().item()
                
                # Onset F1 evaluation
                pred_onsets = (preds["onset_probs"] > 0.40) * mask
                true_onsets = (onsets > 0.40) * mask
                
                val_tp += (pred_onsets * true_onsets).sum().item()
                val_fp += (pred_onsets * (1 - true_onsets)).sum().item()
                val_fn += ((1 - pred_onsets) * true_onsets).sum().item()
                
        val_acc = (val_correct / max(1, val_voiced)) * 100.0
        prec = val_tp / max(1, val_tp + val_fp)
        rec = val_tp / max(1, val_tp + val_fn)
        f1 = (2 * prec * rec / max(1e-6, prec + rec)) * 100.0
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Onset F1: {f1:.1f}%")
        
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
    print(f"Saved Dual-Stream Onsets & Frames Checkpoint to '{save_path}'.")
    print(f"============================================================")
    return model

if __name__ == "__main__":
    train_model(epochs=30)
