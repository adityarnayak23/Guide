"""
NoteScope — CarnaticSwaraNet Deep Neural Network Training Pipeline
Trains a multi-task Deep ResNet-BiLSTM model on actual Indian classical music
for high-precision note boundary detection, 36-class Swara identification, and Gamaka classification.
"""

import os
import glob
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Set device: Apple Silicon GPU (mps) or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

class CarnaticDataset(Dataset):
    def __init__(self, data_dir="dataset", sr=16000, hop_length=160, max_len_sec=12.0):
        self.sr = sr
        self.hop_length = hop_length
        self.max_frames = int(max_len_sec * 100) # 100 fps
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=80,
            f_min=60.0,
            f_max=4000.0
        )
        
        self.samples = []
        meta_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for item in meta:
                wav_path = os.path.join(data_dir, item["audio_file"])
                lbl_path = os.path.join(data_dir, item["label_file"])
                if os.path.exists(wav_path) and os.path.exists(lbl_path):
                    self.samples.append((wav_path, lbl_path, item["sa_hz"]))
                    
        print(f"Loaded {len(self.samples)} tracks into dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, lbl_path, sa_hz = self.samples[idx]
        
        audio, sr = sf.read(wav_path)
        if sr != self.sr:
            # Resample if needed
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            audio_t = torchaudio.transforms.Resample(sr, self.sr)(audio_t)
            audio = audio_t.squeeze(0).numpy()
            
        labels = np.load(lbl_path)
        onsets = labels["onsets"] # (T,)
        swaras = labels["swaras"] # (T,)
        gamakas = labels["gamakas"] # (T,)
        f0 = labels["f0"] if "f0" in labels else np.zeros_like(onsets)
        voicing = labels["voicing"] if "voicing" in labels else np.zeros_like(onsets)
        
        # Audio to Mel Spectrogram (1, 80, T)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        mel = self.mel_transform(audio_tensor).squeeze(0) # (80, T)
        mel = torch.log1p(10.0 * mel) # Log magnitude
        
        n_frames = min(mel.shape[1], len(onsets), len(f0), self.max_frames)
        mel = mel[:, :n_frames]
        onsets = onsets[:n_frames]
        swaras = swaras[:n_frames]
        gamakas = gamakas[:n_frames]
        f0 = f0[:n_frames]
        voicing = voicing[:n_frames]

        # Compute continuous cents relative to Sa
        cents = np.zeros(n_frames, dtype=np.float32)
        valid_f0 = (f0 > 40.0) & (voicing > 0.4)
        if np.any(valid_f0):
            cents[valid_f0] = 1200.0 * np.log2(f0[valid_f0] / float(sa_hz))
        cents_norm = cents / 1200.0
        cents_norm[~valid_f0] = -2.0
        cents_norm = np.clip(cents_norm, -2.5, 2.5)

        # Delta cents (pitch velocity)
        delta_cents = np.zeros(n_frames, dtype=np.float32)
        if n_frames > 1:
            delta_cents[1:] = np.diff(cents_norm)
        delta_cents = np.clip(delta_cents, -0.5, 0.5)
        
        # Pad to max_frames if shorter
        if n_frames < self.max_frames:
            pad_len = self.max_frames - n_frames
            mel = F.pad(mel, (0, pad_len))
            onsets = np.pad(onsets, (0, pad_len), constant_values=0)
            swaras = np.pad(swaras, (0, pad_len), constant_values=0)
            gamakas = np.pad(gamakas, (0, pad_len), constant_values=0)
            cents_norm = np.pad(cents_norm, (0, pad_len), constant_values=-2.0)
            delta_cents = np.pad(delta_cents, (0, pad_len), constant_values=0.0)
            voicing = np.pad(voicing, (0, pad_len), constant_values=0.0)
            mask = np.concatenate([np.ones(n_frames, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
        else:
            mask = np.ones(self.max_frames, dtype=np.float32)
            
        cents_t = torch.from_numpy(cents_norm).float().unsqueeze(0) # (1, max_frames)
        delta_t = torch.from_numpy(delta_cents).float().unsqueeze(0) # (1, max_frames)
        voicing_t = torch.from_numpy(voicing).float().unsqueeze(0) # (1, max_frames)
        
        features = torch.cat([mel, cents_t, delta_t, voicing_t], dim=0) # (83, max_frames)
        
        return {
            "features": features, # (83, max_frames)
            "onsets": torch.from_numpy(onsets).float(), # (max_frames,)
            "swaras": torch.from_numpy(swaras).long(),   # (max_frames,)
            "gamakas": torch.from_numpy(gamakas).long(), # (max_frames,)
            "mask": torch.from_numpy(mask).float(),      # (max_frames,)
            "sa_hz": float(sa_hz)
        }

# ── Neural Network Architecture ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
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

class CarnaticSwaraNet(nn.Module):
    def __init__(self, in_features=83, num_swara_classes=36, num_gamaka_classes=4):
        super().__init__()
        
        # 1. Convolutional Feature Extractor
        self.enc1 = ConvBlock(in_features, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 128)
        
        # 2. Bidirectional Temporal Context (BiLSTM)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 3. Output Heads (BiLSTM output = 256 features per frame)
        # Head 1: Note Onset / Boundary detector
        self.onset_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Head 2: 36 Indian Swarasthanas (12 Swaras x 3 Saptaks)
        self.swara_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_swara_classes)
        )
        
        # Head 3: Gamaka Style Classifier (Steady, Kampitam, Jaaru, Karvai)
        self.gamaka_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_gamaka_classes)
        )

    def forward(self, x):
        # x: (Batch, 81, T)
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2) # (Batch, 128, T)
        
        # Transpose for LSTM: (Batch, T, 128)
        lstm_in = c3.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in) # (Batch, T, 256)
        
        # Head predictions
        onset_logits = self.onset_head(lstm_out).squeeze(-1) # (Batch, T)
        swara_logits = self.swara_head(lstm_out)              # (Batch, T, 36)
        gamaka_logits = self.gamaka_head(lstm_out)            # (Batch, T, 4)
        
        return {
            "onset_logits": onset_logits,
            "swara_logits": swara_logits,
            "gamaka_logits": gamaka_logits
        }

# ── Training Loop ────────────────────────────────────────────────────────────
def train_model(data_dir="dataset", output_model_path="models/carnatic_swara_model.pt", epochs=22, batch_size=8, lr=1e-3):
    dataset = CarnaticDataset(data_dir=data_dir)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = CarnaticSwaraNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Weighted BCE for sparse onsets
    onset_pos_weight = torch.tensor([8.0]).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=onset_pos_weight, reduction='none')
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    best_val_loss = float('inf')
    best_acc = 0.0
    
    print("\nStarting CarnaticSwaraNet Deep Neural Network Training...")
    print("=" * 65)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_swara_correct = 0
        train_swara_total = 0
        
        for batch in train_loader:
            features = batch["features"].to(device)
            target_onsets = batch["onsets"].to(device)
            target_swaras = batch["swaras"].to(device)
            target_gamakas = batch["gamakas"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            preds = model(features)
            
            # 1. Onset Loss (masked)
            loss_onset = (bce_loss_fn(preds["onset_logits"], target_onsets) * mask).sum() / mask.sum()
            
            # 2. Swara Classification Loss (masked)
            loss_swara = (ce_loss_fn(preds["swara_logits"].transpose(1, 2), target_swaras) * mask).sum() / mask.sum()
            
            # 3. Gamaka Classification Loss (masked)
            loss_gamaka = (ce_loss_fn(preds["gamaka_logits"].transpose(1, 2), target_gamakas) * mask).sum() / mask.sum()
            
            total_loss = 1.8 * loss_onset + 1.2 * loss_swara + 0.8 * loss_gamaka
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Accuracy on valid frames
            swara_pred = preds["swara_logits"].argmax(dim=-1)
            valid = mask.bool()
            train_swara_correct += ((swara_pred == target_swaras) & valid).sum().item()
            train_swara_total += valid.sum().item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        train_acc = (train_swara_correct / max(1, train_swara_total)) * 100.0
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_swara_correct = 0
        val_swara_total = 0
        val_onset_tp, val_onset_fp, val_onset_fn = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                target_onsets = batch["onsets"].to(device)
                target_swaras = batch["swaras"].to(device)
                target_gamakas = batch["gamakas"].to(device)
                mask = batch["mask"].to(device)
                
                preds = model(features)
                
                loss_onset = (bce_loss_fn(preds["onset_logits"], target_onsets) * mask).sum() / mask.sum()
                loss_swara = (ce_loss_fn(preds["swara_logits"].transpose(1, 2), target_swaras) * mask).sum() / mask.sum()
                loss_gamaka = (ce_loss_fn(preds["gamaka_logits"].transpose(1, 2), target_gamakas) * mask).sum() / mask.sum()
                
                total_loss = 1.8 * loss_onset + 1.2 * loss_swara + 0.8 * loss_gamaka
                val_loss += total_loss.item()
                
                swara_pred = preds["swara_logits"].argmax(dim=-1)
                valid = mask.bool()
                val_swara_correct += ((swara_pred == target_swaras) & valid).sum().item()
                val_swara_total += valid.sum().item()
                
                # Onset F1
                onset_probs = torch.sigmoid(preds["onset_logits"]) > 0.45
                onset_gt = target_onsets > 0.5
                val_onset_tp += ((onset_probs & onset_gt) & valid).sum().item()
                val_onset_fp += ((onset_probs & (~onset_gt)) & valid).sum().item()
                val_onset_fn += (((~onset_probs) & onset_gt) & valid).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = (val_swara_correct / max(1, val_swara_total)) * 100.0
        onset_prec = (val_onset_tp / max(1, val_onset_tp + val_onset_fp)) * 100.0
        onset_rec = (val_onset_tp / max(1, val_onset_tp + val_onset_fn)) * 100.0
        onset_f1 = (2 * onset_prec * onset_rec / max(1e-4, onset_prec + onset_rec))
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Swara Acc: {val_acc:.1f}% | Onset F1: {onset_f1:.1f}%")
        
        # Save best model checkpoint
        if val_acc > best_acc or (val_acc >= best_acc - 0.5 and val_loss < best_val_loss):
            best_acc = val_acc
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "onset_f1": onset_f1,
                "val_loss": val_loss
            }, output_model_path)
            print(f"  ★ New best model checkpoint saved to '{output_model_path}' (Swara Acc: {val_acc:.1f}%, Onset F1: {onset_f1:.1f}%)")
            
    print("=" * 65)
    print(f"Training Complete! Final Best Swara Accuracy: {best_acc:.1f}%")
    return output_model_path

if __name__ == "__main__":
    train_model(data_dir="dataset", output_model_path="models/carnatic_swara_model.pt", epochs=22)
