"""
train_svara_conformer.py
========================
High-Performance In-Memory Trainer for Svara-Conformer Foundation Model (Option C).
Trains directly on Tonic-Centered 144-bin logarithmic filterbank representations.
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from svara_conformer import SvaraConformer, svara_tokenizer
from synthetic_vocal_factory import (
    synthesize_vocal_phrase,
    collate_svara_batch
)

CHECKPOINT_DIR = "/Users/adityanayak/Documents/GitHub/Guide/Guide/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "svara_conformer_model.pt")

def train_epoch(model, dataloader, optimizer, ctc_loss_fn, bce_loss_fn, ce_loss_fn, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    for batch in dataloader:
        spec = batch["tonic_spec"].to(device)
        spec_lens = batch["spec_lens"].to(device)
        frame_svaras = batch["frame_svaras"].to(device)
        targets = batch["targets"].to(device)
        target_lens = batch["target_lens"].to(device)
        onsets = batch["onsets"].to(device)
        gamakas = batch["gamakas"].to(device)
        
        optimizer.zero_grad()
        out = model(spec, spec_lens)
        
        # 1. Frame-Level Svara Cross-Entropy Loss
        min_sub = min(out["svara_logits"].shape[1], frame_svaras.shape[1])
        svara_logits_aligned = out["svara_logits"][:, :min_sub, :].reshape(-1, model.num_classes)
        frame_svaras_aligned = frame_svaras[:, :min_sub].reshape(-1)
        loss_frame = ce_loss_fn(svara_logits_aligned, frame_svaras_aligned)
        
        # Frame accuracy
        preds = torch.argmax(svara_logits_aligned, dim=-1)
        acc = (preds == frame_svaras_aligned).float().mean().item()
        
        # 2. Sequence CTC Loss
        log_probs = torch.log_softmax(out["svara_logits"], dim=-1).permute(1, 0, 2)
        loss_ctc = ctc_loss_fn(log_probs, targets, out["lengths"], target_lens)
        
        # 3. Acoustic Onset Activation Loss
        onset_logits = out["onset_logits"][:, :min_sub]
        onsets_aligned = onsets[:, :min_sub]
        loss_onset = bce_loss_fn(onset_logits, onsets_aligned)
        
        # 4. Gamaka Classification Loss
        gamaka_logits = out["gamaka_logits"][:, :min_sub, :].reshape(-1, 6)
        gamakas_aligned = gamakas[:, :min_sub].reshape(-1)
        loss_gamaka = ce_loss_fn(gamaka_logits, gamakas_aligned)
        
        loss = loss_frame + 0.3 * loss_ctc + 0.3 * loss_onset + 0.2 * loss_gamaka
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
            
    avg_loss = total_loss / max(1, n_batches)
    avg_acc = total_acc / max(1, n_batches)
    return avg_loss, avg_acc

def evaluate(model, test_loader, ctc_loss_fn, ce_loss_fn, device):
    model.eval()
    total_frame_acc = 0.0
    total_gamaka_acc = 0.0
    total_ctc = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            spec = batch["tonic_spec"].to(device)
            spec_lens = batch["spec_lens"].to(device)
            frame_svaras = batch["frame_svaras"].to(device)
            targets = batch["targets"].to(device)
            target_lens = batch["target_lens"].to(device)
            gamakas = batch["gamakas"].to(device)
            
            out = model(spec, spec_lens)
            min_sub = min(out["svara_logits"].shape[1], frame_svaras.shape[1])
            svara_logits_aligned = out["svara_logits"][:, :min_sub, :].reshape(-1, model.num_classes)
            frame_svaras_aligned = frame_svaras[:, :min_sub].reshape(-1)
            
            preds = torch.argmax(svara_logits_aligned, dim=-1)
            frame_acc = (preds == frame_svaras_aligned).float().mean().item()
            
            log_probs = torch.log_softmax(out["svara_logits"], dim=-1).permute(1, 0, 2)
            ctc_loss = ctc_loss_fn(log_probs, targets, out["lengths"], target_lens)
            
            gamaka_logits = out["gamaka_logits"][:, :min_sub, :].reshape(-1, 6)
            gamakas_aligned = gamakas[:, :min_sub].reshape(-1)
            gamaka_preds = torch.argmax(gamaka_logits, dim=-1)
            gamaka_acc = (gamaka_preds == gamakas_aligned).float().mean().item()
            
            total_frame_acc += frame_acc
            total_gamaka_acc += gamaka_acc
            total_ctc += ctc_loss.item()
            n_batches += 1
            
    return total_frame_acc / max(1, n_batches), total_gamaka_acc / max(1, n_batches), total_ctc / max(1, n_batches)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 65)
    print("Training NoteScope Svara-Conformer Foundation Model (Option C)")
    print(f"Device: {device} | Vocabulary: {svara_tokenizer.vocab_size} tokens | Frontend: Tonic-Centered 144 bins")
    print("=" * 65)
    
    # Pre-generate datasets
    print("Pre-synthesizing in-memory training dataset (600 phrases across 16 ragas)...")
    t0 = time.time()
    train_phrases = [synthesize_vocal_phrase() for _ in range(600)]
    test_phrases = [synthesize_vocal_phrase() for _ in range(60)]
    t1 = time.time()
    print(f"Synthesized 660 diverse vocal phrases in {t1-t0:.2f}s!")
    
    train_loader = DataLoader(
        train_phrases,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_svara_batch
    )
    test_loader = DataLoader(
        test_phrases,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_svara_batch
    )
    
    model = SvaraConformer(
        input_dim=144,
        num_heads=4,
        ffn_dim=256,
        num_layers=4,
        num_classes=svara_tokenizer.vocab_size,
        depthwise_conv_kernel_size=15,
        dropout=0.1
    ).to(device)
    
    print(f"Initialized SvaraConformer with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    num_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    
    best_eval_acc = 0.0
    
    print("\nStarting Training...")
    start_train_t = time.time()
    for epoch in range(num_epochs):
        ep_t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, ctc_loss_fn, bce_loss_fn, ce_loss_fn, device)
        eval_acc, gamaka_acc, eval_ctc = evaluate(model, test_loader, ctc_loss_fn, ce_loss_fn, device)
        ep_t1 = time.time()
        
        print(f"Epoch [{epoch+1:2d}/{num_epochs:2d}] ({ep_t1-ep_t0:.2f}s) | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.1f}% | Eval Acc: {eval_acc*100:.1f}% | Gamaka Acc: {gamaka_acc*100:.1f}% | CTC Loss: {eval_ctc:.3f}")
        
        scheduler.step()
        
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab_size": svara_tokenizer.vocab_size,
                "best_eval_acc": best_eval_acc,
                "best_gamaka_acc": gamaka_acc,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }, CHECKPOINT_PATH)
            
    print("\n" + "=" * 65)
    print(f"Training Complete in {time.time()-start_train_t:.2f}s! Best Eval Frame Accuracy: {best_eval_acc*100:.2f}%")
    print(f"Saved optimal checkpoint to: {CHECKPOINT_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    main()
