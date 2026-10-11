"""
NoteScope — High-Speed Real Indian Classical Music Training Dataset Generator
Extracts high-precision pitch dynamics, acoustic onsets, and ground-truth swara frame alignments
from actual studio recordings of Carnatic vocal masters (CompMusic Carnatic Varnam Dataset).
"""

import os
import json
import numpy as np
import librosa
import torch
import torchcrepe
import compiam

# Device: Apple Silicon GPU (mps) or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Dataset extraction using device: {device}")

# 12 Indian Swaras definition
SWARAS_LIST = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
SWARA_TO_SEMITONE = {
    "S": 0, "R1": 1, "R2": 2, "G2": 3, "G3": 4, "M1": 5, "M2": 6,
    "P": 7, "D1": 8, "D2": 9, "N2": 10, "N3": 11
}

# Raga scale swarasthanas mapping
RAGA_SWARA_MAP = {
    "begada":  {"S": "S", "R": "R2", "G": "G3", "M": "M1", "P": "P", "D": "D2", "N": "N2"},
    "abhogi":  {"S": "S", "R": "R2", "G": "G2", "M": "M1", "P": "P", "D": "D2", "N": "N2"},
    "kalyani": {"S": "S", "R": "R2", "G": "G3", "M": "M2", "P": "P", "D": "D2", "N": "N3"},
    "mohanam": {"S": "S", "R": "R2", "G": "G3", "P": "P", "D": "D2"},
    "sahana":  {"S": "S", "R": "R2", "G": "G3", "M": "M1", "P": "P", "D": "D2", "N": "N2"},
    "saveri":  {"S": "S", "R": "R1", "G": "G2", "M": "M1", "P": "P", "D": "D1", "N": "N2"},
    "sri":     {"S": "S", "R": "R2", "G": "G2", "M": "M1", "P": "P", "D": "D2", "N": "N2"}
}

def parse_event_tokens(ev_str, raga_name):
    """Parses notation strings like 'DP', 'M,', 'S^N', 'GR' into swara tokens."""
    mapping = RAGA_SWARA_MAP.get(raga_name.lower(), {})
    tokens = []
    i = 0
    n = len(ev_str)
    while i < n:
        c = ev_str[i]
        if c in "SRGMPDN":
            saptak = 0
            if i + 1 < n and ev_str[i + 1] == "^":
                saptak = 1
                i += 1
            elif i + 1 < n and ev_str[i + 1] == "_":
                saptak = -1
                i += 1
            swara_spec = mapping.get(c, c)
            tokens.append({"type": "swara", "swara": swara_spec, "saptak": saptak})
        elif c == "_":
            if i + 1 < n and ev_str[i + 1] in "SRGMPDN":
                i += 1
                c2 = ev_str[i]
                swara_spec = mapping.get(c2, c2)
                tokens.append({"type": "swara", "swara": swara_spec, "saptak": -1})
        elif c == ",":
            tokens.append({"type": "sustain"})
        i += 1
    return tokens

def build_real_carnatic_dataset(output_dir="real_carnatic_dataset", max_tracks=20, segment_len_sec=10.0, sr=16000, hop_length=160):
    os.makedirs(output_dir, exist_ok=True)
    data = compiam.load_dataset("compmusic_carnatic_varnam")
    track_ids = data.track_ids[:max_tracks]
    
    metadata = []
    sample_idx = 0
    
    print(f"Processing {len(track_ids)} real Carnatic Varnam tracks...")
    
    for t_idx, tid in enumerate(track_ids):
        track = data.track(tid)
        if not track.notation or not track.tonic or not os.path.exists(track.audio_path):
            continue
            
        sa_hz = float(track.tonic)
        raga = track.raaga.lower()
        intervals = track.notation.intervals
        events = track.notation.events
        
        first_event_start = max(0.0, intervals[0][0] - 0.5)
        # Limit to first 180 seconds of singing per track to keep extraction fast & diverse across all tracks
        singing_dur = min(180.0, intervals[-1][1] - first_event_start)
        
        print(f"\n[{t_idx+1}/{len(track_ids)}] Processing track '{tid}' (Raaga: {raga}, Sa: {sa_hz:.1f}Hz, Singing: {singing_dur:.1f}s)...")
        
        # Load audio chunk
        y_track, _ = librosa.load(track.audio_path, sr=sr, offset=first_event_start, duration=singing_dur)
        
        # Extract f0 and voicing in ONE pass via TorchCrepe
        audio_t = torch.from_numpy(y_track).float().unsqueeze(0).to(device)
        try:
            pitch, periodicity = torchcrepe.predict(
                audio_t,
                sr,
                hop_length=hop_length,
                fmin=65,
                fmax=950,
                model="tiny",
                batch_size=1024,
                device=device,
                return_periodicity=True
            )
            f0_full = pitch.squeeze().cpu().numpy()
            voicing_full = periodicity.squeeze().cpu().numpy()
        except Exception as e:
            print(f"  Crepe error: {e}, falling back to pyin")
            f0_raw, _, voiced_probs = librosa.pyin(y_track, fmin=65, fmax=950, sr=sr, hop_length=hop_length)
            f0_full = np.nan_to_num(f0_raw)
            voicing_full = np.nan_to_num(voiced_probs)
            
        T_full = len(f0_full)
        track_times = first_event_start + np.arange(T_full) * 0.01
        
        # Continuous cents relative to Sa
        cents_full = np.zeros(T_full, dtype=np.float32)
        valid_full = (voicing_full > 0.4) & (f0_full > 40.0)
        if np.any(valid_full):
            cents_full[valid_full] = 1200.0 * np.log2(f0_full[valid_full] / sa_hz)
        cents_norm_full = cents_full / 1200.0
        cents_norm_full[~valid_full] = -2.0
        cents_norm_full = np.clip(cents_norm_full, -2.5, 2.5)
        
        # Pitch velocity & acceleration
        delta_cents_full = np.zeros(T_full, dtype=np.float32)
        delta_cents_full[1:] = np.diff(cents_norm_full)
        delta_cents_full = np.clip(delta_cents_full, -0.5, 0.5)
        
        delta2_cents_full = np.zeros(T_full, dtype=np.float32)
        delta2_cents_full[1:] = np.diff(delta_cents_full)
        delta2_cents_full = np.clip(delta2_cents_full, -0.5, 0.5)
        
        # Acoustic onset strength envelope & RMS energy
        onset_env_full = librosa.onset.onset_strength(y=y_track, sr=sr, hop_length=hop_length)
        onset_env_full = onset_env_full[:T_full]
        if len(onset_env_full) < T_full:
            onset_env_full = np.pad(onset_env_full, (0, T_full - len(onset_env_full)))
        if np.max(onset_env_full) > 0:
            onset_env_full = onset_env_full / np.max(onset_env_full)
            
        rms_full = librosa.feature.rms(y=y_track, hop_length=hop_length)[0][:T_full]
        if len(rms_full) < T_full:
            rms_full = np.pad(rms_full, (0, T_full - len(rms_full)))
        if np.max(rms_full) > 0:
            rms_full = rms_full / np.max(rms_full)
            
        # Multi-scale Gaussian Swara activation maps
        cents_mod1200 = cents_full % 1200.0
        g_fine_full = np.zeros((12, T_full), dtype=np.float32)
        g_coarse_full = np.zeros((12, T_full), dtype=np.float32)
        for k in range(12):
            target_cents = k * 100.0
            dist = np.abs(cents_mod1200 - target_cents)
            dist = np.minimum(dist, 1200.0 - dist)
            g_fine_full[k] = np.exp(-(dist ** 2) / (2.0 * (25.0 ** 2))) * valid_full
            g_coarse_full[k] = np.exp(-(dist ** 2) / (2.0 * (65.0 ** 2))) * valid_full
            
        # Saptak activations (3 channels: Mandra, Madhya, Taar)
        saptak_act_full = np.zeros((3, T_full), dtype=np.float32)
        saptak_act_full[0] = (cents_full < -50.0) & valid_full # Mandra
        saptak_act_full[1] = (cents_full >= -50.0) & (cents_full < 1150.0) & valid_full # Madhya
        saptak_act_full[2] = (cents_full >= 1150.0) & valid_full # Taar
        
        # 33-channel feature matrix for entire track
        features_full = np.vstack([
            cents_norm_full[np.newaxis, :],
            delta_cents_full[np.newaxis, :],
            delta2_cents_full[np.newaxis, :],
            voicing_full[np.newaxis, :],
            onset_env_full[np.newaxis, :],
            rms_full[np.newaxis, :],
            g_fine_full,
            g_coarse_full,
            saptak_act_full
        ]).astype(np.float32) # (33, T_full)
        
        # Build full track ground-truth frame alignments
        frame_onsets_full = np.zeros(T_full, dtype=np.float32)
        frame_swaras_full = np.zeros(T_full, dtype=np.int64)
        
        last_swara_idx = 12 # Default Madhya Sa
        for inv, ev in zip(intervals, events):
            t0, t1 = inv[0], inv[1]
            if t1 < first_event_start or t0 > (first_event_start + singing_dur):
                continue
                
            tokens = parse_event_tokens(ev, raga)
            if not tokens:
                continue
                
            n_tok = len(tokens)
            sub_dur = (t1 - t0) / max(1, n_tok)
            for tok_i, tok in enumerate(tokens):
                t_sub0 = t0 + tok_i * sub_dur
                t_sub1 = t_sub0 + sub_dur
                
                f0_idx = int(round((t_sub0 - first_event_start) * 100))
                f1_idx = int(round((t_sub1 - first_event_start) * 100))
                f0_idx = max(0, min(T_full, f0_idx))
                f1_idx = max(0, min(T_full, f1_idx))
                
                if f0_idx >= f1_idx:
                    continue
                    
                if tok["type"] == "swara":
                    sw_name = tok["swara"]
                    saptak = tok["saptak"] # -1, 0, 1
                    semitone = SWARA_TO_SEMITONE.get(sw_name, 0)
                    sw_class = (saptak + 1) * 12 + semitone # 0 to 35
                    last_swara_idx = sw_class
                    
                    frame_onsets_full[f0_idx] = 1.0
                    if f0_idx + 1 < T_full:
                        frame_onsets_full[f0_idx + 1] = 0.5
                    if f0_idx - 1 >= 0:
                        frame_onsets_full[f0_idx - 1] = 0.5
                    frame_swaras_full[f0_idx:f1_idx] = sw_class
                elif tok["type"] == "sustain":
                    frame_swaras_full[f0_idx:f1_idx] = last_swara_idx
                    
        # Now slice track into 10-second segments (1000 frames) with 3-second overlap (700 frames step)
        seg_frames = int(segment_len_sec * 100) # 1000 frames
        step_frames = 700
        
        cur_f = 0
        track_segments = 0
        while cur_f + seg_frames <= T_full:
            seg_feat = features_full[:, cur_f:cur_f + seg_frames]
            seg_onsets = frame_onsets_full[cur_f:cur_f + seg_frames]
            seg_swaras = frame_swaras_full[cur_f:cur_f + seg_frames]
            seg_cents = cents_full[cur_f:cur_f + seg_frames]
            seg_voicing = voicing_full[cur_f:cur_f + seg_frames]
            
            # Check that segment has reasonable vocal presence
            if np.mean(seg_voicing > 0.4) > 0.25:
                sample_name = f"sample_{sample_idx:05d}_{tid}.npz"
                save_path = os.path.join(output_dir, sample_name)
                np.savez_compressed(
                    save_path,
                    features=seg_feat,
                    onsets=seg_onsets,
                    swaras=seg_swaras,
                    cents=seg_cents,
                    voicing=seg_voicing
                )
                
                metadata.append({
                    "file": sample_name,
                    "track_id": tid,
                    "raga": raga,
                    "sa_hz": sa_hz,
                    "seg_start": first_event_start + cur_f * 0.01,
                    "seg_end": first_event_start + (cur_f + seg_frames) * 0.01,
                    "frames": seg_frames
                })
                sample_idx += 1
                track_segments += 1
                
            cur_f += step_frames
            
        print(f"  Track extracted {track_segments} segments (Total dataset: {sample_idx}).")
        
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n============================================================")
    print(f"Dataset generation complete!")
    print(f"Saved {sample_idx} real Carnatic music training segments in '{output_dir}'.")
    print(f"============================================================")
    return sample_idx

if __name__ == "__main__":
    build_real_carnatic_dataset()
