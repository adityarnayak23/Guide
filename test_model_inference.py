"""
Test script to verify CarnaticSwaraNet inference on audio files.
"""

import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf
from train_carnatic_model import CarnaticSwaraNet, device

SWARAS_LIST = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
GAMAKAS_LIST = ["steady", "kampitam", "jaaru", "karvai"]

def load_trained_model(model_path="models/carnatic_swara_model.pt"):
    checkpoint = torch.load(model_path, map_location=device)
    model = CarnaticSwaraNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded CarnaticSwaraNet (Val Acc: {checkpoint['val_acc']:.1f}%, Onset F1: {checkpoint['onset_f1']:.1f}%)")
    return model

def infer_notes(audio_path, model, sa_hz=146.8, sr=16000, hop_length=160):
    audio, file_sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if file_sr != sr:
        audio_t = torch.from_numpy(audio).float().unsqueeze(0)
        audio_t = torchaudio.transforms.Resample(file_sr, sr)(audio_t)
        audio = audio_t.squeeze(0).numpy()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=hop_length, n_mels=80, f_min=60.0, f_max=4000.0
    )
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    mel = mel_transform(audio_t).squeeze(0) # (80, T)
    mel = torch.log1p(10.0 * mel)

    T = mel.shape[1]
    
    # Extract f0 using torchcrepe
    try:
        import torchcrepe
        pitch, periodicity = torchcrepe.predict(
            audio_t,
            sr,
            hop_length=hop_length,
            fmin=65,
            fmax=950,
            model="tiny",
            batch_size=512,
            device=device,
            return_periodicity=True
        )
        f0 = pitch.squeeze().cpu().numpy()
        voicing = periodicity.squeeze().cpu().numpy()
    except Exception as e:
        import librosa
        f0_raw, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=65, fmax=950, sr=sr, hop_length=hop_length
        )
        f0 = np.nan_to_num(f0_raw)
        voicing = np.nan_to_num(voiced_probs)

    f0 = f0[:T]
    voicing = voicing[:T]
    if len(f0) < T:
        pad = T - len(f0)
        f0 = np.pad(f0, (0, pad))
        voicing = np.pad(voicing, (0, pad))

    cents = np.zeros(T, dtype=np.float32)
    valid = (f0 > 40.0) & (voicing > 0.4)
    if np.any(valid):
        cents[valid] = 1200.0 * np.log2(f0[valid] / float(sa_hz))
    cents_norm = cents / 1200.0
    cents_norm[~valid] = -2.0
    cents_norm = np.clip(cents_norm, -2.5, 2.5)

    delta_cents = np.zeros(T, dtype=np.float32)
    delta_cents[1:] = np.diff(cents_norm)
    delta_cents = np.clip(delta_cents, -0.5, 0.5)

    cents_t = torch.from_numpy(cents_norm).float().unsqueeze(0)
    delta_t = torch.from_numpy(delta_cents).float().unsqueeze(0)
    voicing_t = torch.from_numpy(voicing).float().unsqueeze(0)

    features = torch.cat([mel, cents_t, delta_t, voicing_t], dim=0).unsqueeze(0).to(device) # (1, 83, T)

    with torch.no_grad():
        preds = model(features)
        onset_probs = torch.sigmoid(preds["onset_logits"]).squeeze(0).cpu().numpy() # (T,)
        swara_probs = torch.softmax(preds["swara_logits"], dim=-1).squeeze(0).cpu().numpy() # (T, 36)
        gamaka_probs = torch.softmax(preds["gamaka_logits"], dim=-1).squeeze(0).cpu().numpy() # (T, 4)

    # Detect onsets with adaptive threshold and min interval
    onsets = []
    min_dist_frames = 15 # 150ms minimum duration
    for i in range(1, T - 1):
        if onset_probs[i] > 0.40 and onset_probs[i] > onset_probs[i - 1] and onset_probs[i] >= onset_probs[i + 1]:
            if not onsets or (i - onsets[-1] >= min_dist_frames):
                onsets.append(i)

    if not onsets or onsets[0] > 10:
        onsets.insert(0, 0)
    onsets.append(T)

    notes = []
    for i in range(len(onsets) - 1):
        s_frame = onsets[i]
        e_frame = onsets[i + 1]
        dur_sec = (e_frame - s_frame) * 0.01

        if dur_sec < 0.12:
            continue

        # Segment swara from majority probability
        seg_swaras = swara_probs[s_frame:e_frame]
        mean_swara_probs = np.mean(seg_swaras, axis=0)
        best_sw_idx = int(np.argmax(mean_swara_probs))

        saptak_idx = (best_sw_idx // 12) - 1 # -1, 0, +1
        sw_name = SWARAS_LIST[best_sw_idx % 12]

        seg_gamakas = gamaka_probs[s_frame:e_frame]
        best_gamaka_idx = int(np.argmax(np.mean(seg_gamakas, axis=0)))
        gamaka_name = GAMAKAS_LIST[best_gamaka_idx]

        st_sec = round(s_frame * 0.01, 2)
        et_sec = round(e_frame * 0.01, 2)

        notes.append({
            "start": st_sec,
            "end": et_sec,
            "duration": round(dur_sec, 2),
            "swara": sw_name,
            "saptak": saptak_idx,
            "gamaka": gamaka_name,
            "confidence": round(float(mean_swara_probs[best_sw_idx]), 2)
        })

    return notes

if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "public/samples/vocal_sargam.wav"
    model = load_trained_model()
    notes = infer_notes(audio_path, model, sa_hz=146.8)
    print(f"\nInferred Notes for '{audio_path}':")
    print(f"{'Time Span':16} | {'Swara':6} | {'Saptak':8} | {'Gamaka':10} | {'Conf':5}")
    print("-" * 55)
    for n in notes:
        sap_str = "Taar" if n["saptak"] > 0 else ("Mandra" if n["saptak"] < 0 else "Madhya")
        print(f"[{n['start']:5.2f}s - {n['end']:5.2f}s] | {n['swara']:6} | {sap_str:8} | {n['gamaka']:10} | {n['confidence']:0.2f}")
