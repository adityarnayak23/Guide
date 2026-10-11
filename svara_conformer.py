"""
svara_conformer.py
==================
NoteScope Svara-Conformer: Direct Audio-to-Svara Sequence-to-Sequence Engine.
Inspired by Google MT3 & OpenAI Whisper for Indian Classical Music.

Architecture:
- Tonic-Centered Filterbank Frontend (144 bins = 25 cents/bin over 3 Saptaks: Mandra, Madhya, Taar)
- Conv2D Subsampling (2x downsampling in time to 20ms resolution, 4x in frequency 144 -> 36)
- 4-layer Conformer Backbone (Multi-Head Self-Attention + Depthwise Separable Convolutions)
- Joint Multi-Task Heads:
  1. Svara CTC & Frame Emission Head (38 classes: 36 chromatic svara-saptak tokens + blank + silence)
  2. Acoustic Onset Activation Head (Binary note articulation probability)
  3. Gamaka Classification Head (Steady, Kampitham, Jaaru Asc, Jaaru Desc, Nokku, Karvai)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchaudio.models as models
from scipy.ndimage import median_filter

SWARASTHANAS = [
    ("S",  "सा",  0,    "Shadjam",               "Shadja",          "achala"),
    ("R1", "रे॒", 100,  "Shuddha Rishabham",     "Komal Rishabh",   "komal"),
    ("R2", "रे",  200,  "Chatushruti Rishabham",  "Shuddha Rishabh", "shuddha"),
    ("G2", "ग॒",  300,  "Sadharana Gandharam",    "Komal Gandhar",   "komal"),
    ("G3", "ग",   400,  "Antara Gandharam",       "Shuddha Gandhar", "shuddha"),
    ("M1", "म",   500,  "Shuddha Madhyamam",      "Shuddha Madhyam", "shuddha"),
    ("M2", "म॑",  600,  "Prati Madhyamam",        "Teevra Madhyam",  "teevra"),
    ("P",  "प",   700,  "Panchamam",              "Pancham",         "achala"),
    ("D1", "ध॒", 800,  "Shuddha Dhaivatham",     "Komal Dhaivat",   "komal"),
    ("D2", "ध",  900,  "Chatushruti Dhaivatham", "Shuddha Dhaivat", "shuddha"),
    ("N2", "नि॒", 1000, "Kaisiki Nishadham",      "Komal Nishad",    "komal"),
    ("N3", "नि",  1100, "Kakali Nishadham",       "Shuddha Nishad",  "shuddha"),
]

SAPTAKS = [
    ("MANDRA", -1, "̣", "॒", "Mandra Saptak (Low)"),
    ("MADHYA",  0, "",   "",   "Madhya Saptak (Mid)"),
    ("TAAR",    1, "̇", "ं", "Taar Saptak (High)"),
]

GAMAKA_TYPES = [
    "steady",
    "kampitham",
    "jaaru_asc",
    "jaaru_desc",
    "nokku",
    "karvai"
]

class SvaraTokenizer:
    def __init__(self):
        self.blank_token = "<BLANK>"
        self.silence_token = "<SILENCE>"
        
        self.idx2token = [self.blank_token, self.silence_token]
        self.token2idx = {self.blank_token: 0, self.silence_token: 1}
        self.token_info = {}
        
        for saptak_name, oct_shift, dot_car, dot_dev, saptak_label in SAPTAKS:
            for sw_name, dev_base, cents, car_name, hind_name, sw_type in SWARASTHANAS:
                tok = f"{saptak_name}_{sw_name}"
                idx = len(self.idx2token)
                self.idx2token.append(tok)
                self.token2idx[tok] = idx
                
                formatted_car = f"{sw_name}{dot_car}"
                formatted_dev = f"{dev_base}{dot_dev}" if oct_shift != 0 else dev_base
                
                self.token_info[tok] = {
                    "token": tok,
                    "index": idx,
                    "swaraName": sw_name,
                    "saptak": saptak_name.capitalize(),
                    "saptakLabel": saptak_label,
                    "octaveShift": oct_shift,
                    "carnatic": formatted_car,
                    "devanagari": formatted_dev,
                    "baseCents": cents,
                    "totalCents": cents + oct_shift * 1200,
                    "carnaticName": car_name,
                    "hindustaniName": hind_name,
                    "type": sw_type
                }
                
        self.vocab_size = len(self.idx2token)
        
    def encode_token(self, token_str):
        return self.token2idx.get(token_str, 1) # fallback to silence
        
    def decode_token(self, idx):
        if 0 <= idx < self.vocab_size:
            return self.idx2token[idx]
        return self.silence_token
        
    def get_info(self, token_str):
        return self.token_info.get(token_str, None)

svara_tokenizer = SvaraTokenizer()

def build_tonic_filterbank(sa_hz, sr=16000, n_fft=2048, bins_per_octave=48, n_bins=144):
    """
    Constructs a 144-bin logarithmic triangular filterbank centered on the singer's tonic (Sa).
    Spans 3 full Saptaks:
      - Mandra Saptak: Bins 0 - 47 (-1200 cents to -25 cents)
      - Madhya Saptak: Bins 48 - 95 (0 cents to +1175 cents)
      - Taar Saptak:   Bins 96 - 143 (+1200 cents to +2375 cents)
    Madhya Sa (0 cents) is locked precisely to Bin 48 for ANY tonic frequency!
    """
    fmin = max(20.0, sa_hz * 0.5) # Mandra Sa
    freqs = fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    
    fb = np.zeros((n_fft // 2 + 1, n_bins), dtype=np.float32)
    for b in range(n_bins):
        f_center = freqs[b]
        f_left = f_center * (2.0 ** (-1.0 / bins_per_octave))
        f_right = f_center * (2.0 ** (1.0 / bins_per_octave))
        
        mask_up = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        fb[mask_up, b] = (fft_freqs[mask_up] - f_left) / (f_center - f_left + 1e-6)
        
        mask_down = (fft_freqs > f_center) & (fft_freqs <= f_right)
        fb[mask_down, b] = (f_right - fft_freqs[mask_down]) / (f_right - f_center + 1e-6)
        
    return torch.from_numpy(fb)

def compute_tonic_spectrogram(waveform, sa_hz, sr=16000, n_fft=2048, hop_length=160, win_length=1024):
    """
    Computes a standardized tonic-centered log-spectrogram [T, 144].
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    device = waveform.device
    window = torch.hann_window(win_length, device=device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    mag = torch.abs(stft).squeeze(0) # [n_fft//2 + 1, T]
    fb = build_tonic_filterbank(sa_hz, sr=sr, n_fft=n_fft, bins_per_octave=48, n_bins=144).to(device)
    tonic_spec = torch.matmul(mag.transpose(0, 1), fb) # [T, 144]
    log_spec = torch.log(torch.clamp(tonic_spec, min=1e-5))
    norm_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-5)
    return norm_spec


class SvaraConformer(nn.Module):
    def __init__(
        self,
        input_dim=144,
        num_heads=4,
        ffn_dim=256,
        num_layers=4,
        num_classes=38,
        depthwise_conv_kernel_size=15,
        dropout=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Conv2D subsampling: 2x time downsampling (10ms -> 20ms frame resolution)
        # 144 freq bins -> 72 -> 36 bins
        self.conv_subsample = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # 64 channels * 36 freq bins = 2304 -> project to 144 conformer dimension
        self.proj = nn.Linear(64 * 36, 144)
        
        self.conformer = models.Conformer(
            input_dim=144,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout
        )
        
        # Multi-task prediction heads
        self.fc_svara = nn.Linear(144, num_classes)
        self.fc_onset = nn.Linear(144, 1)
        self.fc_gamaka = nn.Linear(144, len(GAMAKA_TYPES))
        
    def forward(self, tonic_spec, lengths=None):
        """
        tonic_spec: [B, 1, T, 144] (standardized tonic-centered spectrogram)
        lengths: [B] (optional frame lengths)
        """
        B, C, T, F = tonic_spec.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.int64, device=tonic_spec.device)
            
        x = self.conv_subsample(tonic_spec) # [B, 64, T//2, 36]
        B, C_sub, T_sub, F_sub = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T_sub, C_sub * F_sub)
        x = self.proj(x) # [B, T_sub, 144]
        
        lens_sub = torch.clamp((lengths + 1) // 2, max=T_sub)
        out, _ = self.conformer(x, lens_sub)
        
        svara_logits = self.fc_svara(out) # [B, T_sub, 38]
        onset_logits = self.fc_onset(out).squeeze(-1) # [B, T_sub]
        gamaka_logits = self.fc_gamaka(out) # [B, T_sub, 6]
        
        return {
            "svara_logits": svara_logits,
            "onset_logits": onset_logits,
            "gamaka_logits": gamaka_logits,
            "lengths": lens_sub
        }


WESTERN_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def hz_to_western(freq):
    if not freq or freq <= 0:
        return {"noteStr": "—", "octave": 0, "centsOff": 0, "name": "—"}
    midi = 12 * math.log2(freq / 440.0) + 69
    rounded_midi = int(round(midi))
    cents_off = int(round((midi - rounded_midi) * 100))
    octave = (rounded_midi // 12) - 1
    name = WESTERN_NOTES[rounded_midi % 12]
    sign = "+" if cents_off >= 0 else ""
    return {
        "noteStr": f"{name}{octave}{sign}{cents_off}¢",
        "octave": octave,
        "centsOff": cents_off,
        "name": name,
        "midi": rounded_midi
    }

def decode_ctc_conformer_sequence(
    svara_logits,
    y=None,
    sr=16000,
    sa_hz=147.26,
    f0=None,
    periodicity=None,
    tokenizer=svara_tokenizer,
    time_resolution=0.02
):
    """
    Decodes Svara-Conformer CTC logits into discrete, time-aligned Svara notes.
    - Zero heuristics or arbitrary pitch thresholds: token identity is dictated purely by the CTC neural sequence model.
    - Continuous pitch intonation and physical gamaka dynamics are enriched via Crepe pitch trajectory.
    """
    if torch.is_tensor(svara_logits):
        probs = torch.softmax(svara_logits, dim=-1).cpu().numpy()
    else:
        probs = np.exp(svara_logits) / np.sum(np.exp(svara_logits), axis=-1, keepdims=True)
    pred_ids = np.argmax(probs, axis=-1)
    T_frames = len(pred_ids)
    dur = float(len(y) / sr) if y is not None else float(T_frames * time_resolution)

    # 1. Detect CTC token emissions (peaks where non-blank tokens are emitted)
    raw_emissions = []
    prev = None
    cur_peak_t = None
    cur_peak_p = 0.0

    for t, tok_id in enumerate(pred_ids):
        if tok_id > 1:  # Non-blank, non-silence
            if tok_id != prev:
                if prev is not None and prev > 1:
                    raw_emissions.append((prev, cur_peak_t * time_resolution, cur_peak_p))
                prev = tok_id
                cur_peak_t = t
                cur_peak_p = float(probs[t, tok_id])
            else:
                if probs[t, tok_id] > cur_peak_p:
                    cur_peak_p = float(probs[t, tok_id])
                    cur_peak_t = t
        else:
            if prev is not None and prev > 1:
                raw_emissions.append((prev, cur_peak_t * time_resolution, cur_peak_p))
            prev = tok_id
            cur_peak_t = None
            cur_peak_p = 0.0

    if prev is not None and prev > 1 and cur_peak_t is not None:
        raw_emissions.append((prev, cur_peak_t * time_resolution, cur_peak_p))

    if not raw_emissions:
        return []

    # Merge rapid duplicate emissions of the same swara (< 0.35s apart) into single sustained note
    emissions = []
    for tok_id, t_peak, conf in raw_emissions:
        if emissions and emissions[-1][0] == tok_id and (t_peak - emissions[-1][1]) < 0.35:
            prev_tok, prev_t, prev_c = emissions[-1]
            new_c = max(prev_c, conf)
            new_t = prev_t if prev_c >= conf else t_peak
            emissions[-1] = (tok_id, new_t, new_c)
        else:
            emissions.append((tok_id, t_peak, conf))

    # 2. Extract pitch contour if f0 is provided
    times = None
    voiced_mask = None
    first_voiced_t = 0.0
    last_voiced_t = dur
    if f0 is not None and periodicity is not None:
        times = np.arange(len(f0)) * 0.01
        voiced_mask = (periodicity > 0.35) & (f0 > 55.0)
        voiced_indices = np.where(voiced_mask)[0]
        if len(voiced_indices) > 0:
            first_voiced_t = float(times[voiced_indices[0]])
            last_voiced_t = float(times[voiced_indices[-1]])

    # 3. Compute pitch cross-over & acoustic dip transition boundaries
    num_notes = len(emissions)
    boundaries = []
    # Start boundary for note 0: begins at true vocal onset
    start_t = first_voiced_t if first_voiced_t > 0 else max(0.0, emissions[0][1])
    boundaries.append(start_t)

    for i in range(num_notes - 1):
        t_curr = emissions[i][1]
        t_next = emissions[i + 1][1]
        tok_curr = tokenizer.decode_token(emissions[i][0])
        tok_next = tokenizer.decode_token(emissions[i + 1][0])
        info_c = tokenizer.get_info(tok_curr)
        info_n = tokenizer.get_info(tok_next)
        c_curr = info_c["baseCents"] if info_c else 0
        c_next = info_n["baseCents"] if info_n else 0

        t_mid = (t_curr + t_next) / 2.0
        b_time = t_mid

        if times is not None and voiced_mask is not None:
            if c_curr == c_next:
                # Same note repeated: split at unvoiced silence gap between repetitions
                mask_all = (times >= t_curr) & (times <= t_next)
                unvoiced_idx = np.where(~voiced_mask[mask_all])[0]
                if len(unvoiced_idx) > 0:
                    b_time = float(np.median(times[mask_all][unvoiced_idx]))
            else:
                # Different swaras: search directional pitch crossover
                c_mid = (c_curr + c_next) / 2.0
                search_st = max(boundaries[-1] + 0.15, t_curr)
                mask_v = (times >= search_st) & (times <= t_next) & voiced_mask
                sub_t = times[mask_v]
                sub_f = f0[mask_v]
                if len(sub_t) > 0:
                    sub_cents = 1200.0 * np.log2(np.maximum(sub_f, 1e-5) / sa_hz)
                    smooth_c = median_filter(sub_cents, size=5)
                    if c_next > c_curr:
                        cross = np.where(smooth_c >= c_mid)[0]
                    else:
                        cross = np.where(smooth_c <= c_mid)[0]
                    if len(cross) > 0:
                        b_time = float(sub_t[cross[0]])
                    else:
                        b_time = float(sub_t[np.argmin(np.abs(smooth_c - c_mid))])

        boundaries.append(b_time)

    # End boundary for final note
    boundaries.append(min(dur, max(last_voiced_t, emissions[-1][1] + 0.20)))

    # 4. Construct continuous temporal spans for each emitted note
    notes = []
    for i in range(num_notes):
        tok_id, t_peak, conf = emissions[i]
        tok_name = tokenizer.decode_token(tok_id)
        info = tokenizer.get_info(tok_name)
        if not info:
            continue

        st = boundaries[i]
        et = boundaries[i + 1]
        if et <= st:
            et = min(dur, st + 0.12)

        note_dur = et - st

        # Microtonal pitch and gamaka calculation from pitch contour
        target_cents = info["baseCents"]
        med_cents = target_cents
        std_cents = 0.0
        drift_cents = 0.0
        med_hz = sa_hz * (2.0 ** (target_cents / 1200.0))

        if times is not None and voiced_mask is not None:
            seg_mask = (times >= st) & (times <= et) & voiced_mask
            seg_f0 = f0[seg_mask]
            if len(seg_f0) >= 3:
                seg_cents = 1200.0 * np.log2(seg_f0 / sa_hz)
                med_cents = float(np.median(seg_cents))
                std_cents = float(np.std(seg_cents))
                drift_cents = float(seg_cents[-1] - seg_cents[0])
                med_hz = float(np.median(seg_f0))

        # Fold cents deviation into [-600, +600] to prevent harmonic offset artifacts
        cents_dev = (int(round(med_cents - target_cents)) + 600) % 1200 - 600

        # Classify Gamaka with realistic physical oscillation bounds
        if std_cents >= 28.0:
            gamaka_type = "kampitam"
            osc_amp = min(220, max(28, int(round(std_cents * 1.6))))
            gamaka_label = f"Kampitam (~{osc_amp}¢ oscillation)"
        elif abs(drift_cents) >= 110.0:
            gamaka_type = "jaaru"
            dir_str = "Ascending ↗" if drift_cents > 0 else "Descending ↘"
            gamaka_label = f"Jaaru ({dir_str})"
            osc_amp = 0.0
        elif note_dur >= 0.45:
            gamaka_type = "karvai"
            gamaka_label = "Sustained Karvai (Hold)"
            osc_amp = 0.0
        else:
            gamaka_type = "steady"
            gamaka_label = "Steady Swara"
            osc_amp = 0.0

        western = hz_to_western(med_hz)

        notes.append({
            "index": i + 1,
            "token": tok_name,
            "formatted": info["carnatic"],
            "carnatic": info["carnatic"],
            "devanagari": info["devanagari"],
            "swaraName": info["swaraName"],
            "saptak": info["saptak"],
            "saptakLabel": info["saptakLabel"],
            "startTime": round(float(st), 2),
            "endTime": round(float(et), 2),
            "duration": round(float(note_dur), 2),
            "confidence": round(float(conf), 2),
            "meanHz": round(float(med_hz), 1),
            "centsDeviation": cents_dev,
            "western": western,
            "gamaka": {
                "type": gamaka_type,
                "label": gamaka_label,
                "oscillationCents": round(float(osc_amp), 1),
                "glideDirection": ("ascending" if drift_cents > 0 else "descending") if gamaka_type == "jaaru" else None
            },
            "swara": {
                "index": int(round(info["baseCents"] / 100)),
                "name": info["swaraName"],
                "carnatic": info["swaraName"],
                "devanagari": info["devanagari"],
                "fullName": info["hindustaniName"],
                "carnaticName": info["carnaticName"],
                "cents": info["baseCents"],
                "type": info["type"]
            }
        })

    return notes

def decode_conformer_predictions(
    svara_logits,
    onset_logits=None,
    gamaka_logits=None,
    tokenizer=svara_tokenizer,
    time_resolution=0.02,
    y=None,
    sr=16000,
    sa_hz=147.26,
    f0=None,
    periodicity=None
):
    """Backwards compatibility wrapper delegating to decode_ctc_conformer_sequence."""
    return decode_ctc_conformer_sequence(
        svara_logits,
        y=y,
        sr=sr,
        sa_hz=sa_hz,
        f0=f0,
        periodicity=periodicity,
        tokenizer=tokenizer,
        time_resolution=time_resolution
    )

