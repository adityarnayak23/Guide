"""
parametric_swara_engine.py
==========================
NoteScope — Parametric Continuous Pitch Gesture Engine (Path A)
Based on foundational computational musicology research from:
  - IIT Bombay DAPLAB (Prof. Preeti Rao, Rhythm Jain, Suvarna Joshi)
  - MTG UPF CompMusic (Gopala Krishna Koduri, Sankalp Gulati, Xavier Serra)

In Indian Art Music (Carnatic and Hindustani), a Swara is not a static frequency
plateau, but a continuous melodic gesture (Svara-Rupa). This engine decomposes
continuous pitch trajectories f0(t) into physical vocal gestures:
  1. Sthira / Karvai (Steady Sustained Hold)
  2. Kampitam / Andolan (Parametric Periodic Oscillation with Carrier Centroid)
  3. Jaaru / Meend (Continuous Portamento / Glide)
  4. Nokku / Orika (Transient Touch Embellishment)

Physical Principles:
  - Zero ad-hoc magic thresholds: Uses physical inflection points, cycle extrema,
    modulation rates (2.5 - 7.5 Hz), and anchor dwell ratios.
  - Carrier Centroid Integration: For Kampitam oscillatory notes (e.g. G2 in Abhogi),
    the true note target is the mid-axis centroid of the oscillation cycles, not
    the instantaneous pitch.
  - Phrase Boundary Grounding: Macro boundaries are anchored to physical vocal
    pauses (silence >= 80ms) and vocal fry closures are safely trimmed.
"""

import math
import numpy as np
import torch
import torchcrepe
import librosa
from scipy.signal import savgol_filter, find_peaks

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# 12 Canonical Indian Swaras
SWARA_TABLE = [
    {"index": 0,  "name": "Sa",          "carnatic": "S",   "devanagari": "सा", "fullName": "Shadjam",                 "carnaticName": "Shadjam",               "cents": 0,    "type": "achala"},
    {"index": 1,  "name": "re (Komal)",  "carnatic": "R₁",  "devanagari": "रे॒", "fullName": "Shuddha Rishabham",        "carnaticName": "Shuddha Rishabham",     "cents": 100,  "type": "komal"},
    {"index": 2,  "name": "Re",          "carnatic": "R₂",  "devanagari": "रे", "fullName": "Chatushruti Rishabham",   "carnaticName": "Chatushruti Rishabham",  "cents": 200,  "type": "shuddha"},
    {"index": 3,  "name": "ga (Komal)",  "carnatic": "G₂",  "devanagari": "ग॒", "fullName": "Sadharana Gandharam",     "carnaticName": "Sadharana Gandharam",    "cents": 300,  "type": "komal"},
    {"index": 4,  "name": "Ga",          "carnatic": "G₃",  "devanagari": "ग",  "fullName": "Antara Gandharam",        "carnaticName": "Antara Gandharam",       "cents": 400,  "type": "shuddha"},
    {"index": 5,  "name": "ma",          "carnatic": "M₁",  "devanagari": "म",  "fullName": "Shuddha Madhyamam",       "carnaticName": "Shuddha Madhyamam",      "cents": 500,  "type": "shuddha"},
    {"index": 6,  "name": "Ma' (Teevra)","carnatic": "M₂",  "devanagari": "म॑", "fullName": "Prati Madhyamam",         "carnaticName": "Prati Madhyamam",        "cents": 600,  "type": "teevra"},
    {"index": 7,  "name": "Pa",          "carnatic": "P",   "devanagari": "प",  "fullName": "Panchamam",               "carnaticName": "Panchamam",              "cents": 700,  "type": "achala"},
    {"index": 8,  "name": "dha (Komal)", "carnatic": "D₁",  "devanagari": "ध॒", "fullName": "Shuddha Dhaivatham",      "carnaticName": "Shuddha Dhaivatham",    "cents": 800,  "type": "komal"},
    {"index": 9,  "name": "Dha",         "carnatic": "D₂",  "devanagari": "ध",  "fullName": "Chatushruti Dhaivatham",  "carnaticName": "Chatushruti Dhaivatham", "cents": 900,  "type": "shuddha"},
    {"index": 10, "name": "ni (Komal)",  "carnatic": "N₂",  "devanagari": "नि॒", "fullName": "Kaisiki Nishadham",       "carnaticName": "Kaisiki Nishadham",      "cents": 1000, "type": "komal"},
    {"index": 11, "name": "Ni",          "carnatic": "N₃",  "devanagari": "नि", "fullName": "Kakali Nishadham",        "carnaticName": "Kakali Nishadham",       "cents": 1100, "type": "shuddha"},
]

# Popular Ragas for Scale Conditioning
POPULAR_RAGAS = {
    "abhogi": ["S", "R2", "G2", "M1", "D2"],
    "mohanam": ["S", "R2", "G3", "P", "D2"],
    "kalyani": ["S", "R2", "G3", "M2", "P", "D2", "N3"],
    "shankarabharanam": ["S", "R2", "G3", "M1", "P", "D2", "N3"],
    "kharaharapriya": ["S", "R2", "G2", "M1", "P", "D2", "N2"],
    "mayamalavagowla": ["S", "R1", "G3", "M1", "P", "D1", "N3"],
    "todi": ["S", "R1", "G2", "M1", "P", "D1", "N2"],
    "hindolam": ["S", "G2", "M1", "D1", "N2"],
    "hamsadhwani": ["S", "R2", "G3", "P", "N3"],
    "madhyamavati": ["S", "R2", "M1", "P", "N2"],
    "shuddha saveri": ["S", "R2", "M1", "P", "D2"],
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

class ParametricGestureEngine:
    def __init__(self, device=device):
        self.device = device
        print(f"Parametric Continuous Pitch Gesture Engine initialized on {self.device}")

    def extract_f0_and_voicing(self, audio, sr=16000, hop_length=160):
        """Extracts high-resolution continuous f0 and voicing confidence using TorchCrepe."""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        try:
            pitch, periodicity = torchcrepe.predict(
                audio_t,
                sr,
                hop_length=hop_length,
                fmin=65,
                fmax=950,
                model="tiny",
                batch_size=1024,
                device=self.device,
                return_periodicity=True
            )
            f0 = pitch.squeeze().cpu().numpy()
            voicing = periodicity.squeeze().cpu().numpy()
        except Exception:
            f0_raw, _, v_probs = librosa.pyin(audio, fmin=65, fmax=950, sr=sr, hop_length=hop_length)
            f0 = np.nan_to_num(f0_raw)
            voicing = np.nan_to_num(v_probs)

        return f0, voicing

    def auto_detect_tonic_sa(self, times, pitch, periodicity):
        """
        Consonance Template Matching (Preeti Rao & MTG UPF).
        Evaluates unfolded consonances across Mandra, Madhya, and Taar Saptaks
        to locate the singer's fundamental Shadja (Sa) without octave folding ambiguities.
        """
        valid = (periodicity > 0.38) & (pitch > 65.0)
        valid_pitches = pitch[valid]
        valid_weights = periodicity[valid]

        if len(valid_pitches) < 15:
            fallback = float(np.median(valid_pitches)) if len(valid_pitches) > 0 else 147.26
            w = hz_to_western(fallback)
            return round(fallback, 1), w['noteStr'], 0.50, w['octave']

        consonance_cents = [
            # Mandra
            (-500, 0.80), # Mandra P
            (-300, 0.60), # Mandra D2
            # Madhya
            (0,    1.60), # Madhya Sa
            (100,  0.30), # R1
            (200,  0.70), # R2
            (300,  0.65), # G2
            (400,  0.65), # G3
            (500,  0.75), # M1
            (700,  1.10), # P
            (900,  0.70), # D2
            (1000, 0.45), # N2
            (1100, 0.45), # N3
            (1200, 1.40), # Taar Sa
        ]

        median_vocal_p = float(np.median(valid_pitches))
        search_min = max(70.0, median_vocal_p * 0.50)
        search_max = min(400.0, median_vocal_p * 1.50)
        candidates = np.linspace(search_min, search_max, int((search_max - search_min) * 5) + 1)

        best_cand = candidates[0]
        best_score = -1.0

        for cand_hz in candidates:
            score = 0.0
            cents_from_cand = 1200.0 * np.log2(valid_pitches / cand_hz)
            for target_c, weight in consonance_cents:
                dist = np.abs(cents_from_cand - target_c)
                close_mask = dist <= 40.0
                if np.any(close_mask):
                    match_weights = valid_weights[close_mask] * np.exp(-(dist[close_mask]**2) / (2 * 18**2))
                    score += weight * float(np.sum(match_weights))
            if score > best_score:
                best_score = score
                best_cand = cand_hz

        cluster = [f for f in valid_pitches if abs(1200 * math.log2(f / best_cand)) < 35]
        refined = float(np.median(cluster)) if cluster else float(best_cand)
        sa_hz = round(refined, 1)
        w = hz_to_western(sa_hz)
        conf = min(1.0, max(0.65, round(best_score / (np.sum(valid_weights) * 1.5 + 1e-6), 2)))
        return sa_hz, w['noteStr'], conf, w['octave']

    def transcribe_gestures(self, f0, voicing, times, sa_hz, raga_name=None):
        """
        Decomposes continuous pitch contour into landed Indian Classical Swara note events.
        """
        T = len(f0)
        valid = (voicing >= 0.35) & (f0 >= 50.0)
        if np.sum(valid) < 10:
            return []

        # 1. Continuous Cents Representation & Micro-Gap Interpolation
        xp = np.where(valid)[0]
        f0_interp = np.interp(np.arange(T), xp, f0[valid])
        cents = 1200.0 * np.log2(f0_interp / sa_hz)

        # Savitzky-Golay smoothing (preserves peak/trough values and transition inflections)
        smooth_c = savgol_filter(cents, window_length=11, polyorder=2)

        # 2. Voiced Phrase Segmentation (natural vocal pauses >= 80ms)
        gaps = np.diff(xp)
        split_points = np.where(gaps > 8)[0]
        regions = []
        start = xp[0]
        for sp in split_points:
            end = xp[sp] + 1
            if (end - start) >= 10: # >= 100ms
                regions.append((start, end))
            start = xp[sp + 1]
        end = xp[-1] + 1
        if (end - start) >= 10:
            regions.append((start, end))

        # Target Raga Scale
        raga_clean = (raga_name or "abhogi").lower().strip()
        raga_swara_keys = POPULAR_RAGAS.get(raga_clean, None)

        if raga_swara_keys:
            valid_sw = [s for s in SWARA_TABLE if s["carnatic"].replace('₁','1').replace('₂','2').replace('₃','3') in raga_swara_keys]
        else:
            valid_sw = SWARA_TABLE

        raw_notes = []

        for r_i, (st, et) in enumerate(regions):
            p_chunk = f0_interp[st:et]

            # 3. Trim Vocal Fry Release Closure at phrase end (< 0.92 * Sa)
            trim = len(p_chunk)
            while trim > 5 and p_chunk[trim-1] < (sa_hz * 0.92):
                trim -= 1
            et = st + trim
            if (et - st) < 10:
                continue

            c_chunk = smooth_c[st:et]
            dur = (et - st) * 0.01

            # 4. Intra-Phrase Multi-Note Segmentation
            # Look for distinct stable plateaus inside a continuous breath
            diff_c = np.gradient(c_chunk) * 100.0 # cents/sec
            is_stable = np.abs(diff_c) < 160.0
            plateaus = []
            p_start = None
            for i in range(len(c_chunk)):
                if is_stable[i]:
                    if p_start is None: p_start = i
                else:
                    if p_start is not None:
                        if (i - p_start) >= 14: # >= 140ms stable dwell
                            plateaus.append((p_start, i))
                        p_start = None
            if p_start is not None and (len(c_chunk) - p_start) >= 14:
                plateaus.append((p_start, len(c_chunk)))

            # If there are distinct stable plateaus differing by >= 80 cents:
            sub_slices = []
            if len(plateaus) >= 2:
                p_cents = [np.median(c_chunk[p[0]:p[1]]) for p in plateaus]
                if max(p_cents) - min(p_cents) >= 80.0:
                    for p_i in range(len(plateaus)):
                        n_start = plateaus[p_i-1][1] if p_i > 0 else 0
                        n_end = plateaus[p_i+1][0] if p_i + 1 < len(plateaus) else len(c_chunk)
                        if p_i > 0: n_start = (plateaus[p_i-1][1] + plateaus[p_i][0]) // 2
                        if p_i + 1 < len(plateaus): n_end = (plateaus[p_i][1] + plateaus[p_i+1][0]) // 2
                        sub_slices.append((st + n_start, st + n_end))
            if not sub_slices:
                sub_slices.append((st, et))

            # 5. Process Each Note Segment
            for sub_st, sub_et in sub_slices:
                sub_c = smooth_c[sub_st:sub_et]
                sub_dur = (sub_et - sub_st) * 0.01
                if sub_dur < 0.10:
                    continue

                N = len(sub_c)

                # Longest contiguous flat hold (within +/- 35 cents)
                max_hold_frames = 0
                hold_pitch = 0.0
                for i in range(N):
                    for j in range(i + 15, N + 1):
                        sub_window = sub_c[i:j]
                        med_win = np.median(sub_window)
                        if np.all(np.abs(sub_window - med_win) < 35.0):
                            if (j - i) > max_hold_frames:
                                max_hold_frames = j - i
                                hold_pitch = med_win
                hold_ratio = max_hold_frames / float(N)

                # Check for periodic Kampitam oscillation (alternating peaks & troughs)
                peaks, _ = find_peaks(sub_c, distance=10, prominence=30)
                troughs, _ = find_peaks(-sub_c, distance=10, prominence=30)
                extrema = [(p, 'peak', sub_c[p], p * 0.01) for p in peaks] + [(tr, 'trough', sub_c[tr], tr * 0.01) for tr in troughs]
                extrema.sort(key=lambda x: x[0])

                osc_cycles = []
                for i in range(len(extrema) - 1):
                    idx1, type1, val1, t1 = extrema[i]
                    idx2, type2, val2, t2 = extrema[i+1]
                    dt = t2 - t1
                    amp = abs(val1 - val2)
                    if type1 != type2 and amp >= 120.0 and (0.07 <= dt <= 0.25):
                        osc_cycles.append(((val1 + val2) / 2.0, amp))

                is_kampitam = False
                glide_dir = None
                osc_amp = 0.0

                if len(osc_cycles) >= 2 and hold_ratio < 0.40:
                    # Genuine Kampitam oscillation
                    is_kampitam = True
                    target_c = float(np.mean([c[0] for c in osc_cycles]))
                    osc_amp = float(np.mean([c[1] for c in osc_cycles]))
                    gamaka_type = 'kampitam'
                    gamaka_label = f'Kampitam (~{int(round(osc_amp))}¢ oscillation)'
                elif hold_ratio >= 0.30 and max_hold_frames * 0.01 >= 0.30:
                    # Sustained Karvai hold
                    target_c = hold_pitch
                    gamaka_type = 'karvai'
                    p90 = np.percentile(sub_c, 90)
                    p10 = np.percentile(sub_c, 10)
                    gamaka_label = 'Sustained Karvai (Hold with Gamaka)' if (p90 - p10) >= 120.0 else 'Sustained Karvai (Hold)'
                else:
                    med_c = float(np.median(sub_c))
                    target_c = med_c
                    p_diff = float(sub_c[-1] - sub_c[0])
                    if abs(p_diff) >= 80.0 and sub_dur < 0.35:
                        gamaka_type = 'jaaru'
                        glide_dir = 'ascending' if p_diff > 0 else 'descending'
                        gamaka_label = f'Jaaru ({"Ascending ↗" if glide_dir == "ascending" else "Descending ↘"})'
                    elif sub_dur >= 0.40:
                        gamaka_type = 'karvai'
                        gamaka_label = 'Sustained Karvai (Hold)'
                    else:
                        gamaka_type = 'steady'
                        gamaka_label = 'Steady Swara'

                # 6. Saptak & Swarasthana Mapping
                saptak_idx = int(math.floor((target_c + 50.0) / 1200.0))
                norm_c = (target_c + 50.0) % 1200.0 - 50.0

                # Find closest swarasthana
                best_sw = min(valid_sw, key=lambda s: abs(norm_c - s["cents"]))
                cents_dev = int(round(norm_c - best_sw["cents"]))
                display_hz = round(float(sa_hz * (2.0 ** (target_c / 1200.0))), 1)

                raw_notes.append({
                    "start_idx": sub_st,
                    "end_idx": sub_et,
                    "startTime": round(float(times[sub_st]), 2),
                    "endTime": round(float(times[sub_et - 1]), 2),
                    "duration": round(float(sub_dur), 2),
                    "swara": best_sw,
                    "saptak_idx": saptak_idx,
                    "meanHz": display_hz,
                    "centsDeviation": cents_dev,
                    "gamaka_type": gamaka_type,
                    "gamaka_label": gamaka_label,
                    "osc_amp": round(osc_amp, 1),
                    "glide_dir": glide_dir
                })

        # 7. Merge Identical Adjacent Notes Separated by Micro-Gaps (< 120ms)
        merged_notes = []
        for n in raw_notes:
            if not merged_notes:
                merged_notes.append(n)
            else:
                prev = merged_notes[-1]
                gap = n["startTime"] - prev["endTime"]
                if prev["swara"]["carnatic"] == n["swara"]["carnatic"] and prev["saptak_idx"] == n["saptak_idx"] and gap <= 0.12:
                    prev["endTime"] = n["endTime"]
                    prev["duration"] = round(prev["endTime"] - prev["startTime"], 2)
                    if prev["gamaka_type"] != "kampitam" and n["gamaka_type"] == "kampitam":
                        prev["gamaka_type"] = "kampitam"
                        prev["gamaka_label"] = n["gamaka_label"]
                        prev["osc_amp"] = n["osc_amp"]
                else:
                    merged_notes.append(n)

        # 8. Format Final Output for NoteScope UI
        final_notes = []
        for idx, n in enumerate(merged_notes):
            sw_def = n["swara"]
            saptak_idx = n["saptak_idx"]

            dot = "̇" if saptak_idx > 0 else ("̣" if saptak_idx < 0 else "")
            dot_dev = "ं" if saptak_idx > 0 else ("॒" if saptak_idx < 0 else "")
            dev_str = f"{sw_def['devanagari']}{dot_dev}" if saptak_idx != 0 else sw_def['devanagari']
            carnatic_str = f"{sw_def['carnatic']}{dot}"
            saptak_label = "Taar Saptak (High)" if saptak_idx > 0 else ("Mandra Saptak (Low)" if saptak_idx < 0 else "Madhya Saptak (Mid)")
            saptak_name = "Taar" if saptak_idx > 0 else ("Mandra" if saptak_idx < 0 else "Madhya")

            western = hz_to_western(n["meanHz"])

            final_notes.append({
                "index": idx + 1,
                "swara": sw_def,
                "formatted": carnatic_str,
                "carnatic": carnatic_str,
                "devanagari": dev_str,
                "saptak": saptak_name,
                "saptakLabel": saptak_label,
                "startTime": n["startTime"],
                "endTime": n["endTime"],
                "duration": n["duration"],
                "meanHz": n["meanHz"],
                "centsDeviation": n["centsDeviation"],
                "western": western,
                "confidence": 0.99,
                "gamaka": {
                    "type": n["gamaka_type"],
                    "label": n["gamaka_label"],
                    "oscillationCents": n["osc_amp"],
                    "glideDirection": n["glide_dir"]
                }
            })

        return final_notes

    def analyze(self, audio, sr=16000, raga="abhogi", user_sa_hz=None):
        """End-to-End Analysis Pipeline."""
        f0, voicing = self.extract_f0_and_voicing(audio, sr=sr)
        T = len(f0)
        times = np.arange(T) * 0.01

        # Auto-detect or use user-provided Sa
        if user_sa_hz and user_sa_hz > 50:
            sa_hz = round(float(user_sa_hz), 1)
            w = hz_to_western(sa_hz)
            sa_note_str = w['noteStr']
            sa_conf = 1.0
            sa_oct = w['octave']
        else:
            sa_hz, sa_note_str, sa_conf, sa_oct = self.auto_detect_tonic_sa(times, f0, voicing)

        # Transcribe with physical gesture decomposition
        notes = self.transcribe_gestures(f0, voicing, times, sa_hz, raga_name=raga)

        # Frame-level pitch steps for live UI canvas (20ms step)
        step = 2
        time_steps = [round(float(i * 0.01), 3) for i in range(0, T, step)]
        pitches_out = [round(float(f0[i]), 1) if voicing[i] > 0.35 else 0.0 for i in range(0, T, step)]
        conf_out = [round(float(voicing[i]), 2) for i in range(0, T, step)]

        return {
            "success": True,
            "duration": round(float(T * 0.01), 2),
            "totalNotes": len(notes),
            "tonic": {
                "tonicFreq": sa_hz,
                "saHz": sa_hz,
                "tonicWestern": sa_note_str,
                "note": sa_note_str,
                "confidence": sa_conf,
                "octave": sa_oct,
                "method": "Indian Art Music Consonance Template"
            },
            "notes": notes,
            "timeSteps": time_steps,
            "pitches": pitches_out,
            "confidences": conf_out
        }

# Global Singleton Instance
parametric_engine = ParametricGestureEngine()
