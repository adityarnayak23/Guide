"""
NoteScope — Authentic Carnatic & Indian Art Music Dataset Generator
Generates high-fidelity vocal audio of Carnatic compositions, varisais, and alapanas
with authentic gamakas (kampitam, jaaru, karvai), tanpura drone, and time-aligned ground-truth labels.
"""

import os
import math
import json
import random
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# 12 Swaras definitions
SWARAS = [
    {"name": "S",  "cents": 0,    "type": "achala"},
    {"name": "R1", "cents": 100,  "type": "komal"},
    {"name": "R2", "cents": 200,  "type": "shuddha"},
    {"name": "G2", "cents": 300,  "type": "komal"},
    {"name": "G3", "cents": 400,  "type": "shuddha"},
    {"name": "M1", "cents": 500,  "type": "shuddha"},
    {"name": "M2", "cents": 600,  "type": "teevra"},
    {"name": "P",  "cents": 700,  "type": "achala"},
    {"name": "D1", "cents": 800,  "type": "komal"},
    {"name": "D2", "cents": 900,  "type": "shuddha"},
    {"name": "N2", "cents": 1000, "type": "komal"},
    {"name": "N3", "cents": 1100, "type": "shuddha"},
]

SWARA_TO_IDX = {s["name"]: i for i, s in enumerate(SWARAS)}

# Carnatic Raga Scales
RAGAS = {
    "Mayamalavagowla":  ["S", "R1", "G3", "M1", "P", "D1", "N3"],
    "Shankarabharanam":  ["S", "R2", "G3", "M1", "P", "D2", "N3"],
    "Kalyani":           ["S", "R2", "G3", "M2", "P", "D2", "N3"],
    "Kharaharapriya":    ["S", "R2", "G2", "M1", "P", "D2", "N2"],
    "Todi":              ["S", "R1", "G2", "M1", "P", "D1", "N2"],
    "Harikambhoji":      ["S", "R2", "G3", "M1", "P", "D2", "N2"],
    "Natabhairavi":      ["S", "R1", "G2", "M1", "P", "D1", "N2"],
    "Mohanam":           ["S", "R2", "G3", "P", "D2"],
    "Hamsadhwani":       ["S", "R2", "G3", "P", "N3"],
    "Hindolam":          ["S", "G2", "M1", "D1", "N2"],
    "Madhyamavati":      ["S", "R2", "M1", "P", "N2"],
    "Abhogi":            ["S", "R2", "G2", "M1", "D2"],
    "Shuddha Saveri":    ["S", "R2", "M1", "P", "D2"],
    "Malahari":          ["S", "R1", "M1", "P", "D1"],
}

# Formant frequencies for vocal vowels (F1, F2, F3) in Hz
VOWEL_FORMANTS = {
    "a": (800, 1200, 2500),
    "i": (300, 2300, 3000),
    "u": (350, 800, 2300),
    "e": (500, 1800, 2500),
    "o": (500, 1000, 2400),
}

def apply_formants(signal, sr, f1, f2, f3):
    """Applies vocal tract resonance via 3-formant second-order resonant filters."""
    out = np.zeros_like(signal)
    for f, bw, gain in [(f1, 80, 1.0), (f2, 110, 0.6), (f3, 150, 0.3)]:
        nyq = sr / 2.0
        low = max(20, f - bw / 2.0) / nyq
        high = min(nyq - 20, f + bw / 2.0) / nyq
        if 0 < low < high < 1:
            b, a = butter(2, [low, high], btype='band')
            filtered = lfilter(b, a, signal)
            out += filtered * gain
    return out

def generate_tanpura_drone(sa_hz, duration, sr=16000):
    """Synthesizes rich acoustic Tanpura drone with sympathetic string overtones."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pa string (0.75 * Sa or 1.5 * Sa)
    pa_hz = sa_hz * (3.0 / 2.0)
    # 4 strings plucked in cycle: Pa, Sa (mid), Sa (mid), Sa (low)
    drone = np.zeros_like(t)
    strings = [pa_hz, sa_hz, sa_hz, sa_hz * 0.5]
    plucks_per_string = int(sr * 1.8) # Pluck interval

    for idx, f in enumerate(strings):
        for h in range(1, 12):
            amp = (1.0 / (h ** 1.3)) * 0.04
            decay = 0.8 + 0.2 * math.cos(h)
            drone += amp * np.sin(2 * np.pi * f * h * t + (idx * 0.7)) * np.exp(-0.02 * t)
    
    # Soft low-pass
    nyq = sr / 2.0
    b, a = butter(2, 2400 / nyq, btype='low')
    drone = lfilter(b, a, drone)
    return drone * 0.18

def synthesize_carnatic_vocal_phrase(notes_seq, sa_hz, sr=16000):
    """
    Synthesizes a realistic vocal performance of a sequence of swaras with authentic gamakas.
    Each item in notes_seq is a dict:
    {
      "swara": "G3",
      "saptak": 0, # -1, 0, 1
      "dur": 0.5,
      "gamaka": "kampitam" | "jaaru" | "karvai" | "steady",
      "vowel": "a"
    }
    """
    total_dur = sum(n["dur"] for n in notes_seq) + 0.2
    total_samples = int(sr * total_dur)
    vocal = np.zeros(total_samples)
    
    frame_hop = 160 # 10ms frame at 16kHz (100 fps)
    total_frames = int(math.ceil(total_dur * 100))
    
    frame_onsets = np.zeros(total_frames, dtype=np.float32)
    frame_swaras = np.zeros(total_frames, dtype=np.int64) # 0 to 35
    frame_gamakas = np.zeros(total_frames, dtype=np.int64) # 0 to 3
    frame_f0 = np.zeros(total_frames, dtype=np.float32)
    frame_voicing = np.zeros(total_frames, dtype=np.float32)
    
    cur_sample = 0
    cur_time = 0.0

    # Natural micro-vibrato (5.5 Hz)
    vibrato_rate = 5.3 + random.uniform(-0.3, 0.3)

    for note_idx, note_spec in enumerate(notes_seq):
        sw_name = note_spec["swara"]
        saptak = note_spec.get("saptak", 0) # -1, 0, +1
        dur = note_spec["dur"]
        gamaka = note_spec.get("gamaka", "steady")
        vowel = note_spec.get("vowel", "a")
        
        sw_cents = SWARAS[SWARA_TO_IDX[sw_name]]["cents"] + (saptak * 1200)
        target_f0 = sa_hz * (2.0 ** (sw_cents / 1200.0))
        
        n_samples = int(sr * dur)
        t_note = np.linspace(0, dur, n_samples, endpoint=False)
        
        # 1. Pitch Contour with Gamakas
        base_f0 = np.full(n_samples, target_f0)
        
        if gamaka == "kampitam":
            # Sinusoidal / triangular oscillation
            osc_rate = random.uniform(4.5, 6.2) # Hz
            osc_depth_cents = random.uniform(70, 160) # Cents oscillation
            osc_f0 = target_f0 * (2.0 ** ((osc_depth_cents * np.sin(2 * np.pi * osc_rate * t_note)) / 1200.0))
            f0_contour = osc_f0
        elif gamaka == "jaaru":
            # Portamento glide from previous note or to next note
            prev_f0 = target_f0 * 0.85
            if note_idx > 0:
                prev_sw = notes_seq[note_idx - 1]["swara"]
                prev_sap = notes_seq[note_idx - 1].get("saptak", 0)
                prev_cents = SWARAS[SWARA_TO_IDX[prev_sw]]["cents"] + (prev_sap * 1200)
                prev_f0 = sa_hz * (2.0 ** (prev_cents / 1200.0))
            # Sigmoidal glide over first 40% of duration
            glide_ratio = 1.0 / (1.0 + np.exp(-12.0 * (t_note / dur - 0.25)))
            f0_contour = prev_f0 + (target_f0 - prev_f0) * glide_ratio
        elif gamaka == "karvai":
            # Sustained with subtle micro-vibrato
            vib = np.sin(2 * np.pi * vibrato_rate * t_note) * 22.0 # 22 cents
            f0_contour = target_f0 * (2.0 ** (vib / 1200.0))
        else: # steady
            vib = np.sin(2 * np.pi * vibrato_rate * t_note) * 15.0 # 15 cents
            f0_contour = target_f0 * (2.0 ** (vib / 1200.0))

        # 2. Glottal Excitation Waveform with Harmonic Overtones
        phase = np.cumsum(2 * np.pi * f0_contour / sr)
        # Pulse-like glottal waveform with 10 harmonics
        pulse = np.zeros(n_samples)
        for h in range(1, 14):
            harm_amp = (1.0 / (h ** 0.85)) * (0.8 + 0.2 * math.cos(h))
            pulse += harm_amp * np.sin(h * phase)
            
        # 3. Articulatory Envelope (ADSR) with syllabic attack
        attack_len = int(sr * 0.04) # 40ms attack
        release_len = int(sr * 0.04)
        env = np.ones(n_samples)
        if n_samples > attack_len + release_len:
            env[:attack_len] = np.linspace(0, 1, attack_len)
            env[-release_len:] = np.linspace(1, 0, release_len)
        else:
            env = np.sin(np.pi * t_note / dur)
            
        # Add subtle breathiness / aspiration noise
        breath = np.random.randn(n_samples) * 0.03
        raw_vocal = (pulse + breath) * env
        
        # 4. Apply Vocal Formants for the syllable
        f1, f2, f3 = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS["a"])
        # Formants vary slightly with pitch
        f1_shifted = f1 * (1.0 + 0.1 * (target_f0 - 150) / 300)
        vocal_note = apply_formants(raw_vocal, sr, f1_shifted, f2, f3)
        
        # Place into vocal track
        end_sample = min(total_samples, cur_sample + n_samples)
        actual_len = end_sample - cur_sample
        vocal[cur_sample:end_sample] += vocal_note[:actual_len]
        
        # 5. Populate Ground-Truth Frame Labels
        start_frame = int(round(cur_time * 100))
        end_frame = int(round((cur_time + dur) * 100))
        end_frame = min(total_frames, end_frame)
        
        if start_frame < total_frames:
            frame_onsets[start_frame] = 1.0 # True note onset!
            
        swara_class_idx = (saptak + 1) * 12 + SWARA_TO_IDX[sw_name] # 0 to 35
        gamaka_class_idx = {"steady": 0, "kampitam": 1, "jaaru": 2, "karvai": 3}.get(gamaka, 0)
        
        for f in range(start_frame, end_frame):
            if f < total_frames:
                frame_swaras[f] = swara_class_idx
                frame_gamakas[f] = gamaka_class_idx
                frame_voicing[f] = 1.0
                sample_offset = min(n_samples - 1, int((f - start_frame) * 0.01 * sr))
                frame_f0[f] = f0_contour[sample_offset]
                
        cur_sample += n_samples
        cur_time += dur

    # Normalize vocal
    max_val = np.max(np.abs(vocal))
    if max_val > 0:
        vocal = vocal / max_val * 0.85
        
    # Mix with acoustic tanpura drone
    tanpura = generate_tanpura_drone(sa_hz, total_dur, sr=sr)
    mix = vocal + tanpura[:len(vocal)]
    mix = mix / np.max(np.abs(mix)) * 0.92
    
    return mix, frame_onsets, frame_swaras, frame_gamakas, frame_f0, frame_voicing

def generate_full_dataset(output_dir="dataset", num_samples=180, sr=16000):
    """Generates an extensive training and validation dataset of Carnatic & Indian music."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    
    print(f"Generating {num_samples} authentic Carnatic music training tracks...")
    
    raga_keys = list(RAGAS.keys())
    vowels = ["a", "i", "u", "e", "o"]
    
    for i in range(num_samples):
        raga_name = random.choice(raga_keys)
        scale = RAGAS[raga_name]
        
        # Random tonic Sa between 120 Hz and 260 Hz (covering male D3, C3, female G3, A3)
        sa_hz = random.choice([130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 196.00, 220.00, 246.94])
        sa_hz *= (1.0 + random.uniform(-0.015, 0.015)) # slight micro-tuning
        
        # Decide musical pattern type
        pattern_type = random.choice(["sarali", "janta", "dhaatu", "alapana", "varnam"])
        
        notes_seq = []
        
        if pattern_type == "sarali":
            # Ascending and descending scale
            dur_per_note = random.choice([0.25, 0.35, 0.5, 0.7])
            # Arohana
            for sw in scale:
                notes_seq.append({
                    "swara": sw, "saptak": 0, "dur": dur_per_note,
                    "gamaka": random.choice(["steady", "karvai", "kampitam"]),
                    "vowel": random.choice(vowels)
                })
            # Taar Sa
            notes_seq.append({
                "swara": "S", "saptak": 1, "dur": dur_per_note * 1.5,
                "gamaka": "karvai", "vowel": "a"
            })
            # Avarohana
            for sw in reversed(scale):
                notes_seq.append({
                    "swara": sw, "saptak": 0, "dur": dur_per_note,
                    "gamaka": random.choice(["steady", "kampitam", "jaaru"]),
                    "vowel": random.choice(vowels)
                })
        elif pattern_type == "janta":
            # Janta varisai doublets: S S R R G G M M
            dur_per_note = random.choice([0.25, 0.35, 0.45])
            for sw in scale[:5]:
                notes_seq.append({"swara": sw, "saptak": 0, "dur": dur_per_note, "gamaka": "steady", "vowel": "a"})
                notes_seq.append({"swara": sw, "saptak": 0, "dur": dur_per_note, "gamaka": "steady", "vowel": "a"})
        elif pattern_type == "dhaatu":
            # Skipping varisai: S G R M G P M D
            dur_per_note = random.choice([0.3, 0.45])
            if len(scale) >= 6:
                for idx in range(min(5, len(scale) - 2)):
                    notes_seq.append({"swara": scale[idx], "saptak": 0, "dur": dur_per_note, "gamaka": "steady", "vowel": "a"})
                    notes_seq.append({"swara": scale[idx + 2], "saptak": 0, "dur": dur_per_note, "gamaka": "jaaru", "vowel": "a"})
        elif pattern_type == "alapana":
            # Unmetered alapana with heavy kampitam and jaaru
            num_phrases = random.randint(8, 14)
            for _ in range(num_phrases):
                sw = random.choice(scale)
                saptak = random.choice([-1, 0, 0, 0, 1])
                dur = random.uniform(0.35, 1.2)
                gamaka = random.choice(["kampitam", "jaaru", "karvai", "steady"])
                notes_seq.append({
                    "swara": sw, "saptak": saptak, "dur": dur,
                    "gamaka": gamaka, "vowel": random.choice(vowels)
                })
        else: # varnam / kriti phrase
            num_notes = random.randint(10, 18)
            cur_idx = 0
            for _ in range(num_notes):
                cur_idx = max(0, min(len(scale) - 1, cur_idx + random.choice([-2, -1, 1, 1, 2])))
                notes_seq.append({
                    "swara": scale[cur_idx],
                    "saptak": random.choice([0, 0, 0, 1 if cur_idx == 0 else 0]),
                    "dur": random.choice([0.22, 0.35, 0.5, 0.8]),
                    "gamaka": random.choice(["steady", "kampitam", "jaaru"]),
                    "vowel": random.choice(vowels)
                })

        audio, onsets, swaras, gamakas, f0, voicing = synthesize_carnatic_vocal_phrase(notes_seq, sa_hz, sr=sr)
        
        filename = f"track_{i:04d}_{raga_name}_{pattern_type}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio, sr)
        
        # Save frame labels as compressed npz
        label_filename = f"track_{i:04d}_labels.npz"
        label_path = os.path.join(output_dir, label_filename)
        np.savez_compressed(
            label_path,
            onsets=onsets,
            swaras=swaras,
            gamakas=gamakas,
            f0=f0,
            voicing=voicing,
            sa_hz=sa_hz,
            raga=raga_name
        )
        
        metadata.append({
            "id": i,
            "audio_file": filename,
            "label_file": label_filename,
            "raga": raga_name,
            "pattern": pattern_type,
            "sa_hz": round(float(sa_hz), 1),
            "num_notes": len(notes_seq),
            "duration": round(len(audio) / sr, 2)
        })
        
        if (i + 1) % 30 == 0:
            print(f"Generated {i + 1}/{num_samples} tracks...")
            
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Dataset generation complete! Generated {len(metadata)} tracks in '{output_dir}'.")
    return metadata

if __name__ == "__main__":
    generate_full_dataset("dataset", num_samples=160)
