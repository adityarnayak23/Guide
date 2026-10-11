#!/usr/bin/env python3
"""
Generate Master Studio Acoustic Tanpura Audio Assets for NoteScope
Authentic Physical Modeling of Miraj and Tanjore Tanpura
"""

import os
import numpy as np
import scipy.signal as signal
import soundfile as sf

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public', 'audio', 'tanpura')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 44100
BASE_SA = 146.83 # D3 (Classical standard Adhara Shadja)

def synthesize_acoustic_string(
    f0,
    duration_s=6.5,
    is_kharaj=False,
    is_jodi2=False,
    timbre='miraj',
    sr=SR
):
    N = int(sr * duration_s)
    t = np.linspace(0, duration_s, N, endpoint=False)
    
    # Inharmonicity factor (curved bone bridge & string stiffness)
    B = 0.00017 if not is_kharaj else 0.00028
    
    actual_f0 = f0
    if is_jodi2:
        # Acoustic thread (jiva) detune +1.3 cents
        actual_f0 = f0 * (2.0 ** (1.3 / 1200.0))
        
    num_partials = 48 if not is_kharaj else 34
    
    # Attack tension relax pitch glide (18 cents over 130ms)
    cents_glide = 16.0 if not is_kharaj else 22.0
    glide_factor = 2.0 ** ((cents_glide * np.exp(-t / 0.13)) / 1200.0)
    
    # Jiva thread rolling flutter
    flutter_freq = 3.15
    flutter_depth = 0.003
    flutter = 1.0 + flutter_depth * np.sin(2.0 * np.pi * flutter_freq * t + np.random.uniform(0, 2*np.pi))
    
    base_trajectory = actual_f0 * glide_factor * flutter
    signal_out = np.zeros(N, dtype=np.float64)
    
    for k in range(1, num_partials + 1):
        partial_mult = k * np.sqrt(1.0 + B * (k ** 2))
        fk_trajectory = base_trajectory * partial_mult
        
        if np.max(fk_trajectory) >= sr * 0.48:
            break
            
        # Harmonic spectral weighting
        if is_kharaj:
            amp = 1.0 if k == 1 else (0.85 / (k ** 1.05))
            if k <= 6:
                amp *= 1.4
        elif timbre == 'miraj':
            formant = np.exp(-0.5 * (((k - 12.5) / 5.8) ** 2)) * 3.3 + 0.35
            amp = (1.0 / (k ** 0.88)) * formant
        else: # tanjore
            formant = np.exp(-0.5 * (((k - 16.5) / 6.2) ** 2)) * 3.6 + 0.28
            amp = (1.0 / (k ** 0.82)) * formant
            
        # Dynamic Javari blooming envelopes
        if k <= 2:
            att = 1.0 - np.exp(-t / 0.025)
            dec = np.exp(-t / (5.8 if not is_kharaj else 7.5))
            env = att * dec
        elif k <= 6:
            att = (1.0 - np.exp(-t / 0.07)) ** 1.6
            dec = np.exp(-t / 4.8)
            env = att * dec
        else:
            rise_time = 0.16 + (k / num_partials) * 0.22
            decay_time = 3.6 / np.sqrt(k * 0.35)
            att = (1.0 - np.exp(-t / rise_time)) ** 2.2
            dec = np.exp(-t / decay_time)
            amp_flutter = 1.0 + 0.16 * np.sin(2.0 * np.pi * (flutter_freq * 1.4) * t + k)
            env = att * dec * amp_flutter
            
        phase = 2.0 * np.pi * np.cumsum(fk_trajectory) / sr + np.random.uniform(0, 2 * np.pi)
        signal_out += amp * env * np.sin(phase)
        
    # Soft fleshy pluck transient
    pluck_len = int(sr * 0.035)
    noise = np.random.normal(0, 0.2, pluck_len)
    b_lp, a_lp = signal.butter(2, min(1600.0, actual_f0 * 4.0) / (sr / 2.0), btype='low')
    fleshy_pluck = signal.lfilter(b_lp, a_lp, noise) * ((1.0 - np.linspace(0, 1, pluck_len)) ** 2.5)
    signal_out[:pluck_len] += fleshy_pluck * (0.35 if not is_kharaj else 0.55)
    
    # Asymmetric bridge wave-shaping saturation
    norm_val = np.max(np.abs(signal_out)) + 1e-6
    scaled = signal_out / norm_val
    pos_mask = scaled > 0
    neg_mask = ~pos_mask
    shaped = np.zeros_like(scaled)
    shaped[pos_mask] = np.tanh(scaled[pos_mask] * 1.55) * 1.12
    shaped[neg_mask] = np.tanh(scaled[neg_mask] * 1.1) * 0.95
    signal_out = shaped * norm_val
    
    # Gourd body filter cascade
    b_h, a_h = signal.iirpeak(115.0, 3.2, sr)
    signal_out += 0.32 * signal.lfilter(b_h, a_h, signal_out)
    b_w, a_w = signal.iirpeak(235.0, 2.6, sr)
    signal_out += 0.28 * signal.lfilter(b_w, a_w, signal_out)
    b_c, a_c = signal.iirpeak(460.0, 2.8, sr)
    signal_out += 0.22 * signal.lfilter(b_c, a_c, signal_out)
    b_b, a_b = signal.iirpeak(1480.0, 3.4, sr)
    signal_out += 0.42 * signal.lfilter(b_b, a_b, signal_out)
    b_s, a_s = signal.iirpeak(2800.0, 3.0, sr)
    signal_out += 0.26 * signal.lfilter(b_s, a_s, signal_out)
    
    max_amp = np.max(np.abs(signal_out)) + 1e-7
    return signal_out / max_amp

def render_stereo_string(s_mono, pan=0.0):
    # pan in [-1.0, 1.0]
    angle = (pan + 1.0) * (np.pi / 4.0)
    left = s_mono * np.cos(angle)
    right = s_mono * np.sin(angle)
    return np.column_stack([left, right])

def generate_all_assets():
    print(f"Generating master Tanpura assets in {OUTPUT_DIR}...")
    
    # Base pitches for D3 (146.83 Hz)
    pa_hz = BASE_SA * 1.5           # 220.25 Hz
    ma_hz = BASE_SA * (4.0 / 3.0)   # 195.77 Hz
    ni_hz = BASE_SA * (15.0 / 8.0)  # 275.31 Hz
    jodi_hz = BASE_SA * 2.0         # 293.66 Hz
    kharaj_hz = BASE_SA * 1.0       # 146.83 Hz
    
    # 1. Individual strings
    print("Synthesizing String 1: Panchama (Pa)...")
    s_pa = synthesize_acoustic_string(pa_hz, duration_s=7.0, is_kharaj=False)
    sf.write(os.path.join(OUTPUT_DIR, 'string_pa.wav'), render_stereo_string(s_pa, pan=-0.22), SR, subtype='PCM_16')
    
    print("Synthesizing String 1: Madhyama (Ma)...")
    s_ma = synthesize_acoustic_string(ma_hz, duration_s=7.0, is_kharaj=False)
    sf.write(os.path.join(OUTPUT_DIR, 'string_ma.wav'), render_stereo_string(s_ma, pan=-0.22), SR, subtype='PCM_16')
    
    print("Synthesizing String 1: Nishadha (Ni)...")
    s_ni = synthesize_acoustic_string(ni_hz, duration_s=7.0, is_kharaj=False)
    sf.write(os.path.join(OUTPUT_DIR, 'string_ni.wav'), render_stereo_string(s_ni, pan=-0.22), SR, subtype='PCM_16')
    
    print("Synthesizing String 2: Jodi Sa 1...")
    s_jodi1 = synthesize_acoustic_string(jodi_hz, duration_s=7.0, is_kharaj=False)
    sf.write(os.path.join(OUTPUT_DIR, 'string_jodi1.wav'), render_stereo_string(s_jodi1, pan=-0.08), SR, subtype='PCM_16')
    
    print("Synthesizing String 3: Jodi Sa 2 (with Jiva detune)...")
    s_jodi2 = synthesize_acoustic_string(jodi_hz, duration_s=7.0, is_kharaj=False, is_jodi2=True)
    sf.write(os.path.join(OUTPUT_DIR, 'string_jodi2.wav'), render_stereo_string(s_jodi2, pan=0.08), SR, subtype='PCM_16')
    
    print("Synthesizing String 4: Kharaj Sa...")
    s_kharaj = synthesize_acoustic_string(kharaj_hz, duration_s=8.5, is_kharaj=True)
    sf.write(os.path.join(OUTPUT_DIR, 'string_kharaj.wav'), render_stereo_string(s_kharaj, pan=0.0), SR, subtype='PCM_16')
    
    # 2. Continuous full 4-string acoustic cycles (Pa, Ma, Ni)
    for first_type, s_first, pan1 in [('pa', s_pa, -0.22), ('ma', s_ma, -0.22), ('ni', s_ni, -0.22)]:
        print(f"Assembling seamless loop cycle for {first_type.upper()}...")
        pluck_interval = 1.15
        kharaj_pause = 1.80
        cycle_len = 3 * pluck_interval + kharaj_pause # 5.25 sec
        total_len = cycle_len * 3 # 3 full cycles (15.75 sec)
        total_samples = int(SR * total_len)
        
        left = np.zeros(total_samples, dtype=np.float64)
        right = np.zeros(total_samples, dtype=np.float64)
        
        cycle_strings = [
            (s_first, pan1, 0.92),
            (s_jodi1, -0.08, 0.88),
            (s_jodi2, 0.08, 0.88),
            (s_kharaj, 0.0, 1.10),
        ]
        
        for c_idx in range(3):
            c_start = c_idx * cycle_len
            for s_idx, (s_data, pan, vol) in enumerate(cycle_strings):
                t_p = c_start + s_idx * pluck_interval
                p_start = int(t_p * SR)
                p_end = min(total_samples, p_start + len(s_data))
                chunk_len = p_end - p_start
                if chunk_len > 0:
                    angle = (pan + 1.0) * (np.pi / 4.0)
                    left[p_start:p_end] += s_data[:chunk_len] * np.cos(angle) * vol
                    right[p_start:p_end] += s_data[:chunk_len] * np.sin(angle) * vol
                    
        cycle_stereo = np.column_stack([left, right])
        # Soft limiter normalization
        max_p = np.max(np.abs(cycle_stereo))
        cycle_stereo = (cycle_stereo / (max_p + 1e-6)) * 0.92
        
        out_name = f'tanpura_cycle_{first_type}.wav'
        sf.write(os.path.join(OUTPUT_DIR, out_name), cycle_stereo, SR, subtype='PCM_16')
        print(f"Saved {out_name} (15.75s, 44.1kHz stereo)")

    print("All master acoustic Tanpura assets generated successfully!")

if __name__ == '__main__':
    generate_all_assets()
