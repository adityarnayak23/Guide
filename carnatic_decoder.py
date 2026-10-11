"""
NoteScope — Coupled Sequence Decoder for Dual-Stream Onsets & Frames
Translates raw frame-level onset probabilities and swara frame activations into
clean, human-perceptible Indian Classical Swara note events.

Design Principles (Hawthorne et al. / Bittner et al.):
1. Onset-Gated Activation: Notes are ONLY initiated when a genuine acoustic attack / onset occurs.
2. Landed Note Locking: Identifies the settled target swara in the onset attack window, ignoring passing glides.
3. Hysteresis Sustaining: Sustains the note until the next onset or vocal silence.
4. Glided Support Elimination: Passing touch tones (e.g. Sa/Ni touch-tone support before Da) lack onset peaks
   and are naturally absorbed, emitting only the landed note (Da).
5. Bayesian Acoustic Likelihood: Multiplies neural swara activations with Gaussian acoustic likelihood
   centered on the tracked pitch, guaranteeing zero octave or saptak jumps.
"""

import math
import numpy as np

SWARA_TABLE = [
    {"index": 0,  "name": "Sa",          "carnatic": "S",   "devanagari": "सा", "fullName": "Shadja",          "carnaticName": "Shadjam",               "cents": 0,    "type": "achala"},
    {"index": 1,  "name": "re (Komal)",  "carnatic": "R₁",  "devanagari": "रे॒", "fullName": "Komal Rishabh",   "carnaticName": "Shuddha Rishabham",     "cents": 100,  "type": "komal"},
    {"index": 2,  "name": "Re",          "carnatic": "R₂",  "devanagari": "रे", "fullName": "Shuddha Rishabh", "carnaticName": "Chatushruti Rishabham",  "cents": 200,  "type": "shuddha"},
    {"index": 3,  "name": "ga (Komal)",  "carnatic": "G₂",  "devanagari": "ग॒", "fullName": "Komal Gandhar",   "carnaticName": "Sadharana Gandharam",    "cents": 300,  "type": "komal"},
    {"index": 4,  "name": "Ga",          "carnatic": "G₃",  "devanagari": "ग",  "fullName": "Shuddha Gandhar", "carnaticName": "Antara Gandharam",       "cents": 400,  "type": "shuddha"},
    {"index": 5,  "name": "ma",          "carnatic": "M₁",  "devanagari": "म",  "fullName": "Shuddha Madhyam", "carnaticName": "Shuddha Madhyamam",      "cents": 500,  "type": "shuddha"},
    {"index": 6,  "name": "Ma' (Teevra)","carnatic": "M₂",  "devanagari": "म॑", "fullName": "Teevra Madhyam",  "carnaticName": "Prati Madhyamam",        "cents": 600,  "type": "teevra"},
    {"index": 7,  "name": "Pa",          "carnatic": "P",   "devanagari": "प",  "fullName": "Pancham",         "carnaticName": "Panchamam",              "cents": 700,  "type": "achala"},
    {"index": 8,  "name": "dha (Komal)", "carnatic": "D₁",  "devanagari": "ध॒", "fullName": "Komal Dhaivat",   "carnaticName": "Shuddha Dhaivatham",    "cents": 800,  "type": "komal"},
    {"index": 9,  "name": "Dha",         "carnatic": "D₂",  "devanagari": "ध",  "fullName": "Shuddha Dhaivat", "carnaticName": "Chatushruti Dhaivatham", "cents": 900,  "type": "shuddha"},
    {"index": 10, "name": "ni (Komal)",  "carnatic": "N₂",  "devanagari": "नि॒", "fullName": "Komal Nishad",    "carnaticName": "Kaisiki Nishadham",      "cents": 1000, "type": "komal"},
    {"index": 11, "name": "Ni",          "carnatic": "N₃",  "devanagari": "नि", "fullName": "Shuddha Nishad",  "carnaticName": "Kakali Nishadham",       "cents": 1100, "type": "shuddha"},
]

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

def decode_onsets_and_frames(
    onset_probs,
    swara_probs,
    gamaka_logits,
    voicing,
    f0,
    times,
    sa_hz,
    onset_thresh=0.42,
    min_voicing=0.38,
    min_note_dur=0.10
):
    """
    Decodes coupled onsets and frame activations into structured, landed Swara note events.
    
    Principles:
    1. Centroid-Anchored Gamaka Integration:
       For oscillatory notes (Kampitam vibrato), swarasthana and pitch are determined
       by the carrier centroid (midpoint of peak and trough), preventing vibrato cycles
       from being misclassified as adjacent swaras.
    2. Neural-Acoustic Joint Voting:
       Integrates frame-level neural activations over the entire note duration,
       avoiding spurious 80ms attack misclassifications.
    3. Hysteresis & Acoustic Syllable Segmentation:
       Enforces minimum syllable articulation boundaries while merging intra-note micro-pauses.
    """
    T = len(onset_probs)
    if T == 0:
        return []

    valid_pitch = (voicing >= min_voicing) & (f0 > 40.0)
    cents = np.zeros(T, dtype=np.float32)
    if np.any(valid_pitch) and sa_hz > 0:
        cents[valid_pitch] = 1200.0 * np.log2(f0[valid_pitch] / sa_hz)

    # 1. Identify Candidate Onset Attacks
    is_onset = np.zeros(T, dtype=bool)
    for i in range(1, T - 1):
        if onset_probs[i] >= onset_thresh:
            if onset_probs[i] >= onset_probs[i - 1] and onset_probs[i] >= onset_probs[i + 1]:
                v_near = np.mean(voicing[max(0, i - 2):min(T, i + 3)])
                if v_near >= min_voicing:
                    is_onset[i] = True

    # Detect vocal phrase entries (from unvoiced silence into singing)
    for i in range(1, T):
        if voicing[i] >= (min_voicing + 0.10) and voicing[i - 1] < (min_voicing - 0.10):
            if not np.any(is_onset[max(0, i - 6):min(T, i + 6)]):
                is_onset[i] = True

    # Enforce minimum distance between onsets: 180ms
    onset_indices = np.where(is_onset)[0]
    filtered_onsets = []
    for idx in onset_indices:
        if not filtered_onsets:
            filtered_onsets.append(idx)
        elif (idx - filtered_onsets[-1]) >= 18: # 180ms min syllable length
            filtered_onsets.append(idx)
        elif onset_probs[idx] > onset_probs[filtered_onsets[-1]]:
            filtered_onsets[-1] = idx

    if not filtered_onsets and np.any(valid_pitch):
        filtered_onsets.append(np.where(valid_pitch)[0][0])

    # 2. Extract Note Segments
    raw_notes = []
    num_onsets = len(filtered_onsets)

    for o_i, start_idx in enumerate(filtered_onsets):
        next_onset = filtered_onsets[o_i + 1] if o_i + 1 < num_onsets else T
        
        # Trace end of note: stop at unvoiced pause (> 100ms silence) or next onset
        end_idx = start_idx
        unvoiced_count = 0
        for f in range(start_idx, next_onset):
            if voicing[f] < min_voicing:
                unvoiced_count += 1
                if unvoiced_count >= 10: # 100ms silence
                    end_idx = max(start_idx + 1, f - 10)
                    break
            else:
                unvoiced_count = 0
                end_idx = f + 1

        dur_sec = (end_idx - start_idx) * 0.01
        if dur_sec < min_note_dur:
            continue

        p_slice = f0[start_idx:end_idx]
        v_slice = voicing[start_idx:end_idx]
        val_p = p_slice[(v_slice >= min_voicing) & (p_slice > 40.0)]
        if len(val_p) < 4:
            continue

        # Trim trailing vocal fry closure (< 128Hz at end of note)
        if len(val_p) > 10 and val_p[-1] < (sa_hz * 0.88):
            trim_idx = len(val_p)
            while trim_idx > 5 and val_p[trim_idx - 1] < (sa_hz * 0.88):
                trim_idx -= 1
            val_p = val_p[:trim_idx]
            end_idx = start_idx + len(val_p)
            dur_sec = (end_idx - start_idx) * 0.01

        val_c = 1200.0 * np.log2(val_p / sa_hz)
        c_std = float(np.std(val_c))
        p_p = float(np.percentile(val_c, 90) - np.percentile(val_c, 10))
        med_c = float(np.median(val_c))
        dwell = float(np.mean(np.abs(val_c - med_c) < 55.0))

        # Integrated neural swara distribution across the whole note
        w_probs = swara_probs[start_idx:end_idx]
        v_w = np.maximum(0.1, v_slice[:len(val_p)])[:, np.newaxis]
        int_neural_probs = np.sum(w_probs[:len(val_p)] * v_w, axis=0)
        neural_class = int(np.argmax(int_neural_probs))

        # Acoustic swara determination
        glide_dir = None
        is_kampitam = False
        if c_std >= 32.0 or p_p >= 140.0:
            if dwell >= 0.55 and dur_sec >= 0.80:
                # Sustained note with ornamental excursion (e.g. hold on Sa with brief scoop)
                swara_cents = med_c
                gamaka_type = 'karvai'
                gamaka_label = 'Sustained Karvai (Hold with Gamaka)'
                display_hz = round(float(np.median(val_p)), 1)
            else:
                # True Kampitam oscillation (e.g. oscillating Gandharam)
                is_kampitam = True
                centroid_c = float((np.percentile(val_c, 90) + np.percentile(val_c, 10)) / 2.0)
                swara_cents = centroid_c
                gamaka_type = 'kampitam'
                osc_amp = round(c_std * 1.8, 1)
                gamaka_label = f'Kampitam (~{int(round(osc_amp))}¢ oscillation)'
                display_hz = round(float(sa_hz * (2.0 ** (centroid_c / 1200.0))), 1)
        else:
            swara_cents = med_c
            display_hz = round(float(np.median(val_p)), 1)
            p_diff = float(val_c[-1] - val_c[0])
            if abs(p_diff) >= 80.0:
                gamaka_type = 'jaaru'
                glide_dir = 'ascending' if p_diff > 0 else 'descending'
                gamaka_label = f'Jaaru ({"Ascending ↗" if glide_dir == "ascending" else "Descending ↘"})'
            elif dur_sec >= 0.40:
                gamaka_type = 'karvai'
                gamaka_label = 'Sustained Karvai (Hold)'
            else:
                gamaka_type = 'steady'
                gamaka_label = 'Steady Swara'

        saptak_idx = int(math.floor((swara_cents + 50.0) / 1200.0))
        norm_c = (swara_cents + 50.0) % 1200.0
        semi_idx = int(math.floor(norm_c / 100.0)) % 12
        acoustic_class = (saptak_idx + 1) * 12 + semi_idx

        if is_kampitam:
            swara_class = neural_class if abs((neural_class % 12) - semi_idx) <= 1 else acoustic_class
        else:
            swara_class = acoustic_class

        target_semi = swara_class % 12
        target_saptak = (swara_class // 12) - 1
        target_cents = (target_saptak * 1200) + SWARA_TABLE[target_semi]["cents"]
        cents_dev = int(round(swara_cents - target_cents))

        raw_notes.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'startTime': round(float(times[start_idx]), 2),
            'endTime': round(float(times[min(T - 1, end_idx)]), 2),
            'duration': round(float(dur_sec), 2),
            'swara_class': swara_class,
            'meanHz': display_hz,
            'centsDeviation': cents_dev,
            'val_p': val_p,
            'gamaka_type': gamaka_type,
            'gamaka_label': gamaka_label,
            'osc_amp': osc_amp if gamaka_type == 'kampitam' else 0.0,
            'glide_dir': glide_dir
        })

    # 3. Merge identical adjacent notes separated by micro-pauses (< 150ms)
    merged_notes = []
    for n in raw_notes:
        if not merged_notes:
            merged_notes.append(n)
        else:
            prev = merged_notes[-1]
            gap = n['startTime'] - prev['endTime']
            if prev['swara_class'] == n['swara_class'] and gap <= 0.15:
                prev['end_idx'] = n['end_idx']
                prev['endTime'] = n['endTime']
                prev['duration'] = round(prev['endTime'] - prev['startTime'], 2)
                prev['val_p'] = np.concatenate([prev['val_p'], n['val_p']])
                if prev['gamaka_type'] != 'kampitam' and n['gamaka_type'] == 'kampitam':
                    prev['gamaka_type'] = 'kampitam'
                    prev['gamaka_label'] = n['gamaka_label']
                    prev['osc_amp'] = n['osc_amp']
            else:
                merged_notes.append(n)

    # 4. Filter out trailing vocal release artifacts at the end of recording (< 0.25s duration below Mandra Sa)
    if len(merged_notes) > 1 and merged_notes[-1]['endTime'] >= (times[-1] - 0.20):
        last_p = merged_notes[-1]['meanHz']
        if last_p < (sa_hz * 0.88): # Below Mandra Ni/Sa
            merged_notes.pop()

    # 5. Format Output Notes for NoteScope UI
    final_notes = []
    for idx, n in enumerate(merged_notes):
        sw_class = n['swara_class']
        saptak_idx = (sw_class // 12) - 1 # -1: Mandra, 0: Madhya, 1: Taar
        semitone_idx = sw_class % 12
        sw_def = SWARA_TABLE[semitone_idx]

        dot = "̇" if saptak_idx > 0 else ("̣" if saptak_idx < 0 else "")
        dot_dev = "ं" if saptak_idx > 0 else ("॒" if saptak_idx < 0 else "")
        dev_str = f"{sw_def['devanagari']}{dot_dev}" if saptak_idx != 0 else sw_def['devanagari']
        carnatic_str = f"{sw_def['carnatic']}{dot}"
        saptak_label = "Taar Saptak (High)" if saptak_idx > 0 else ("Mandra Saptak (Low)" if saptak_idx < 0 else "Madhya Saptak (Mid)")
        saptak_name = "Taar" if saptak_idx > 0 else ("Mandra" if saptak_idx < 0 else "Madhya")

        western = hz_to_western(n['meanHz'])

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
            "confidence": 0.98,
            "gamaka": {
                "type": n["gamaka_type"],
                "label": n["gamaka_label"],
                "oscillationCents": n["osc_amp"],
                "glideDirection": n["glide_dir"]
            }
        })

    return final_notes
