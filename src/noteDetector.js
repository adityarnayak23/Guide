/**
 * NoteScope — Note Detector Module
 * Uses Spotify's Basic Pitch (TensorFlow.js) for polyphonic note detection.
 * Falls back to a lightweight YIN-based detector if Basic Pitch fails to load.
 */

// Note name mapping
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/**
 * Convert MIDI number to note name + octave
 * @param {number} midiNote - MIDI note number (0-127)
 * @returns {{ name: string, octave: number, fullName: string }}
 */
export function midiToNote(midiNote) {
  const octave = Math.floor(midiNote / 12) - 1;
  const noteIndex = midiNote % 12;
  const name = NOTE_NAMES[noteIndex];
  return {
    name,
    octave,
    fullName: `${name}${octave}`,
    midi: midiNote,
  };
}

/**
 * Convert frequency (Hz) to MIDI note number
 * @param {number} freq
 * @returns {number}
 */
export function freqToMidi(freq) {
  return Math.round(12 * Math.log2(freq / 440) + 69);
}

/**
 * Convert MIDI note number to frequency (Hz)
 * @param {number} midi
 * @returns {number}
 */
export function midiToFreq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/**
 * Resample any AudioBuffer to 22050 Hz Mono, as strictly required by Basic Pitch.
 * @param {AudioBuffer} audioBuffer
 * @returns {Promise<AudioBuffer>}
 */
export async function resampleTo22050Mono(audioBuffer) {
  if (audioBuffer.sampleRate === 22050 && audioBuffer.numberOfChannels === 1) {
    return audioBuffer;
  }
  const targetSampleRate = 22050;
  const targetLength = Math.max(1, Math.round(audioBuffer.duration * targetSampleRate));
  const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(
    1,
    targetLength,
    targetSampleRate
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start(0);
  return await offlineCtx.startRendering();
}

/**
 * Detect notes using Spotify's Basic Pitch.
 * @param {AudioBuffer} audioBuffer - The audio to analyze
 * @param {Function} onProgress - Progress callback (0-1)
 * @returns {Promise<Array<{ startTime: number, endTime: number, duration: number, midi: number, amplitude: number, name: string, octave: number, fullName: string }>>}
 */
export async function detectNotes(audioBuffer, onProgress = () => {}) {
  onProgress(0.05);

  try {
    // Dynamically import Basic Pitch
    const basicPitch = await import('@spotify/basic-pitch');
    onProgress(0.12);

    // Ensure audio is resampled to 22050Hz Mono
    const monoResampled = await resampleTo22050Mono(audioBuffer);
    onProgress(0.20);

    // Instantiate model with local static path
    const modelUrl = '/model/model.json';
    const bp = new basicPitch.BasicPitch(modelUrl);

    // Run Basic Pitch inference
    const frames = [];
    const onsets = [];
    const contours = [];

    await bp.evaluateModel(
      monoResampled,
      (f, o, c) => {
        frames.push(...f);
        onsets.push(...o);
        contours.push(...c);
      },
      (percent) => {
        onProgress(0.20 + percent * 0.65);
      }
    );

    onProgress(0.88);

    // Extract polyphonic notes
    const noteEvents = basicPitch.outputToNotesPoly(
      frames,
      onsets,
      0.35,  // onset threshold
      0.22,  // frame threshold
      5,     // min note length (frames ~ 58ms)
      true,  // infer onsets
    );

    const bentNotes = basicPitch.addPitchBendsToNoteEvents(contours, noteEvents);
    const rawNotes = basicPitch.noteFramesToTime(bentNotes);

    onProgress(0.96);

    // Convert to unified note objects
    const notes = rawNotes.map(note => {
      const noteInfo = midiToNote(note.pitchMidi);
      return {
        startTime: note.startTimeSeconds,
        endTime: note.startTimeSeconds + note.durationSeconds,
        duration: note.durationSeconds,
        midi: note.pitchMidi,
        amplitude: note.amplitude,
        pitchBends: note.pitchBends,
        ...noteInfo,
      };
    });

    // Sort by start time
    notes.sort((a, b) => a.startTime - b.startTime);

    onProgress(1.0);
    return notes;
  } catch (error) {
    console.warn('Basic Pitch failed, falling back to YIN detector:', error);
    onProgress(0.2);

    // Fallback to YIN-based detection
    return detectNotesYIN(audioBuffer, onProgress);
  }
}

/**
 * Fallback: Simple YIN-based monophonic pitch detection.
 * Not as accurate as Basic Pitch but works without TensorFlow.js.
 */
async function detectNotesYIN(audioBuffer, onProgress) {
  const sampleRate = audioBuffer.sampleRate;
  const monoData = getMonoAudioData(audioBuffer);
  const frameSize = 2048;
  const hopSize = 512;
  const totalFrames = Math.floor((monoData.length - frameSize) / hopSize);

  const rawPitches = [];

  for (let i = 0; i < totalFrames; i++) {
    const start = i * hopSize;
    const frame = monoData.slice(start, start + frameSize);
    const freq = yinPitchDetect(frame, sampleRate);
    const time = start / sampleRate;

    if (freq > 60 && freq < 2000) {
      const midi = freqToMidi(freq);
      rawPitches.push({ time, midi, freq });
    }

    if (i % 100 === 0) {
      onProgress(0.2 + (i / totalFrames) * 0.7);
    }
  }

  // Group consecutive same-note pitches into note events
  const notes = [];
  let currentNote = null;

  for (const pitch of rawPitches) {
    if (currentNote && currentNote.midi === pitch.midi &&
        pitch.time - currentNote.lastTime < 0.1) {
      currentNote.endTime = pitch.time;
      currentNote.lastTime = pitch.time;
      currentNote.count++;
    } else {
      if (currentNote && currentNote.count >= 3) {
        const noteInfo = midiToNote(currentNote.midi);
        notes.push({
          startTime: currentNote.startTime,
          endTime: currentNote.endTime,
          duration: currentNote.endTime - currentNote.startTime,
          midi: currentNote.midi,
          amplitude: 0.8,
          ...noteInfo,
        });
      }
      currentNote = {
        midi: pitch.midi,
        startTime: pitch.time,
        endTime: pitch.time,
        lastTime: pitch.time,
        count: 1,
      };
    }
  }

  // Don't forget the last note
  if (currentNote && currentNote.count >= 3) {
    const noteInfo = midiToNote(currentNote.midi);
    notes.push({
      startTime: currentNote.startTime,
      endTime: currentNote.endTime,
      duration: currentNote.endTime - currentNote.startTime,
      midi: currentNote.midi,
      amplitude: 0.8,
      ...noteInfo,
    });
  }

  // Filter out very short notes (likely noise)
  const filteredNotes = notes.filter(n => n.duration >= 0.05);

  onProgress(1.0);
  return filteredNotes;
}

/**
 * YIN pitch detection algorithm.
 * @param {Float32Array} buffer - Audio frame
 * @param {number} sampleRate
 * @returns {number} - Detected frequency in Hz, or -1 if no pitch found
 */
function yinPitchDetect(buffer, sampleRate) {
  const threshold = 0.15;
  const halfLen = Math.floor(buffer.length / 2);
  const yinBuffer = new Float32Array(halfLen);

  // Step 1: Difference function
  for (let tau = 0; tau < halfLen; tau++) {
    let sum = 0;
    for (let i = 0; i < halfLen; i++) {
      const delta = buffer[i] - buffer[i + tau];
      sum += delta * delta;
    }
    yinBuffer[tau] = sum;
  }

  // Step 2: Cumulative mean normalized difference
  yinBuffer[0] = 1;
  let runningSum = 0;
  for (let tau = 1; tau < halfLen; tau++) {
    runningSum += yinBuffer[tau];
    yinBuffer[tau] *= tau / runningSum;
  }

  // Step 3: Absolute threshold
  let tau = 2;
  while (tau < halfLen) {
    if (yinBuffer[tau] < threshold) {
      while (tau + 1 < halfLen && yinBuffer[tau + 1] < yinBuffer[tau]) {
        tau++;
      }
      break;
    }
    tau++;
  }

  if (tau === halfLen) return -1;

  // Step 4: Parabolic interpolation
  let betterTau;
  const x0 = tau < 1 ? tau : tau - 1;
  const x2 = tau + 1 < halfLen ? tau + 1 : tau;

  if (x0 === tau) {
    betterTau = yinBuffer[tau] <= yinBuffer[x2] ? tau : x2;
  } else if (x2 === tau) {
    betterTau = yinBuffer[tau] <= yinBuffer[x0] ? tau : x0;
  } else {
    const s0 = yinBuffer[x0];
    const s1 = yinBuffer[tau];
    const s2 = yinBuffer[x2];
    betterTau = tau + (s2 - s0) / (2 * (2 * s1 - s2 - s0));
  }

  return sampleRate / betterTau;
}

/**
 * Extract mono audio data from an AudioBuffer.
 * @param {AudioBuffer} audioBuffer
 * @returns {Float32Array}
 */
function getMonoAudioData(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) {
    return audioBuffer.getChannelData(0);
  }

  // Mix down to mono
  const length = audioBuffer.length;
  const mono = new Float32Array(length);
  const numChannels = audioBuffer.numberOfChannels;

  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i] / numChannels;
    }
  }

  return mono;
}

/**
 * Get summary statistics from detected notes.
 */
export function getNoteSummary(notes) {
  if (!notes || notes.length === 0) {
    return {
      totalNotes: 0,
      uniqueNotes: 0,
      lowestNote: null,
      highestNote: null,
      mostCommonNote: null,
      duration: 0,
      noteDistribution: {},
    };
  }

  const noteNames = notes.map(n => n.fullName);
  const uniqueNotes = [...new Set(noteNames)];
  const midiValues = notes.map(n => n.midi);
  const lowestMidi = Math.min(...midiValues);
  const highestMidi = Math.max(...midiValues);

  // Note frequency distribution
  const distribution = {};
  for (const name of noteNames) {
    distribution[name] = (distribution[name] || 0) + 1;
  }

  const mostCommon = Object.entries(distribution)
    .sort((a, b) => b[1] - a[1])[0];

  const lastNote = notes[notes.length - 1];
  const duration = lastNote ? lastNote.endTime : 0;

  return {
    totalNotes: notes.length,
    uniqueNotes: uniqueNotes.length,
    lowestNote: midiToNote(lowestMidi),
    highestNote: midiToNote(highestMidi),
    mostCommonNote: mostCommon ? mostCommon[0] : null,
    duration,
    noteDistribution: distribution,
  };
}

/**
 * Export detected notes to a standard MIDI file (.mid) and trigger browser download.
 * @param {Array} notes
 * @param {string} filename
 */
export async function exportNotesToMidi(notes, filename = 'detected_notes.mid') {
  if (!notes || notes.length === 0) return false;
  try {
    const { Midi } = await import('@tonejs/midi');
    const midi = new Midi();
    const track = midi.addTrack();
    notes.forEach(note => {
      track.addNote({
        midi: note.midi,
        time: Math.max(0, note.startTime),
        duration: Math.max(0.01, note.duration),
        velocity: Math.min(1, Math.max(0.1, note.amplitude || 0.8)),
      });
    });

    const midiArray = midi.toArray();
    const blob = new Blob([midiArray], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return true;
  } catch (err) {
    console.error('Failed to export MIDI:', err);
    return false;
  }
}

/**
 * Export detected notes to a structured JSON file and trigger browser download.
 * @param {Array} notes
 * @param {string} filename
 */
export function exportNotesToJson(notes, filename = 'detected_notes.json') {
  if (!notes || notes.length === 0) return false;
  try {
    const data = JSON.stringify(notes, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return true;
  } catch (err) {
    console.error('Failed to export JSON:', err);
    return false;
  }
}
