/**
 * NoteScope — Vocal Pitch Tracker & Auto-Tonic (Sa) Indian Music Engine
 * 
 * Specialized for singing/vocals:
 * 1. High-resolution F0 pitch tracking optimized for human vocal range (70 Hz - 900 Hz).
 * 2. Automatic Tonic (Sa) Identification from pitch class distribution (no pre-set default).
 * 3. 12 Indian Swaras mapping (Sa, re, Re, ga, Ga, ma, Ma', Pa, dha, Dha, ni, Ni)
 *    across 3 Saptaks (Mandra, Madhya, Taar).
 * 4. Continuous vocal pitch curve (Hz) capturing meend (slides), gamaks, and intonation.
 */

// 12 Indian Swaras definition (semitones relative to Sa: 0 to 11)
export const SWARA_DEFINITIONS = [
  { index: 0,  symbol: 'S',  name: 'Sa',          fullName: 'Shadja',           type: 'achala',  cents: 0 },
  { index: 1,  symbol: 'r',  name: 're (Komal)',  fullName: 'Komal Rishabh',    type: 'komal',   cents: 100 },
  { index: 2,  symbol: 'R',  name: 'Re',          fullName: 'Shuddha Rishabh',  type: 'shuddha', cents: 200 },
  { index: 3,  symbol: 'g',  name: 'ga (Komal)',  fullName: 'Komal Gandhar',    type: 'komal',   cents: 300 },
  { index: 4,  symbol: 'G',  name: 'Ga',          fullName: 'Shuddha Gandhar',  type: 'shuddha', cents: 400 },
  { index: 5,  symbol: 'm',  name: 'ma',          fullName: 'Shuddha Madhyam',  type: 'shuddha', cents: 500 },
  { index: 6,  symbol: 'M',  name: "Ma' (Teevra)",fullName: 'Teevra Madhyam',   type: 'teevra',  cents: 600 },
  { index: 7,  symbol: 'P',  name: 'Pa',          fullName: 'Pancham',          type: 'achala',  cents: 700 },
  { index: 8,  symbol: 'd',  name: 'dha (Komal)', fullName: 'Komal Dhaivat',    type: 'komal',   cents: 800 },
  { index: 9,  symbol: 'D',  name: 'Dha',         fullName: 'Shuddha Dhaivat',  type: 'shuddha', cents: 900 },
  { index: 10, symbol: 'n',  name: 'ni (Komal)',  fullName: 'Komal Nishad',     type: 'komal',   cents: 1000 },
  { index: 11, symbol: 'N',  name: 'Ni',          fullName: 'Shuddha Nishad',   type: 'shuddha', cents: 1100 },
];

const WESTERN_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/**
 * Convert frequency (Hz) to Western note representation for reference.
 * @param {number} freq - Frequency in Hz
 * @returns {{ name: string, octave: number, centsOff: number, noteStr: string }}
 */
export function hzToWesternNote(freq) {
  if (!freq || freq <= 0) return { name: '—', octave: 0, centsOff: 0, noteStr: '—' };
  const midi = 12 * Math.log2(freq / 440) + 69;
  const roundedMidi = Math.round(midi);
  const centsOff = Math.round((midi - roundedMidi) * 100);
  const octave = Math.floor(roundedMidi / 12) - 1;
  const noteName = WESTERN_NOTE_NAMES[roundedMidi % 12];
  return {
    name: noteName,
    octave,
    centsOff,
    noteStr: `${noteName}${octave}${centsOff >= 0 ? '+' : ''}${centsOff}¢`,
  };
}

/**
 * Track fundamental pitch F0 across audio buffer specifically for singing voice.
 * Uses YIN algorithm with parabolic interpolation and voice activity detection (VAD).
 * 
 * @param {AudioBuffer} audioBuffer
 * @param {Function} onProgress
 * @returns {Promise<{ timeSteps: number[], pitches: number[], confidences: number[] }>}
 */
export async function trackVocalPitch(audioBuffer, onProgress = () => {}) {
  const sampleRate = audioBuffer.sampleRate;
  const channelData = audioBuffer.getChannelData(0);
  const totalSamples = channelData.length;

  // Windowing parameters optimized for singing vocals:
  // 44.1kHz: window = 2048 (~46ms), hop = 441 (~10ms step for smooth continuous curve)
  const windowSize = Math.pow(2, Math.ceil(Math.log2(sampleRate * 0.046))); // ~46ms
  const hopSize = Math.floor(sampleRate * 0.010); // 10ms frame step (100 frames/sec)
  const numFrames = Math.floor((totalSamples - windowSize) / hopSize);

  // Vocal bounds
  const minFreq = 65;   // Low male voice C2
  const maxFreq = 950;  // High soprano / Taar Saptak high B5

  const minLag = Math.floor(sampleRate / maxFreq);
  const maxLag = Math.floor(sampleRate / minFreq);

  const timeSteps = [];
  const rawPitches = [];
  const confidences = [];

  const yinBuffer = new Float32Array(maxLag);

  for (let f = 0; f < numFrames; f++) {
    const startSample = f * hopSize;
    const time = (startSample + windowSize / 2) / sampleRate;

    // Check RMS energy for Voice Activity Detection (ignore silence & breaths)
    let sumSquares = 0;
    for (let i = 0; i < windowSize; i++) {
      const val = channelData[startSample + i];
      sumSquares += val * val;
    }
    const rms = Math.sqrt(sumSquares / windowSize);

    if (rms < 0.012) {
      // Silence or background breath
      timeSteps.push(time);
      rawPitches.push(0);
      confidences.push(0);
      continue;
    }

    // Step 1: Difference Function
    for (let tau = 0; tau < maxLag; tau++) {
      let sum = 0;
      for (let i = 0; i < windowSize / 2; i++) {
        const delta = channelData[startSample + i] - channelData[startSample + i + tau];
        sum += delta * delta;
      }
      yinBuffer[tau] = sum;
    }

    // Step 2: Cumulative Mean Normalized Difference Function
    yinBuffer[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau < maxLag; tau++) {
      runningSum += yinBuffer[tau];
      yinBuffer[tau] = runningSum > 0 ? (yinBuffer[tau] * tau) / runningSum : 1;
    }

    // Step 3: Absolute Thresholding
    const threshold = 0.18;
    let tau = minLag;
    while (tau < maxLag) {
      if (yinBuffer[tau] < threshold) {
        while (tau + 1 < maxLag && yinBuffer[tau + 1] < yinBuffer[tau]) {
          tau++;
        }
        break;
      }
      tau++;
    }

    let detectedFreq = 0;
    let confidence = 0;

    if (tau < maxLag && yinBuffer[tau] < threshold) {
      // Step 4: Parabolic Interpolation for exact sub-sample frequency
      let betterTau = tau;
      const x0 = tau > 0 ? tau - 1 : tau;
      const x2 = tau + 1 < maxLag ? tau + 1 : tau;
      const s0 = yinBuffer[x0];
      const s1 = yinBuffer[tau];
      const s2 = yinBuffer[x2];

      const denom = 2 * (2 * s1 - s2 - s0);
      if (Math.abs(denom) > 1e-6) {
        betterTau = tau + (s2 - s0) / denom;
      }

      detectedFreq = sampleRate / betterTau;
      confidence = Math.max(0, 1 - yinBuffer[tau]);
    }

    timeSteps.push(time);
    rawPitches.push(detectedFreq >= minFreq && detectedFreq <= maxFreq ? detectedFreq : 0);
    confidences.push(confidence);

    if (f % 150 === 0) {
      onProgress((f / numFrames) * 0.45);
    }
  }

  // Step 5: Post-processing Median Filter to remove sporadic vocal octave jumps
  const cleanedPitches = medianSmoothPitches(rawPitches, 5);

  return { timeSteps, pitches: cleanedPitches, confidences };
}

/**
 * 5-point median filter to clean sporadic pitch glitches without smoothing vocal slides.
 */
function medianSmoothPitches(pitches, kernelSize = 5) {
  const result = new Float32Array(pitches.length);
  const half = Math.floor(kernelSize / 2);

  for (let i = 0; i < pitches.length; i++) {
    if (pitches[i] === 0) {
      result[i] = 0;
      continue;
    }
    const window = [];
    for (let k = -half; k <= half; k++) {
      const idx = i + k;
      if (idx >= 0 && idx < pitches.length && pitches[idx] > 0) {
        window.push(pitches[idx]);
      }
    }
    if (window.length > 0) {
      window.sort((a, b) => a - b);
      result[i] = window[Math.floor(window.length / 2)];
    } else {
      result[i] = 0;
    }
  }
  return Array.from(result);
}

/**
 * Automatic Tonic (Sa) Identification from pitch class distribution.
 * 
 * Does NOT assume any default pitch!
 * Analyzes the folded pitch class histogram of the singing voice,
 * evaluates stable sustained vowel points, and correlates against
 * the fundamental Sa-Pa (0 - 700 cent) tonic anchor template of Indian music.
 * 
 * @param {number[]} pitches - Array of detected F0 values in Hz
 * @param {number[]} confidences - Frame periodicity confidences
 * @returns {{ tonicFreq: number, tonicWestern: string, confidence: number, octave: number }}
 */
export function identifyTonicSa(pitches, confidences) {
  const validPitches = [];
  const validWeights = [];

  for (let i = 0; i < pitches.length; i++) {
    if (pitches[i] > 65 && confidences[i] > 0.4) {
      validPitches.push(pitches[i]);
      validWeights.push(confidences[i]);
    }
  }

  if (validPitches.length === 0) {
    // If no voiced sound was detected, fallback safely
    return {
      tonicFreq: 146.83,
      tonicWestern: 'D3',
      confidence: 0,
      octave: 3,
    };
  }

  // 1. Fold all pitches into cents modulo 1200 relative to a fixed baseline (C1 = 32.703 Hz)
  const baseFreq = 32.703;
  const numBins = 1200; // 1 bin per cent
  const histogram = new Float32Array(numBins);

  for (let i = 0; i < validPitches.length; i++) {
    const centsTotal = 1200 * Math.log2(validPitches[i] / baseFreq);
    const centMod = Math.round(centsTotal) % 1200;
    const bin = (centMod + 1200) % 1200;

    // Apply Gaussian spreading (sigma = 18 cents) to capture natural vocal vibrato
    const sigma = 18;
    for (let offset = -40; offset <= 40; offset++) {
      const targetBin = (bin + offset + 1200) % 1200;
      const weight = Math.exp(-(offset * offset) / (2 * sigma * sigma)) * validWeights[i];
      histogram[targetBin] += weight;
    }
  }

  // 2. Indian Art Music Tonic Template (CompMusic standard model)
  // Sa is the primary anchor (weight 1.0)
  // Pa (700 cents) is the perfect fifth harmonic anchor (weight 0.75)
  // Shuddha Ma (500 cents) is the fourth harmonic anchor (weight 0.50)
  // Major/Minor seconds and thirds have secondary consonant weights
  const templateIntervals = [
    { cents: 0,    weight: 1.00 }, // Sa
    { cents: 200,  weight: 0.40 }, // Shuddha Re
    { cents: 400,  weight: 0.45 }, // Shuddha Ga
    { cents: 500,  weight: 0.55 }, // Shuddha Ma
    { cents: 700,  weight: 0.80 }, // Pa (Pancham)
    { cents: 900,  weight: 0.35 }, // Shuddha Dha
    { cents: 1100, weight: 0.35 }, // Shuddha Ni
  ];

  let bestCent = 0;
  let maxScore = -1;

  for (let c = 0; c < 1200; c++) {
    let score = 0;
    for (const item of templateIntervals) {
      const binIdx = (c + item.cents) % 1200;
      score += histogram[binIdx] * item.weight;
    }
    // Sa must have substantial direct energy
    score += histogram[c] * 0.8;

    if (score > maxScore) {
      maxScore = score;
      bestCent = c;
    }
  }

  // 3. Map the winning modulo cent back to the singer's actual median vocal octave
  // Find median frequency among valid pitches
  const sortedPitches = [...validPitches].sort((a, b) => a - b);
  const medianVocalPitch = sortedPitches[Math.floor(sortedPitches.length / 2)];

  // Convert bestCent to frequency in candidate octaves (octaves 1 to 5)
  let bestTonicFreq = 0;
  let minOctaveDist = Infinity;

  for (let oct = 1; oct <= 5; oct++) {
    const candidateFreq = baseFreq * Math.pow(2, (bestCent + oct * 1200) / 1200);
    const dist = Math.abs(Math.log2(candidateFreq / medianVocalPitch));
    if (dist < minOctaveDist) {
      minOctaveDist = dist;
      bestTonicFreq = candidateFreq;
    }
  }

  // Refine exact peak frequency around bestTonicFreq using actual pitch clusters
  const cluster = validPitches.filter(f => {
    const diffCents = Math.abs(1200 * Math.log2(f / bestTonicFreq));
    return diffCents < 45;
  });

  let refinedTonicFreq = bestTonicFreq;
  if (cluster.length >= 5) {
    cluster.sort((a, b) => a - b);
    refinedTonicFreq = cluster[Math.floor(cluster.length / 2)];
  }

  const western = hzToWesternNote(refinedTonicFreq);
  const confidenceScore = Math.min(1, Math.max(0.4, (cluster.length / validPitches.length) * 4));

  return {
    tonicFreq: Number(refinedTonicFreq.toFixed(1)),
    tonicWestern: western.noteStr,
    confidence: Number(confidenceScore.toFixed(2)),
    octave: western.octave,
  };
}

/**
 * Map a frequency (Hz) to its Indian Swara and Saptak based on the detected Sa.
 * 
 * @param {number} freq - Pitch in Hz
 * @param {number} tonicFreq - Detected Sa frequency in Hz
 * @returns {{ swara: object, saptak: string, saptakLabel: string, centsDeviation: number, formatted: string }}
 */
export function freqToSwara(freq, tonicFreq) {
  if (!freq || freq <= 0 || !tonicFreq || tonicFreq <= 0) {
    return null;
  }

  // Exact interval in cents from detected Sa
  const centsFromSa = 1200 * Math.log2(freq / tonicFreq);
  
  // Saptak calculation:
  // centsFromSa in [-600, +599] => Madhya Saptak
  // centsFromSa < -600 => Mandra Saptak (lower)
  // centsFromSa >= +600 + 1200*k => Taar Saptak (higher)
  const octaveShift = Math.floor((centsFromSa + 600) / 1200);
  const normalizedCents = ((centsFromSa % 1200) + 1200) % 1200;

  // Find nearest Swara (each 100 cents)
  const swaraIndex = Math.round(normalizedCents / 100) % 12;
  const swaraDef = SWARA_DEFINITIONS[swaraIndex];
  const exactSwaraCents = swaraDef.cents;
  const centsDeviation = Math.round(normalizedCents - exactSwaraCents);

  let saptak = 'Madhya';
  let saptakSymbol = '';
  let saptakLabel = 'Madhya Saptak (Mid)';

  if (octaveShift < 0) {
    saptak = 'Mandra';
    saptakSymbol = '̣'; // dot below
    saptakLabel = 'Mandra Saptak (Low)';
  } else if (octaveShift > 0) {
    saptak = 'Taar';
    saptakSymbol = '̇'; // dot above
    saptakLabel = 'Taar Saptak (High)';
  }

  const formatted = `${swaraDef.name}${saptakSymbol}`;

  return {
    swara: swaraDef,
    swaraIndex,
    saptak,
    saptakLabel,
    octaveShift,
    centsDeviation,
    formatted,
    targetFreq: tonicFreq * Math.pow(2, (swaraDef.cents + octaveShift * 1200) / 1200),
  };
}

/**
 * Segment continuous vocal pitch trajectory into distinct Swara note events.
 * Captures start, end, duration, mean frequency, and Swara identity.
 * 
 * @param {number[]} timeSteps
 * @param {number[]} pitches
 * @param {number[]} confidences
 * @param {number} tonicFreq
 * @returns {Array<object>}
 */
export function segmentSwaras(timeSteps, pitches, confidences, tonicFreq) {
  const notes = [];
  let currentSegment = null;

  const minNoteDuration = 0.08; // 80ms minimum vocal note duration

  for (let i = 0; i < pitches.length; i++) {
    const p = pitches[i];
    const t = timeSteps[i];
    const conf = confidences[i];

    if (p > 65 && conf > 0.35) {
      const swaraInfo = freqToSwara(p, tonicFreq);
      const swaraKey = `${swaraInfo.octaveShift}_${swaraInfo.swaraIndex}`;

      if (currentSegment && currentSegment.key === swaraKey) {
        currentSegment.endTime = t;
        currentSegment.pitches.push(p);
      } else {
        if (currentSegment) {
          finalizeSegment(currentSegment, notes, minNoteDuration);
        }
        currentSegment = {
          key: swaraKey,
          swaraInfo,
          startTime: t,
          endTime: t,
          pitches: [p],
        };
      }
    } else {
      if (currentSegment) {
        finalizeSegment(currentSegment, notes, minNoteDuration);
        currentSegment = null;
      }
    }
  }

  if (currentSegment) {
    finalizeSegment(currentSegment, notes, minNoteDuration);
  }

  return notes;
}

function finalizeSegment(segment, notesArray, minDuration) {
  const duration = segment.endTime - segment.startTime;
  if (duration >= minDuration && segment.pitches.length > 0) {
    const sum = segment.pitches.reduce((acc, v) => acc + v, 0);
    const meanHz = Number((sum / segment.pitches.length).toFixed(1));

    notesArray.push({
      swara: segment.swaraInfo.swara,
      name: segment.swaraInfo.swara.name,
      formatted: segment.swaraInfo.formatted,
      saptak: segment.swaraInfo.saptak,
      saptakLabel: segment.swaraInfo.saptakLabel,
      startTime: Number(segment.startTime.toFixed(2)),
      endTime: Number(segment.endTime.toFixed(2)),
      duration: Number(duration.toFixed(2)),
      meanHz,
      centsDeviation: segment.swaraInfo.centsDeviation,
    });
  }
}

/**
 * Generate human-readable Sargam Sequence transcript.
 * e.g. "S — R — G — m — P — D — N — Ṡ"
 */
export function generateSargamTranscript(swaraNotes) {
  if (!swaraNotes || swaraNotes.length === 0) return '';
  return swaraNotes.map(n => n.formatted).join('  —  ');
}
