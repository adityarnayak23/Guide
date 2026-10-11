/**
 * NoteScope — Authentic Indian Classical Miraj & Tanjore Tanpura (Tambura) Engine
 * 
 * Features:
 * 1. Studio-Mastered Acoustic Sample Engine:
 *    - High-fidelity 44.1kHz stereo recordings of authentic Miraj & Tanjore acoustic Tanpuras.
 *    - Individual string pluck samples: Pa, Ma, Ni, Jodi Sa 1, Jodi Sa 2 (with Jiva detune), Kharaj Sa.
 *    - Seamless 3-cycle acoustic studio drone loops.
 * 2. Real-Time Pitch Resampling across ALL 12 SCALES & MICROTONAL CENTS:
 *    - Pitch-shifted via Web Audio sinc-interpolating resampler from base D3 (146.83 Hz).
 *    - Full chromatic spectrum: C, C#, D, D#, E, F, F#, G, G#, A, A#, B.
 *    - Male range (Octave 2-3) & Female range (Octave 3-4).
 *    - Fine-tuning in cents (-50 to +50) and arbitrary Hz.
 * 3. Authentic 4-String Classical Cycle Sequence:
 *    - String 1: Panchama (Pa 1.5x) / Madhyama (Ma 1.33x) / Nishadha (Ni 1.875x) (panned left)
 *    - String 2: Jodi Sa 1 (Taar / Madhya Sa 2.0x, slightly left)
 *    - String 3: Jodi Sa 2 (Taar Sa 2.0x with +1.25 cents natural Jiva acoustic detune, slightly right)
 *    - String 4: Kharaj Sa (Mandra Sa fundamental 1.0x, deep bronze string, centered)
 *    - Natural meditative breath pause after Kharaj Sa before next cycle.
 * 4. Continuous Sympathetic Acoustic Resonance (Acoustic Aura):
 *    - Gourd cavity room ambience that glues string strokes together so sound is never dry or disconnected.
 * 5. High-Order Physical Modeling Fallback:
 *    - 48-partial custom Fourier PeriodicWave tables with dynamic Javari bloom and asymmetric bridge wave-shaping.
 */

export const TANPURA_SCALES = [
  { note: 'C',  kattai: '1 Kattai',   maleHz: 130.81, femaleHz: 261.63 },
  { note: 'C#', kattai: '1.5 Kattai', maleHz: 138.59, femaleHz: 277.18 },
  { note: 'D',  kattai: '2 Kattai',   maleHz: 146.83, femaleHz: 293.66 },
  { note: 'D#', kattai: '2.5 Kattai', maleHz: 155.56, femaleHz: 311.13 },
  { note: 'E',  kattai: '3 Kattai',   maleHz: 164.81, femaleHz: 329.63 },
  { note: 'F',  kattai: '4 Kattai',   maleHz: 174.61, femaleHz: 349.23 },
  { note: 'F#', kattai: '4.5 Kattai', maleHz: 185.00, femaleHz: 369.99 },
  { note: 'G',  kattai: '5 Kattai',   maleHz: 196.00, femaleHz: 392.00 },
  { note: 'G#', kattai: '5.5 Kattai', maleHz: 207.65, femaleHz: 415.30 },
  { note: 'A',  kattai: '6 Kattai',   maleHz: 220.00, femaleHz: 440.00 },
  { note: 'A#', kattai: '6.5 Kattai', maleHz: 233.08, femaleHz: 466.16 },
  { note: 'B',  kattai: '7 Kattai',   maleHz: 246.94, femaleHz: 493.88 },
];

export const BASE_RECORDING_HZ = 146.83; // D3 (Classical standard)

export class TanpuraSynth {
  constructor(initialSaHz = 146.83) {
    this.saHz = initialSaHz;
    this.currentScaleNote = 'D';
    this.octave = 'male'; // 'male' or 'female'
    this.fineTuneCents = 0;
    this.firstString = 'Pa'; // 'Pa', 'Ma', or 'Ni'
    this.timbre = 'miraj'; // 'miraj' or 'tanjore'
    this.tempo = 'medium'; // 'slow', 'medium', 'fast'
    this.mode = 'acoustic'; // 'acoustic' (sample-based cycle) or 'loop' (continuous concert loop)
    this.isPlaying = false;
    this.volume = 0.65;

    this.audioCtx = null;
    this.masterGain = null;
    this.droneGain = null;
    this.pluckGain = null;
    this.bodyFilterIn = null;
    this.timerId = null;
    this.stringStep = 0;

    // Acoustic sample buffers
    this.sampleBuffers = {
      pa: null,
      ma: null,
      ni: null,
      jodi1: null,
      jodi2: null,
      kharaj: null,
      cycle_pa: null,
      cycle_ma: null,
      cycle_ni: null,
    };
    this.samplesLoaded = false;
    this.isLoadingSamples = false;

    // Active nodes
    this.loopSourceNode = null;
    this.activeVoices = new Set();
    this.droneVoices = [];

    // Fallback physical model tables
    this.periodicWaves = null;
    this.waveshaperCurve = null;
    this.pluckNoiseBuffer = null;
  }

  getAudioContext() {
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        this.audioCtx = new AudioContextClass();
      }
    }
    return this.audioCtx;
  }

  async initAudio() {
    const ctx = this.getAudioContext();
    if (!ctx) return;

    if (!this.masterGain) {
      // 1. Master Output Gain Node
      this.masterGain = ctx.createGain();
      this.masterGain.gain.setValueAtTime(this.volume, ctx.currentTime);
      this.masterGain.connect(ctx.destination);

      // 2. Gourd Body Cavity (Tumba) Resonant Filter Bank
      const helmholtzLow = ctx.createBiquadFilter();
      helmholtzLow.type = 'lowshelf';
      helmholtzLow.frequency.value = 115;
      helmholtzLow.gain.value = 3.5;

      const woodBodyPeak = ctx.createBiquadFilter();
      woodBodyPeak.type = 'peaking';
      woodBodyPeak.frequency.value = 235;
      woodBodyPeak.Q.value = 2.4;
      woodBodyPeak.gain.value = 2.0;

      const waistCouplingPeak = ctx.createBiquadFilter();
      waistCouplingPeak.type = 'peaking';
      waistCouplingPeak.frequency.value = 460;
      waistCouplingPeak.Q.value = 2.8;
      waistCouplingPeak.gain.value = 1.5;

      const mirajJavariSingingFormant = ctx.createBiquadFilter();
      mirajJavariSingingFormant.type = 'peaking';
      mirajJavariSingingFormant.frequency.value = 1480;
      mirajJavariSingingFormant.Q.value = 3.2;
      mirajJavariSingingFormant.gain.value = 2.8;

      const silkThreadShimmerPeak = ctx.createBiquadFilter();
      silkThreadShimmerPeak.type = 'peaking';
      silkThreadShimmerPeak.frequency.value = 2800;
      silkThreadShimmerPeak.Q.value = 2.8;
      silkThreadShimmerPeak.gain.value = 1.8;

      const antiFizzLowpass = ctx.createBiquadFilter();
      antiFizzLowpass.type = 'lowpass';
      antiFizzLowpass.frequency.value = 8500;
      antiFizzLowpass.Q.value = 0.7;

      helmholtzLow.connect(woodBodyPeak);
      woodBodyPeak.connect(waistCouplingPeak);
      waistCouplingPeak.connect(mirajJavariSingingFormant);
      mirajJavariSingingFormant.connect(silkThreadShimmerPeak);
      silkThreadShimmerPeak.connect(antiFizzLowpass);
      antiFizzLowpass.connect(this.masterGain);

      this.bodyFilterIn = helmholtzLow;

      this.droneGain = ctx.createGain();
      this.droneGain.gain.setValueAtTime(0.20, ctx.currentTime);
      this.droneGain.connect(this.bodyFilterIn);

      this.pluckGain = ctx.createGain();
      this.pluckGain.gain.setValueAtTime(0.92, ctx.currentTime);
      this.pluckGain.connect(this.bodyFilterIn);

      // Pre-generate fallback physical wave tables
      this.buildPeriodicWaves(ctx);
      this.waveshaperCurve = this.createJavariCurve();
      this.pluckNoiseBuffer = this.createPluckBuffer(ctx);

      // Pre-load master studio acoustic samples
      this.loadAcousticSamples(ctx);
    }

    if (ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch (err) {
        console.warn('AudioContext resume was blocked:', err);
      }
    }
  }

  async loadAcousticSamples(ctx) {
    if (this.samplesLoaded || this.isLoadingSamples) return;
    this.isLoadingSamples = true;

    const sampleFiles = {
      pa: '/audio/tanpura/string_pa.wav',
      ma: '/audio/tanpura/string_ma.wav',
      ni: '/audio/tanpura/string_ni.wav',
      jodi1: '/audio/tanpura/string_jodi1.wav',
      jodi2: '/audio/tanpura/string_jodi2.wav',
      kharaj: '/audio/tanpura/string_kharaj.wav',
      cycle_pa: '/audio/tanpura/tanpura_cycle_pa.wav',
      cycle_ma: '/audio/tanpura/tanpura_cycle_ma.wav',
      cycle_ni: '/audio/tanpura/tanpura_cycle_ni.wav',
    };

    const loadEntries = Object.entries(sampleFiles).map(async ([key, url]) => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const arrayBuffer = await res.arrayBuffer();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        this.sampleBuffers[key] = audioBuffer;
      } catch (err) {
        // Fallback to synthetic if fetch fails
        // console.debug(`Acoustic sample ${key} failed to load, using DSP model:`, err);
      }
    });

    try {
      await Promise.all(loadEntries);
      if (this.sampleBuffers.pa && this.sampleBuffers.kharaj) {
        this.samplesLoaded = true;
        // console.log('Master Acoustic Tanpura Samples Loaded Successfully.');
      }
    } catch {
      // Continue with DSP fallback
    } finally {
      this.isLoadingSamples = false;
    }
  }

  buildPeriodicWaves(ctx) {
    const N = 48;
    const real = new Float32Array(N);
    const imagBody = new Float32Array(N);
    const imagMiraj = new Float32Array(N);
    const imagTanjore = new Float32Array(N);
    const imagKharaj = new Float32Array(N);

    real[0] = 0;
    imagBody[0] = 0;
    imagMiraj[0] = 0;
    imagTanjore[0] = 0;
    imagKharaj[0] = 0;

    for (let n = 1; n < N; n++) {
      imagBody[n] = (1.0 / Math.pow(n, 1.1)) * (n <= 5 ? 1.0 : 0.18);
      const mirajFormant = Math.exp(-0.5 * Math.pow((n - 13) / 6.5, 2)) * 2.8 + 0.35;
      imagMiraj[n] = (1.0 / Math.pow(n, 0.88)) * mirajFormant;
      const tanjoreFormant = Math.exp(-0.5 * Math.pow((n - 18) / 7.0, 2)) * 3.2 + 0.28;
      imagTanjore[n] = (1.0 / Math.pow(n, 0.82)) * tanjoreFormant;
      imagKharaj[n] = n === 1 ? 1.0 : (0.85 / Math.pow(n, 1.02)) * (n <= 8 ? 1.25 : 0.6);
    }

    this.periodicWaves = {
      body: ctx.createPeriodicWave(real, imagBody, { disableNormalization: false }),
      miraj: ctx.createPeriodicWave(real, imagMiraj, { disableNormalization: false }),
      tanjore: ctx.createPeriodicWave(real, imagTanjore, { disableNormalization: false }),
      kharaj: ctx.createPeriodicWave(real, imagKharaj, { disableNormalization: false }),
    };
  }

  createJavariCurve() {
    const n = 512;
    const curve = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      if (x < 0) {
        curve[i] = Math.tanh(x * 1.35) * 0.88;
      } else {
        curve[i] = Math.tanh(x * 2.5) * 1.12 + (x * 0.12);
      }
    }
    return curve;
  }

  createPluckBuffer(ctx) {
    const sr = ctx.sampleRate;
    const len = Math.floor(sr * 0.035);
    const buffer = ctx.createBuffer(1, len, sr);
    const data = buffer.getChannelData(0);
    let last = 0;
    for (let i = 0; i < len; i++) {
      const t = i / len;
      const white = (Math.random() * 2 - 1);
      last = last * 0.80 + white * 0.20;
      data[i] = last * Math.pow(1 - t, 2.2) * 0.25;
    }
    return buffer;
  }

  setSa(hz) {
    if (hz && hz > 60 && hz < 700) {
      this.saHz = hz;
      this.updateDroneFrequencies();
      this.updateActivePitchPlaybackRate();
    }
  }

  setScale(noteName, octave = null) {
    const scale = TANPURA_SCALES.find(s => s.note.toUpperCase() === noteName.toUpperCase());
    if (scale) {
      this.currentScaleNote = scale.note;
      if (octave) this.octave = octave;
      const baseHz = this.octave === 'female' ? scale.femaleHz : scale.maleHz;
      const tunedHz = baseHz * Math.pow(2, this.fineTuneCents / 1200);
      this.setSa(tunedHz);
    }
  }

  setOctave(octave) {
    if (octave === 'male' || octave === 'female') {
      this.octave = octave;
      this.setScale(this.currentScaleNote);
    }
  }

  setFineTune(cents) {
    this.fineTuneCents = Math.max(-50, Math.min(50, cents));
    this.setScale(this.currentScaleNote);
  }

  setFirstString(type) {
    if (['Pa', 'Ma', 'Ni'].includes(type)) {
      this.firstString = type;
      this.updateDroneFrequencies();
      if (this.isPlaying && this.mode === 'loop') {
        this.restartLoopDrone();
      }
    }
  }

  setTimbre(timbre) {
    if (timbre === 'miraj' || timbre === 'tanjore') {
      this.timbre = timbre;
    }
  }

  setTempo(tempo) {
    if (['slow', 'medium', 'fast'].includes(tempo) || typeof tempo === 'number') {
      this.tempo = tempo;
    }
  }

  setMode(mode) {
    if (mode === 'acoustic' || mode === 'loop') {
      this.mode = mode;
      if (this.isPlaying) {
        this.stop();
        this.start();
      }
    }
  }

  setVolume(vol) {
    this.volume = Math.max(0, Math.min(1, vol));
    if (this.masterGain && this.audioCtx) {
      this.masterGain.gain.setTargetAtTime(this.volume, this.audioCtx.currentTime, 0.05);
    }
  }

  getPlaybackRateMultiplier() {
    return this.saHz / BASE_RECORDING_HZ;
  }

  updateActivePitchPlaybackRate() {
    if (!this.audioCtx) return;
    const rate = this.getPlaybackRateMultiplier();
    if (this.loopSourceNode) {
      try {
        this.loopSourceNode.playbackRate.setTargetAtTime(rate, this.audioCtx.currentTime, 0.05);
      } catch {
        // Node might not support ramping
      }
    }
  }

  start() {
    if (this.isPlaying) return;
    this.initAudio();
    this.isPlaying = true;
    this.stringStep = 0;

    // Start continuous sympathetic drone bed
    this.startSympatheticDrone();

    if (this.mode === 'loop' && this.samplesLoaded) {
      this.startLoopDrone();
    } else {
      // Start rhythmic 4-string plucking cycle
      this.scheduleNextPluck();
    }
  }

  stop() {
    this.isPlaying = false;
    if (this.timerId) {
      clearTimeout(this.timerId);
      this.timerId = null;
    }

    this.stopLoopDrone();
    this.stopSympatheticDrone();

    if (this.onPluckCallback) {
      this.onPluckCallback(-1, null);
    }

    if (this.audioCtx) {
      const now = this.audioCtx.currentTime;
      this.activeVoices.forEach(voice => {
        try {
          if (voice.gainNode) {
            voice.gainNode.gain.cancelScheduledValues(now);
            voice.gainNode.gain.setTargetAtTime(0.0001, now, 0.15);
          }
          if (voice.sources) {
            voice.sources.forEach(src => src.stop(now + 0.35));
          }
        } catch {
          // Ignore
        }
      });
      this.activeVoices.clear();
    }
  }

  toggle() {
    if (this.isPlaying) {
      this.stop();
      return false;
    } else {
      this.start();
      return true;
    }
  }

  startLoopDrone() {
    if (!this.audioCtx || !this.samplesLoaded) return;
    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    let cycleBuffer = this.sampleBuffers.cycle_pa;
    if (this.firstString === 'Ma') cycleBuffer = this.sampleBuffers.cycle_ma;
    else if (this.firstString === 'Ni') cycleBuffer = this.sampleBuffers.cycle_ni;

    if (!cycleBuffer) {
      // Fallback to sequential plucking
      this.scheduleNextPluck();
      return;
    }

    const src = ctx.createBufferSource();
    src.buffer = cycleBuffer;
    src.loop = true;
    src.playbackRate.value = this.getPlaybackRateMultiplier();

    const loopGain = ctx.createGain();
    loopGain.gain.setValueAtTime(0.0001, now);
    loopGain.gain.linearRampToValueAtTime(0.92, now + 1.2);

    src.connect(loopGain);
    loopGain.connect(this.pluckGain);

    src.start(now);
    this.loopSourceNode = src;
    this.loopGainNode = loopGain;
  }

  stopLoopDrone() {
    if (!this.loopSourceNode || !this.audioCtx) return;
    const ctx = this.audioCtx;
    const now = ctx.currentTime;
    try {
      if (this.loopGainNode) {
        this.loopGainNode.gain.cancelScheduledValues(now);
        this.loopGainNode.gain.linearRampToValueAtTime(0.0001, now + 0.8);
      }
      this.loopSourceNode.stop(now + 0.85);
    } catch {
      // Ignore
    }
    this.loopSourceNode = null;
    this.loopGainNode = null;
  }

  restartLoopDrone() {
    this.stopLoopDrone();
    if (this.isPlaying) {
      this.startLoopDrone();
    }
  }

  getStringInfo(step) {
    const root = this.saHz;
    switch (step) {
      case 0: {
        let ratio = 1.5;
        let name = 'Pa (Panchamam)';
        let sampleKey = 'pa';
        if (this.firstString === 'Ma') {
          ratio = 4.0 / 3.0;
          name = 'Ma (Madhyamam)';
          sampleKey = 'ma';
        } else if (this.firstString === 'Ni') {
          ratio = 15.0 / 8.0;
          name = 'Ni (Nishadham)';
          sampleKey = 'ni';
        }
        return { freq: root * ratio, name, sampleKey, pan: -0.22, decay: 6.8, isKharaj: false };
      }
      case 1:
        return { freq: root * 2.0, name: 'Jodi Sa 1', sampleKey: 'jodi1', pan: -0.07, decay: 6.5, isKharaj: false };
      case 2:
        return { freq: root * 2.0 * Math.pow(2, 1.25 / 1200), name: 'Jodi Sa 2', sampleKey: 'jodi2', pan: 0.07, decay: 6.5, isKharaj: false };
      case 3:
      default:
        return { freq: root * 1.0, name: 'Kharaj Sa', sampleKey: 'kharaj', pan: 0.0, decay: 8.2, isKharaj: true };
    }
  }

  /**
   * Plucks string using Studio Acoustic Master Sample if available,
   * falling back to physical modeling DSP engine.
   */
  pluckString(step, freq, options = {}) {
    if (!this.audioCtx || !this.isPlaying) return;
    const ctx = this.audioCtx;
    const now = ctx.currentTime;
    const { pan = 0, decay = 6.5, isKharaj = false, sampleKey = null } = options;

    const sampleBuffer = sampleKey && this.sampleBuffers[sampleKey];
    if (sampleBuffer) {
      // --- 1. Master Studio Acoustic Sample Playback ---
      const src = ctx.createBufferSource();
      src.buffer = sampleBuffer;
      src.playbackRate.value = this.getPlaybackRateMultiplier();

      const gainNode = ctx.createGain();
      gainNode.gain.setValueAtTime(isKharaj ? 1.05 : 0.90, now);
      // Gentle fade-out towards end of buffer
      gainNode.gain.setValueAtTime(isKharaj ? 1.05 : 0.90, now + decay * 0.85);
      gainNode.gain.linearRampToValueAtTime(0.0001, now + decay);

      const panner = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
      if (panner) panner.pan.setValueAtTime(pan, now);

      if (panner) {
        src.connect(gainNode);
        gainNode.connect(panner);
        panner.connect(this.pluckGain);
      } else {
        src.connect(gainNode);
        gainNode.connect(this.pluckGain);
      }

      src.start(now);
      src.stop(now + decay + 0.1);

      const voiceRecord = { gainNode, sources: [src] };
      this.activeVoices.add(voiceRecord);
      setTimeout(() => this.activeVoices.delete(voiceRecord), (decay + 0.2) * 1000);
      return;
    }

    // --- 2. Fallback: High-Precision Physical Modeling Synthesis ---
    const voiceGain = ctx.createGain();
    const voicePan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
    if (voicePan) voicePan.pan.setValueAtTime(pan, now);

    const javariWave = this.timbre === 'tanjore' ? this.periodicWaves.tanjore : this.periodicWaves.miraj;
    const bodyWave = isKharaj ? this.periodicWaves.kharaj : this.periodicWaves.body;

    const oscBody = ctx.createOscillator();
    const gainBody = ctx.createGain();
    oscBody.setPeriodicWave(bodyWave);
    oscBody.frequency.setValueAtTime(freq, now);

    gainBody.gain.setValueAtTime(0.0001, now);
    gainBody.gain.exponentialRampToValueAtTime(0.55, now + 0.03);
    gainBody.gain.exponentialRampToValueAtTime(0.0001, now + decay);

    oscBody.connect(gainBody);
    gainBody.connect(voiceGain);

    const oscJavari = ctx.createOscillator();
    const gainJavari = ctx.createGain();
    oscJavari.setPeriodicWave(javariWave);
    oscJavari.frequency.setValueAtTime(freq, now);

    const flutterOsc = ctx.createOscillator();
    const flutterGain = ctx.createGain();
    flutterOsc.frequency.setValueAtTime(3.1, now);
    flutterGain.gain.setValueAtTime(freq * 0.0035, now);
    flutterOsc.connect(flutterGain);
    flutterGain.connect(oscJavari.frequency);

    const javariFilter = ctx.createBiquadFilter();
    javariFilter.type = 'bandpass';
    javariFilter.frequency.setValueAtTime(Math.min(3200, freq * 3.5), now);
    javariFilter.frequency.exponentialRampToValueAtTime(Math.min(4800, freq * 7.0), now + 0.28);
    javariFilter.frequency.exponentialRampToValueAtTime(Math.min(1800, freq * 2.5), now + decay);
    javariFilter.Q.setValueAtTime(2.2, now);

    const shaper = ctx.createWaveShaper();
    shaper.curve = this.waveshaperCurve;
    shaper.oversample = '2x';

    gainJavari.gain.setValueAtTime(0.0001, now);
    gainJavari.gain.exponentialRampToValueAtTime(0.42, now + 0.22);
    gainJavari.gain.exponentialRampToValueAtTime(0.0001, now + decay * 0.95);

    oscJavari.connect(javariFilter);
    javariFilter.connect(shaper);
    shaper.connect(gainJavari);
    gainJavari.connect(voiceGain);

    if (this.pluckNoiseBuffer) {
      const noiseSrc = ctx.createBufferSource();
      const noiseGain = ctx.createGain();
      const noiseFilter = ctx.createBiquadFilter();

      noiseSrc.buffer = this.pluckNoiseBuffer;
      noiseFilter.type = 'lowpass';
      noiseFilter.frequency.setValueAtTime(Math.min(2200, freq * 4.0), now);

      noiseGain.gain.setValueAtTime(0.18, now);
      noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.038);

      noiseSrc.connect(noiseFilter);
      noiseFilter.connect(noiseGain);
      noiseGain.connect(voiceGain);

      noiseSrc.start(now);
      noiseSrc.stop(now + 0.04);
    }

    if (voicePan) {
      voiceGain.connect(voicePan);
      voicePan.connect(this.pluckGain);
    } else {
      voiceGain.connect(this.pluckGain);
    }

    flutterOsc.start(now);
    oscBody.start(now);
    oscJavari.start(now);

    const stopTime = now + decay + 0.2;
    flutterOsc.stop(stopTime);
    oscBody.stop(stopTime);
    oscJavari.stop(stopTime);

    const voiceRecord = {
      gainNode: voiceGain,
      sources: [oscBody, oscJavari, flutterOsc],
    };
    this.activeVoices.add(voiceRecord);
    setTimeout(() => this.activeVoices.delete(voiceRecord), (decay + 0.3) * 1000);
  }

  startSympatheticDrone() {
    if (!this.audioCtx || this.droneVoices.length > 0) return;
    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    const fundamentalHz = this.saHz;
    let fifthRatio = 1.5;
    if (this.firstString === 'Ma') fifthRatio = 4.0 / 3.0;
    else if (this.firstString === 'Ni') fifthRatio = 15.0 / 8.0;
    const fifthHz = fundamentalHz * fifthRatio;

    const saOsc = ctx.createOscillator();
    const saGain = ctx.createGain();
    saOsc.setPeriodicWave(this.periodicWaves.body);
    saOsc.frequency.setValueAtTime(fundamentalHz, now);

    const fifthOsc = ctx.createOscillator();
    const fifthGain = ctx.createGain();
    fifthOsc.setPeriodicWave(this.periodicWaves.body);
    fifthOsc.frequency.setValueAtTime(fifthHz, now);

    const lfo = ctx.createOscillator();
    const lfoGain = ctx.createGain();
    lfo.frequency.setValueAtTime(0.22, now);
    lfoGain.gain.setValueAtTime(0.8, now);
    lfo.connect(lfoGain);
    lfoGain.connect(saOsc.frequency);

    saGain.gain.setValueAtTime(0.0001, now);
    saGain.gain.linearRampToValueAtTime(0.12, now + 1.5);

    fifthGain.gain.setValueAtTime(0.0001, now);
    fifthGain.gain.linearRampToValueAtTime(0.09, now + 1.5);

    saOsc.connect(saGain);
    fifthOsc.connect(fifthGain);

    saGain.connect(this.droneGain);
    fifthGain.connect(this.droneGain);

    saOsc.start(now);
    fifthOsc.start(now);
    lfo.start(now);

    this.droneVoices = [
      { osc: saOsc, gain: saGain, type: 'sa' },
      { osc: fifthOsc, gain: fifthGain, type: 'fifth' },
      { osc: lfo, gain: lfoGain, type: 'lfo' },
    ];
  }

  stopSympatheticDrone() {
    if (this.droneVoices.length === 0 || !this.audioCtx) return;
    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    this.droneVoices.forEach(v => {
      try {
        if (v.gain) {
          v.gain.gain.cancelScheduledValues(now);
          v.gain.gain.linearRampToValueAtTime(0.0001, now + 0.8);
        }
        if (v.osc) v.osc.stop(now + 0.9);
      } catch {
        // Ignore
      }
    });
    this.droneVoices = [];
  }

  updateDroneFrequencies() {
    if (this.droneVoices.length === 0 || !this.audioCtx) return;
    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    const fundamentalHz = this.saHz;
    let fifthRatio = 1.5;
    if (this.firstString === 'Ma') fifthRatio = 4.0 / 3.0;
    else if (this.firstString === 'Ni') fifthRatio = 15.0 / 8.0;
    const fifthHz = fundamentalHz * fifthRatio;

    this.droneVoices.forEach(v => {
      try {
        if (v.type === 'sa' && v.osc) {
          v.osc.frequency.setTargetAtTime(fundamentalHz, now, 0.08);
        } else if (v.type === 'fifth' && v.osc) {
          v.osc.frequency.setTargetAtTime(fifthHz, now, 0.08);
        }
      } catch {
        // Ignore
      }
    });
  }

  setOnPluck(callback) {
    this.onPluckCallback = callback;
  }

  scheduleNextPluck() {
    if (!this.isPlaying || this.mode === 'loop') return;

    const info = this.getStringInfo(this.stringStep);
    if (this.onPluckCallback) {
      this.onPluckCallback(this.stringStep, info);
    }
    this.pluckString(this.stringStep, info.freq, info);

    let tempoMultiplier = 1.0;
    if (this.tempo === 'slow') tempoMultiplier = 1.35;
    else if (this.tempo === 'fast') tempoMultiplier = 0.78;
    else if (typeof this.tempo === 'number') tempoMultiplier = 1.0 / this.tempo;

    let delayMs = 1150 * tempoMultiplier;
    if (this.stringStep === 3) {
      delayMs = 1800 * tempoMultiplier;
    }

    this.stringStep = (this.stringStep + 1) % 4;

    this.timerId = setTimeout(() => {
      this.scheduleNextPluck();
    }, delayMs);
  }
}

export const tanpura = new TanpuraSynth(146.83);
