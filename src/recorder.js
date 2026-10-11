/**
 * NoteScope — Audio Recorder Module
 * Handles microphone recording with live waveform visualization.
 */

export class AudioRecorder {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
    this.analyser = null;
    this.audioContext = null;
    this.isRecording = false;
    this.startTime = 0;
    this.timerInterval = null;
    this.animationFrame = null;
  }

  /**
   * Request microphone access and prepare for recording.
   */
  async init() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
          channelCount: 1,
        }
      });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 44100,
      });

      const source = this.audioContext.createMediaStreamSource(this.stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.85;
      source.connect(this.analyser);

      return true;
    } catch (err) {
      console.error('Microphone access denied:', err);
      return false;
    }
  }

  /**
   * Start recording audio.
   * @param {Function} onTimerUpdate - Callback for timer updates (receives formatted time string)
   */
  start(onTimerUpdate) {
    if (!this.stream) return;

    this.audioChunks = [];
    this.mediaRecorder = new MediaRecorder(this.stream, {
      mimeType: this.getSupportedMimeType(),
    });

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        this.audioChunks.push(e.data);
      }
    };

    this.mediaRecorder.start(100); // Collect data every 100ms
    this.isRecording = true;
    this.startTime = Date.now();

    // Start timer
    if (onTimerUpdate) {
      this.timerInterval = setInterval(() => {
        const elapsed = Date.now() - this.startTime;
        onTimerUpdate(this.formatTime(elapsed));
      }, 100);
    }
  }

  /**
   * Stop recording and return the audio blob.
   * @returns {Promise<Blob>}
   */
  stop() {
    return new Promise((resolve) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = () => {
        const mimeType = this.getSupportedMimeType();
        const blob = new Blob(this.audioChunks, { type: mimeType });
        this.isRecording = false;
        clearInterval(this.timerInterval);
        resolve(blob);
      };

      this.mediaRecorder.stop();
    });
  }

  /**
   * Draw live waveform and track vocal pitch in real time.
   * @param {HTMLCanvasElement} canvas
   * @param {Function} [onPitchUpdate] - Callback with { pitchHz, rms }
   */
  drawLiveWaveform(canvas, onPitchUpdate) {
    if (!this.analyser || !canvas) return;

    const ctx = canvas.getContext('2d');
    const bufferLength = this.analyser.frequencyBinCount;
    const floatData = new Float32Array(bufferLength);
    const sampleRate = this.audioContext.sampleRate || 44100;

    let frameCount = 0;

    const draw = () => {
      if (!this.isRecording) return;

      this.animationFrame = requestAnimationFrame(draw);
      this.analyser.getFloatTimeDomainData(floatData);

      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      // Compute RMS energy
      let sumSq = 0;
      for (let i = 0; i < bufferLength; i++) {
        sumSq += floatData[i] * floatData[i];
      }
      const rms = Math.sqrt(sumSq / bufferLength);

      // Real-time Pitch Detection (runs every 2 frames for silky smooth 30-60fps updates)
      frameCount++;
      if (onPitchUpdate && frameCount % 2 === 0) {
        if (rms > 0.015) {
          const minFreq = 65;
          const maxFreq = 950;
          const minLag = Math.floor(sampleRate / maxFreq);
          const maxLag = Math.floor(sampleRate / minFreq);

          let bestLag = -1;
          let maxCorr = -1;
          let prevCorr = 0;
          let isRising = false;

          for (let lag = minLag; lag < maxLag; lag++) {
            let corr = 0;
            for (let i = 0; i < bufferLength - lag; i++) {
              corr += floatData[i] * floatData[i + lag];
            }
            if (corr > prevCorr) {
              isRising = true;
            } else if (isRising && prevCorr > 0.35 * sumSq) {
              if (prevCorr > maxCorr) {
                maxCorr = prevCorr;
                bestLag = lag - 1;
              }
              isRising = false;
            }
            prevCorr = corr;
          }

          if (bestLag > 0) {
            const pitchHz = sampleRate / bestLag;
            onPitchUpdate({ pitchHz: Number(pitchHz.toFixed(1)), rms });
          } else {
            onPitchUpdate({ pitchHz: 0, rms });
          }
        } else {
          onPitchUpdate({ pitchHz: 0, rms });
        }
      }

      // Draw waveform
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#FDCB6E'; // Gold accent
      ctx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = floatData[i];
        const y = (0.5 + v * 0.45) * height;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();

      // Glow effect
      ctx.lineWidth = 5;
      ctx.strokeStyle = 'rgba(253, 203, 110, 0.25)';
      ctx.beginPath();
      x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = floatData[i];
        const y = (0.5 + v * 0.45) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    draw();
  }

  /**
   * Stop live waveform animation.
   */
  stopLiveWaveform() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Clean up resources.
   */
  destroy() {
    this.stopLiveWaveform();
    clearInterval(this.timerInterval);

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }

  /**
   * Get a supported audio MIME type.
   */
  getSupportedMimeType() {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return 'audio/webm';
  }

  /**
   * Format milliseconds to MM:SS.ms
   */
  formatTime(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const centiseconds = Math.floor((ms % 1000) / 10);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
  }
}
