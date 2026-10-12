/**
 * NoteScope — Luminous Golden Audio Waveform with Ethereal Beat Pulse Beads
 * 
 * Renders the exact glowing waveform from the reference artwork:
 * - Obsidian background with golden gradient symmetrical audio bars
 * - Centerline dotted baseline
 * - Luminous glowing bead nodes (•) at every Akshara beat onset
 * - Real-time playhead with radial bloom pulse on beat strikes
 * - Interactive scrubbing and seeking
 */

export class LuminousWaveform {
  constructor(canvasContainerId, options = {}) {
    this.container = typeof canvasContainerId === 'string' ? document.getElementById(canvasContainerId) : canvasContainerId;
    if (!this.container) return;

    this.options = {
      goldColor: '#9B2C2C',
      goldMuted: 'rgba(155, 44, 44, 0.35)',
      goldGlow: 'rgba(155, 44, 44, 0.3)',
      barWidth: 2,
      barGap: 2,
      onSeek: options.onSeek || (() => {}),
      ...options,
    };

    this.audioBuffer = null;
    this.peaks = [];
    this.beatTimestamps = [];
    this.currentTime = 0;
    this.duration = 1;
    this.activeBeatIndex = -1;
    this.activeNote = null;

    this.initCanvas();
    this.initEvents();
  }

  setActiveNote(note) {
    this.activeNote = note;
  }

  initCanvas() {
    this.container.innerHTML = `
      <div class="luminous-waveform-wrapper">
        <canvas class="luminous-waveform-canvas"></canvas>
      </div>
    `;
    this.canvas = this.container.querySelector('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.resize();

    window.addEventListener('resize', () => {
      this.resize();
      this.draw();
    });
  }

  resize() {
    if (!this.canvas || !this.container) return;
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || 800;
    this.height = 90;

    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;

    this.ctx.scale(dpr, dpr);
  }

  setAudioBuffer(audioBuffer, beatTimestamps = []) {
    this.audioBuffer = audioBuffer;
    this.duration = audioBuffer ? audioBuffer.duration : 1;
    this.beatTimestamps = beatTimestamps;
    this.extractPeaks();
    this.draw();
  }

  setBeatTimestamps(timestamps) {
    this.beatTimestamps = timestamps;
    this.draw();
  }

  extractPeaks() {
    if (!this.audioBuffer) {
      this.peaks = [];
      return;
    }

    const channelData = this.audioBuffer.getChannelData(0);
    const totalBars = Math.floor(this.width / (this.options.barWidth + this.options.barGap));
    const step = Math.floor(channelData.length / totalBars);
    this.peaks = new Float32Array(totalBars);

    for (let i = 0; i < totalBars; i++) {
      const start = i * step;
      let max = 0;
      for (let j = 0; j < step; j += 4) {
        const val = Math.abs(channelData[start + j] || 0);
        if (val > max) max = val;
      }
      this.peaks[i] = max;
    }
  }

  setTime(time) {
    this.currentTime = Math.max(0, Math.min(this.duration, time));
    this.draw();
  }

  draw() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // 1. Draw Dotted Centerline Baseline (Charred Sepia Hairline)
    ctx.save();
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(155, 44, 44, 0.22)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();
    ctx.restore();

    // 2. Draw Symmetrical Manuscript Audio Bars
    if (this.peaks && this.peaks.length > 0) {
      const barW = this.options.barWidth;
      const gap = this.options.barGap;
      const playProgressX = (this.currentTime / this.duration) * w;

      for (let i = 0; i < this.peaks.length; i++) {
        const x = i * (barW + gap);
        const amp = this.peaks[i];
        const barH = Math.max(2, amp * (h * 0.42));
        const isPast = x <= playProgressX;

        // Gradient for bars (Kumkum Terracotta to Turmeric Gold)
        const grad = ctx.createLinearGradient(x, midY - barH, x, midY + barH);
        if (isPast) {
          grad.addColorStop(0, 'rgba(155, 44, 44, 0.92)');
          grad.addColorStop(0.5, 'rgba(192, 125, 30, 0.85)');
          grad.addColorStop(1, 'rgba(155, 44, 44, 0.92)');
        } else {
          grad.addColorStop(0, 'rgba(94, 81, 72, 0.28)');
          grad.addColorStop(0.5, 'rgba(192, 125, 30, 0.22)');
          grad.addColorStop(1, 'rgba(94, 81, 72, 0.28)');
        }

        ctx.fillStyle = grad;
        ctx.beginPath();
        // Symmetrical top and bottom bars with rounded tips
        ctx.roundRect(x, midY - barH, barW, barH * 2, barW / 2);
        ctx.fill();
      }
    }

    // 3. Draw Radiant Beat Beads (•) at every Akshara timestamp
    if (this.beatTimestamps && this.beatTimestamps.length > 0) {
      this.beatTimestamps.forEach((t) => {
        const x = (t / this.duration) * w;
        const isCurrentBeat = Math.abs(this.currentTime - t) < 0.18;

        ctx.save();
        if (isCurrentBeat) {
          // Intense radiant bloom glow on active beat
          ctx.shadowColor = 'rgba(155, 44, 44, 0.6)';
          ctx.shadowBlur = 10;

          // Outer aura
          ctx.fillStyle = 'rgba(155, 44, 44, 0.35)';
          ctx.beginPath();
          ctx.arc(x, midY, 6.5, 0, Math.PI * 2);
          ctx.fill();

          // Bright center bead
          ctx.fillStyle = '#9B2C2C';
          ctx.beginPath();
          ctx.arc(x, midY, 3, 0, Math.PI * 2);
          ctx.fill();
        } else {
          // Warm resting terracotta bead
          ctx.shadowColor = 'rgba(155, 44, 44, 0.2)';
          ctx.shadowBlur = 3;
          ctx.fillStyle = 'rgba(155, 44, 44, 0.75)';
          ctx.beginPath();
          ctx.arc(x, midY, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      });
    }

    // 4. Draw Playhead Needle and Vocal Glow Resonance Aura
    const playX = (this.currentTime / this.duration) * w;
    ctx.save();
    if (this.activeNote) {
      // Golden vocal resonance bloom
      ctx.shadowColor = 'rgba(212, 175, 55, 0.9)';
      ctx.shadowBlur = 16;
      ctx.fillStyle = 'rgba(212, 175, 55, 0.35)';
      ctx.beginPath();
      ctx.arc(playX, midY, 8, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#FFE066';
      ctx.beginPath();
      ctx.arc(playX, midY, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = this.activeNote ? '#D4AF37' : '#9B2C2C';
    ctx.lineWidth = this.activeNote ? 2.5 : 2;
    ctx.shadowColor = this.activeNote ? 'rgba(212, 175, 55, 0.75)' : 'rgba(155, 44, 44, 0.35)';
    ctx.shadowBlur = this.activeNote ? 10 : 6;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, h);
    ctx.stroke();
    ctx.restore();
  }

  initEvents() {
    if (!this.canvas) return;

    const handleSeek = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / rect.width));
      const seekTime = pct * this.duration;
      this.setTime(seekTime);
      this.options.onSeek(seekTime);
    };

    let isDragging = false;

    this.canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      handleSeek(e);
    });

    window.addEventListener('mousemove', (e) => {
      if (isDragging) handleSeek(e);
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length > 0) {
        handleSeek(e.touches[0]);
      }
    }, { passive: true });
  }
}
