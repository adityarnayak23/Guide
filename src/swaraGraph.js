/**
 * NoteScope — Indian Swara & Vocal Pitch Canvas Renderer
 * 
 * Renders:
 * 1. Dedicated left gutter with Indian Swara labels (Sa, Re, Ga, Ma, Pa...) and target frequencies in Hz.
 * 2. Continuous vocal pitch curve (in Hz) capturing slides (meend), gamak, and vibrato.
 * 3. Quantized Swara blocks with note names and duration.
 * 4. Interactive playhead and click-to-seek support.
 */

import { SWARA_DEFINITIONS } from './vocalPitchTracker.js';

export class SwaraGraph {
  constructor(options = {}) {
    this.container = options.container;
    this.onSeek = options.onSeek || null;
    this.tonicFreq = options.tonicFreq || 146.8;

    this.timeSteps = [];
    this.pitches = [];
    this.swaraNotes = [];
    this.duration = 0;
    this.currentTime = 0;

    // Gutter width for Swara labels on left
    this.labelWidth = 95;
    this.pixelsPerSecond = 90;
    this.dpr = window.devicePixelRatio || 1;
    this.canvas = null;
    this.playheadEl = null;

    // Pitch display bounds (in cents relative to Sa)
    this.minCents = -700;  // Mandra Pa
    this.maxCents = 1900;  // Taar Pa
  }

  /**
   * Initialize with analysis data
   */
  init(timeSteps, pitches, swaraNotes, duration, tonicFreq) {
    this.timeSteps = timeSteps;
    this.pitches = pitches;
    this.swaraNotes = swaraNotes;
    this.duration = Math.max(1, duration);
    this.tonicFreq = tonicFreq;

    // Dynamically adjust cents range based on actual pitches detected
    const validCents = [];
    for (const p of pitches) {
      if (p > 65) {
        validCents.push(1200 * Math.log2(p / tonicFreq));
      }
    }

    if (validCents.length > 0) {
      const minC = Math.min(...validCents);
      const maxC = Math.max(...validCents);
      this.minCents = Math.floor((minC - 300) / 100) * 100;
      this.maxCents = Math.ceil((maxC + 300) / 100) * 100;
    }

    this.render();
  }

  /**
   * Build DOM and draw
   */
  render() {
    if (!this.container) return;
    this.container.innerHTML = '';

    const graphHeight = 360;
    const containerWidth = this.container.clientWidth || 1000;
    this.pixelsPerSecond = Math.max(90, (containerWidth - this.labelWidth) / this.duration);
    const graphWidth = Math.ceil(this.labelWidth + this.duration * this.pixelsPerSecond);

    const wrapper = document.createElement('div');
    wrapper.className = 'swara-graph-wrapper';
    wrapper.id = 'swara-graph-wrapper';

    const canvasContainer = document.createElement('div');
    canvasContainer.className = 'swara-canvas-container';
    canvasContainer.style.width = `${graphWidth}px`;
    canvasContainer.style.height = `${graphHeight}px`;
    canvasContainer.style.position = 'relative';

    this.canvas = document.createElement('canvas');
    this.canvas.className = 'swara-canvas';
    this.canvas.width = graphWidth * this.dpr;
    this.canvas.height = graphHeight * this.dpr;
    this.canvas.style.width = `${graphWidth}px`;
    this.canvas.style.height = `${graphHeight}px`;

    // Playhead line
    this.playheadEl = document.createElement('div');
    this.playheadEl.className = 'swara-playhead';
    this.playheadEl.style.left = `${this.labelWidth}px`;

    canvasContainer.appendChild(this.canvas);
    canvasContainer.appendChild(this.playheadEl);
    wrapper.appendChild(canvasContainer);
    this.container.appendChild(wrapper);

    // Click on canvas to seek
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const clickedTime = Math.max(0, Math.min(this.duration, (x - this.labelWidth) / this.pixelsPerSecond));
      if (this.onSeek) {
        this.onSeek(clickedTime);
      }
    });

    this.draw(graphWidth, graphHeight);
  }

  /**
   * Main draw call
   */
  draw(width, height) {
    const ctx = this.canvas.getContext('2d');
    ctx.scale(this.dpr, this.dpr);

    // Dark Indian classical aesthetic background
    ctx.fillStyle = '#0b0b12';
    ctx.fillRect(0, 0, width, height);

    // Gutter background on the left for Swara labels
    ctx.fillStyle = '#11111c';
    ctx.fillRect(0, 0, this.labelWidth, height);

    // Gutter divider
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(253, 203, 110, 0.25)';
    ctx.lineWidth = 1;
    ctx.moveTo(this.labelWidth, 0);
    ctx.lineTo(this.labelWidth, height);
    ctx.stroke();

    // 1. Draw Swara Horizontal Grid
    this.drawSwaraGrid(ctx, width, height);

    // 2. Draw Time Vertical Grid Lines
    this.drawTimeGrid(ctx, width, height);

    // 3. Draw Detected Swara Blocks
    this.drawSwaraBlocks(ctx, width, height);

    // 4. Draw Continuous Vocal Pitch Curve (Hz)
    this.drawVocalPitchCurve(ctx, width, height);
  }

  /**
   * Convert cents relative to Sa into Y canvas coordinate
   */
  centsToY(cents, height) {
    const totalRange = this.maxCents - this.minCents;
    const normalized = (cents - this.minCents) / totalRange;
    return height - (normalized * (height - 40) + 20);
  }

  /**
   * Draw horizontal Swara guide lines across all Saptaks
   */
  drawSwaraGrid(ctx, width, height) {
    const startOct = Math.floor(this.minCents / 1200);
    const endOct = Math.ceil(this.maxCents / 1200);

    for (let oct = startOct; oct <= endOct; oct++) {
      for (const swara of SWARA_DEFINITIONS) {
        const swaraCents = swara.cents + oct * 1200;
        if (swaraCents < this.minCents || swaraCents > this.maxCents) continue;

        const y = this.centsToY(swaraCents, height);
        const isSa = swara.symbol === 'S';
        const isPa = swara.symbol === 'P';
        const isAchala = isSa || isPa;

        // Swara guide line in audio timeline area
        ctx.beginPath();
        if (isSa) {
          ctx.strokeStyle = 'rgba(253, 203, 110, 0.45)'; // Saffron/Gold for Sa
          ctx.lineWidth = 1.5;
        } else if (isPa) {
          ctx.strokeStyle = 'rgba(0, 206, 201, 0.35)'; // Cyan for Pa
          ctx.lineWidth = 1.2;
        } else if (swara.type === 'shuddha') {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
          ctx.lineWidth = 0.8;
        } else {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.035)';
          ctx.lineWidth = 0.6;
        }

        ctx.moveTo(this.labelWidth, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        // Swara label in left gutter
        let saptakMark = '';
        if (oct < 0) saptakMark = '̣';
        else if (oct > 0) saptakMark = '̇';

        const label = `${swara.name.split(' ')[0]}${saptakMark}`;
        const targetFreq = this.tonicFreq * Math.pow(2, swaraCents / 1200);

        ctx.font = isAchala ? 'bold 11px Inter, sans-serif' : '10px Inter, sans-serif';
        ctx.fillStyle = isSa
          ? '#FDCB6E'
          : isPa
          ? '#00CEC9'
          : swara.type === 'shuddha'
          ? 'rgba(255,255,255,0.75)'
          : 'rgba(255,255,255,0.35)';

        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${label}`, 8, y);

        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.textAlign = 'right';
        ctx.fillText(`${Math.round(targetFreq)}Hz`, this.labelWidth - 8, y);
      }
    }
  }

  /**
   * Draw vertical time marks
   */
  drawTimeGrid(ctx, width, height) {
    const stepSeconds = this.pixelsPerSecond >= 80 ? 0.5 : 1.0;

    for (let t = 0; t <= this.duration; t += stepSeconds) {
      const x = this.labelWidth + t * this.pixelsPerSecond;
      const isSecond = Math.abs(t - Math.round(t)) < 0.01;

      ctx.beginPath();
      ctx.strokeStyle = isSecond ? 'rgba(255, 255, 255, 0.07)' : 'rgba(255, 255, 255, 0.025)';
      ctx.lineWidth = 1;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      if (isSecond && t > 0) {
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.textAlign = 'center';
        ctx.fillText(`${t}s`, x, height - 6);
      }
    }
  }

  /**
   * Draw quantized Swara blocks
   */
  drawSwaraBlocks(ctx, width, height) {
    for (const note of this.swaraNotes) {
      const x = this.labelWidth + note.startTime * this.pixelsPerSecond;
      const blockWidth = Math.max(4, note.duration * this.pixelsPerSecond);

      const cents = 1200 * Math.log2(note.meanHz / this.tonicFreq);
      const y = this.centsToY(cents, height);

      const isSa = note.swara.symbol === 'S';
      const isPa = note.swara.symbol === 'P';

      // Swara Block Background
      ctx.fillStyle = isSa
        ? 'rgba(253, 203, 110, 0.22)'
        : isPa
        ? 'rgba(0, 206, 201, 0.18)'
        : 'rgba(108, 92, 231, 0.18)';

      const blockHeight = 16;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x, y - blockHeight / 2, blockWidth, blockHeight, 3);
      } else {
        ctx.rect(x, y - blockHeight / 2, blockWidth, blockHeight);
      }
      ctx.fill();

      // Border
      ctx.strokeStyle = isSa
        ? 'rgba(253, 203, 110, 0.7)'
        : isPa
        ? 'rgba(0, 206, 201, 0.6)'
        : 'rgba(108, 92, 231, 0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Swara Name Label
      if (blockWidth > 20) {
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.fillStyle = isSa ? '#FDCB6E' : isPa ? '#00CEC9' : '#FFFFFF';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(note.formatted, x + 4, y);
      }
    }
  }

  /**
   * Draw continuous vocal pitch trajectory (Hz)
   * Captures vocal slides (meend), gamak, and intonation curve
   */
  drawVocalPitchCurve(ctx, width, height) {
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Glowing pitch curve
    ctx.shadowColor = 'rgba(253, 121, 168, 0.6)';
    ctx.shadowBlur = 8;
    ctx.strokeStyle = '#FD79A8'; // Vibrant singing line

    ctx.beginPath();
    let isDrawing = false;

    for (let i = 0; i < this.pitches.length; i++) {
      const p = this.pitches[i];
      const t = this.timeSteps[i];

      if (p > 65) {
        const cents = 1200 * Math.log2(p / this.tonicFreq);
        const x = this.labelWidth + t * this.pixelsPerSecond;
        const y = this.centsToY(cents, height);

        if (!isDrawing) {
          ctx.moveTo(x, y);
          isDrawing = true;
        } else {
          ctx.lineTo(x, y);
        }
      } else {
        if (isDrawing) {
          ctx.stroke();
          ctx.beginPath();
          isDrawing = false;
        }
      }
    }

    if (isDrawing) {
      ctx.stroke();
    }

    ctx.shadowBlur = 0;
  }

  /**
   * Synchronize playhead position
   */
  updatePlayhead(time) {
    if (!this.playheadEl) return;
    this.currentTime = time;
    const x = this.labelWidth + time * this.pixelsPerSecond;
    this.playheadEl.style.left = `${x}px`;

    // Auto-scroll to keep playhead centered
    const wrapper = document.getElementById('swara-graph-wrapper');
    if (wrapper) {
      const scrollLeft = wrapper.scrollLeft;
      const visibleWidth = wrapper.clientWidth;

      if (x > scrollLeft + visibleWidth - 120 || x < scrollLeft) {
        wrapper.scrollLeft = Math.max(0, x - 120);
      }
    }
  }
}
