/**
 * NoteScope — Piano Roll Renderer
 * High-performance Canvas-based piano roll visualization.
 */

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_KEYS = [1, 3, 6, 8, 10]; // Indices of sharps/flats

// Color palette for notes based on pitch range
const NOTE_COLORS = {
  low:  { fill: 'rgba(0, 206, 201, 0.7)',  stroke: 'rgba(0, 206, 201, 0.9)',  glow: 'rgba(0, 206, 201, 0.3)' },  // Teal
  mid:  { fill: 'rgba(108, 92, 231, 0.7)', stroke: 'rgba(108, 92, 231, 0.9)', glow: 'rgba(108, 92, 231, 0.3)' }, // Purple
  high: { fill: 'rgba(253, 121, 168, 0.7)', stroke: 'rgba(253, 121, 168, 0.9)', glow: 'rgba(253, 121, 168, 0.3)' }, // Pink
};

export class PianoRoll {
  constructor(options = {}) {
    this.container = options.container;
    this.onSeek = options.onSeek || null;
    this.notes = [];
    this.duration = 0;
    this.currentTime = 0;

    // Rendering config
    this.pixelsPerSecond = options.pixelsPerSecond || 80;
    this.noteHeight = options.noteHeight || 10;
    this.pianoKeyWidth = 52;
    this.headerHeight = 0;

    // MIDI range
    this.minMidi = 21;  // A0
    this.maxMidi = 108; // C8

    // Canvas elements
    this.keysCanvas = null;
    this.rollCanvas = null;
    this.playhead = null;

    // State
    this.hoveredNote = null;
    this.dpr = window.devicePixelRatio || 1;
  }

  /**
   * Initialize the piano roll with detected notes.
   * @param {Array} notes - Array of note objects from detector
   * @param {number} duration - Total audio duration in seconds
   */
  init(notes, duration) {
    this.notes = notes;
    this.duration = duration;

    // Calculate MIDI range from actual notes
    if (notes.length > 0) {
      const midiValues = notes.map(n => n.midi);
      this.minMidi = Math.max(21, Math.min(...midiValues) - 4);
      this.maxMidi = Math.min(108, Math.max(...midiValues) + 4);
    }

    this.totalKeys = this.maxMidi - this.minMidi + 1;
    this.totalHeight = this.totalKeys * this.noteHeight;
    this.totalWidth = Math.max(800, this.duration * this.pixelsPerSecond);

    this.render();
  }

  /**
   * Build and render the piano roll.
   */
  render() {
    if (!this.container) return;

    // Clear
    this.container.innerHTML = '';

    // Create wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'piano-roll-wrapper';
    wrapper.id = 'piano-roll-wrapper';

    const canvasContainer = document.createElement('div');
    canvasContainer.className = 'piano-roll-canvas-container';
    canvasContainer.style.height = `${this.totalHeight}px`;
    canvasContainer.style.width = `${this.pianoKeyWidth + this.totalWidth}px`;

    // Piano keys canvas
    this.keysCanvas = document.createElement('canvas');
    this.keysCanvas.className = 'piano-keys-canvas';
    this.keysCanvas.width = this.pianoKeyWidth * this.dpr;
    this.keysCanvas.height = this.totalHeight * this.dpr;
    this.keysCanvas.style.width = `${this.pianoKeyWidth}px`;
    this.keysCanvas.style.height = `${this.totalHeight}px`;

    // Roll canvas
    this.rollCanvas = document.createElement('canvas');
    this.rollCanvas.className = 'piano-roll-canvas';
    this.rollCanvas.width = this.totalWidth * this.dpr;
    this.rollCanvas.height = this.totalHeight * this.dpr;
    this.rollCanvas.style.width = `${this.totalWidth}px`;
    this.rollCanvas.style.height = `${this.totalHeight}px`;

    // Playhead
    this.playhead = document.createElement('div');
    this.playhead.className = 'piano-roll-playhead';
    this.playhead.style.left = `${this.pianoKeyWidth}px`;

    canvasContainer.appendChild(this.keysCanvas);
    canvasContainer.appendChild(this.rollCanvas);
    canvasContainer.appendChild(this.playhead);
    wrapper.appendChild(canvasContainer);
    this.container.appendChild(wrapper);

    // Draw
    this.drawPianoKeys();
    this.drawNoteRoll();

    // Tooltip on hover
    this.rollCanvas.addEventListener('mousemove', (e) => this.handleHover(e));
    this.rollCanvas.addEventListener('mouseleave', () => this.clearHover());

    // Click on canvas to seek
    this.rollCanvas.addEventListener('click', (e) => {
      const rect = this.rollCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const time = Math.max(0, Math.min(this.duration, x / this.pixelsPerSecond));
      if (this.onSeek) {
        this.onSeek(time);
      }
    });
  }

  /**
   * Draw the piano keyboard on the left side.
   */
  drawPianoKeys() {
    const ctx = this.keysCanvas.getContext('2d');
    ctx.scale(this.dpr, this.dpr);

    for (let midi = this.maxMidi; midi >= this.minMidi; midi--) {
      const noteIndex = midi % 12;
      const isBlack = BLACK_KEYS.includes(noteIndex);
      const y = (this.maxMidi - midi) * this.noteHeight;

      // Key background
      ctx.fillStyle = isBlack ? '#1A1A25' : '#222233';
      ctx.fillRect(0, y, this.pianoKeyWidth, this.noteHeight);

      // Border
      ctx.fillStyle = 'rgba(255,255,255,0.04)';
      ctx.fillRect(0, y + this.noteHeight - 1, this.pianoKeyWidth, 1);

      // Note label (only for C notes and every other note if zoomed)
      if (noteIndex === 0 || this.noteHeight >= 14) {
        const octave = Math.floor(midi / 12) - 1;
        const noteName = NOTE_NAMES[noteIndex];
        const label = `${noteName}${octave}`;

        ctx.fillStyle = noteIndex === 0 ? '#8888AA' : '#555577';
        ctx.font = `${Math.min(9, this.noteHeight - 2)}px Inter, sans-serif`;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, this.pianoKeyWidth - 6, y + this.noteHeight / 2);
      }
    }

    // Right border
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.fillRect(this.pianoKeyWidth - 1, 0, 1, this.totalHeight);
  }

  /**
   * Draw the note roll (grid + note blocks).
   */
  drawNoteRoll() {
    const ctx = this.rollCanvas.getContext('2d');
    ctx.scale(this.dpr, this.dpr);

    // Background
    ctx.fillStyle = '#0F0F18';
    ctx.fillRect(0, 0, this.totalWidth, this.totalHeight);

    // Draw grid lines
    this.drawGrid(ctx);

    // Draw notes
    for (const note of this.notes) {
      this.drawNote(ctx, note);
    }
  }

  /**
   * Draw the background grid.
   */
  drawGrid(ctx) {
    // Horizontal lines (per MIDI note)
    for (let midi = this.maxMidi; midi >= this.minMidi; midi--) {
      const y = (this.maxMidi - midi) * this.noteHeight;
      const noteIndex = midi % 12;
      const isBlack = BLACK_KEYS.includes(noteIndex);

      // Alternating row colors
      ctx.fillStyle = isBlack ? 'rgba(255,255,255,0.015)' : 'rgba(255,255,255,0.025)';
      ctx.fillRect(0, y, this.totalWidth, this.noteHeight);

      // C note lines (octave boundaries)
      if (noteIndex === 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.06)';
        ctx.fillRect(0, y, this.totalWidth, 1);
      }
    }

    // Vertical lines (time markers)
    const interval = this.getTimeGridInterval();

    for (let t = 0; t <= this.duration; t += interval) {
      const x = t * this.pixelsPerSecond;
      const isWholeSecond = Math.abs(t - Math.round(t)) < 0.01;

      ctx.fillStyle = isWholeSecond ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.025)';
      ctx.fillRect(x, 0, 1, this.totalHeight);

      // Time label
      if (isWholeSecond && t > 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(this.formatTime(t), x, 10);
      }
    }
  }

  /**
   * Draw a single note block.
   */
  drawNote(ctx, note) {
    const x = note.startTime * this.pixelsPerSecond;
    const width = Math.max(3, note.duration * this.pixelsPerSecond);
    const y = (this.maxMidi - note.midi) * this.noteHeight;
    const height = this.noteHeight - 1;

    // Color based on pitch range
    const colors = this.getNoteColor(note.midi);

    // Glow (behind)
    ctx.shadowColor = colors.glow;
    ctx.shadowBlur = 6;

    // Note body
    ctx.fillStyle = colors.fill;
    ctx.beginPath();
    const radius = Math.min(3, height / 2);
    if (ctx.roundRect) {
      ctx.roundRect(x, y, width, height, radius);
    } else {
      ctx.rect(x, y, width, height);
    }
    ctx.fill();

    // Border
    ctx.shadowBlur = 0;
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Note name label (if wide enough)
    if (width > 25) {
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.font = `bold ${Math.min(8, height - 2)}px Inter, sans-serif`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(note.fullName, x + 4, y + height / 2);
    }
  }

  /**
   * Get color for a note based on its MIDI value.
   */
  getNoteColor(midi) {
    if (midi < 52) return NOTE_COLORS.low;   // Below E3
    if (midi < 72) return NOTE_COLORS.mid;   // E3 to B4
    return NOTE_COLORS.high;                  // C5 and above
  }

  /**
   * Update playhead position.
   * @param {number} time - Current time in seconds
   */
  updatePlayhead(time) {
    if (!this.playhead) return;
    this.currentTime = time;
    const x = this.pianoKeyWidth + time * this.pixelsPerSecond;
    this.playhead.style.left = `${x}px`;

    // Auto-scroll to keep playhead visible
    const wrapper = document.getElementById('piano-roll-wrapper');
    if (wrapper) {
      const scrollLeft = wrapper.scrollLeft;
      const visibleWidth = wrapper.clientWidth;
      const playheadX = time * this.pixelsPerSecond;

      if (playheadX > scrollLeft + visibleWidth - 100 || playheadX < scrollLeft) {
        wrapper.scrollLeft = playheadX - 100;
      }
    }
  }

  /**
   * Handle mouse hover on notes.
   */
  handleHover(e) {
    const rect = this.rollCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const time = x / this.pixelsPerSecond;
    const midi = this.maxMidi - Math.floor(y / this.noteHeight);

    const hoveredNote = this.notes.find(n =>
      time >= n.startTime && time <= n.endTime && n.midi === midi
    );

    if (hoveredNote !== this.hoveredNote) {
      this.hoveredNote = hoveredNote;
      this.rollCanvas.style.cursor = hoveredNote ? 'pointer' : 'default';
      this.rollCanvas.title = hoveredNote
        ? `${hoveredNote.fullName} (${hoveredNote.startTime.toFixed(2)}s - ${hoveredNote.endTime.toFixed(2)}s)`
        : '';
    }
  }

  /**
   * Clear hover state.
   */
  clearHover() {
    this.hoveredNote = null;
    this.rollCanvas.style.cursor = 'default';
    this.rollCanvas.title = '';
  }

  /**
   * Get appropriate time grid interval based on zoom level.
   */
  getTimeGridInterval() {
    if (this.pixelsPerSecond >= 100) return 0.25;
    if (this.pixelsPerSecond >= 50) return 0.5;
    if (this.pixelsPerSecond >= 20) return 1;
    return 2;
  }

  /**
   * Format seconds to M:SS.
   */
  formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  }
}
