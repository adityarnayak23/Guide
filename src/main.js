/**
 * NoteScope — Indian Vocal Pitch & Swara Recognition (V2 Revamped)
 * 
 * Specifically designed for singing/vocals:
 * - 100% Automatic Tonic (Sa) Identification directly from vocal pitch distribution (NO default pitch!)
 * - Deep Neural Note Recognition via Spotify Basic Pitch model for clean, non-jittery note boundaries
 * - Cinema-grade Subtitle Display showing sung notes in real-time with Devanagari glyphs & tuning meter
 * - Horizontal Sung Notes Track directly under the waveform, aligned 1-to-1 with the audio timeline
 * - Continuous vocal pitch curve (Hz) capturing meend (slides), gamak, and intonation
 * - Live real-time singing subtitle feedback during microphone recording
 */

import './style.css';
import { AudioRecorder } from './recorder.js';
import {
  recognizeVocalNotes,
  pitchToLiveSwara,
  generateSargamTranscript,
  hzToWesternNote
} from './vocalRecognitionEngine.js';
import {
  CARNATIC_TALAS,
  detectCarnaticTala,
  buildTalaGrid,
  buildAdiTalaGrid,
  buildFreeFlowPhrases,
  detectBeatStructure,
  detectAudioBpm,
  identifyCarnaticRaga,
  formatCarnaticSwara,
  exportCarnaticTextNotation
} from './carnaticNotation.js';
import { tanpura, TANPURA_SCALES } from './tanpuraSynth.js';
import { CarnaticHandStage } from './carnaticHandStage.js';
import { LuminousWaveform } from './luminousWaveform.js';
import { TalaHandAnimator } from './talaHand.js';
import WaveSurfer from 'wavesurfer.js';

// ── State ───────────────────────────────────────────────────────────────────
const state = {
  audioBlob: null,
  audioBuffer: null,
  timeSteps: [],
  pitches: [],
  confidences: [],
  swaraNotes: [],
  tonic: null,
  customTonicFreq: null,
  selectedRoot: 'D',
  selectedOctave: 3,
  isAutoSa: true,
  currentTheme: 'manuscript',
  liveRunSwaras: [],
  wavesurfer: null,
  recorder: null,
  isPlaying: false,
  activeNoteIndex: -1,
  topHandStage: null,
  handStage: null,
  luminousWaveform: null,
  talaHandAnimator: null,
  activeTalaId: 'adi',
  talaMode: 'auto', // 'auto' or 'manual'
  autoDetectedTala: null,
  talaData: null,
  adiTalaData: null,
  freeFlowData: null,
  beatAnalysis: null,
  raga: null,
  carnaticMode: 'tala', // 'tala' or 'freeflow'
  scoreLineWindow: 0, // 0-based page index for 4-line stage
  detectedBpm: 76,
  detectedKala: 'Madhyama Kala',
  activeTempoMode: 'auto', // 'auto', '56', '76', '108'
  rafLoopId: null, // requestAnimationFrame handle for 60FPS playback sync
};


// ── DOM Layout ──────────────────────────────────────────────────────────────
function createApp() {
  const app = document.getElementById('app');
  app.innerHTML = `
    <!-- Header -->
    <header class="header">
      <div class="logo">
        <div class="logo-icon">🪕</div>
        <div class="logo-title-group">
          <span class="logo-text">NoteScope</span>
          <span class="logo-subtext">Carnatic Vocal &amp; Tala Archive</span>
        </div>
        <span class="logo-badge">Carnatic Swara Archive</span>
      </div>
      <div class="header-actions">
        <button class="btn" id="btn-about" title="How Root Sa & Swara Detection Works">
          <span>ℹ️</span> How it Works
        </button>
      </div>
    </header>

    <!-- Hero (Sharp, Compact, World-Class) -->
    <section class="hero" id="hero-section">
      <h1>Vocal Swara &amp; Notation Scribe</h1>
      <p class="hero-sub">Real-time pitch intonation, root tonic detection, and authentic manuscript notation.</p>
    </section>

    <!-- Compact Dual Input Dock (Upload & Sing) -->
    <section class="input-zone" id="input-zone">
      <!-- Upload Card -->
      <div class="input-card" id="upload-card">
        <input type="file" class="file-input" id="file-input" accept="audio/*" />
        
        <div class="card-header-row">
          <div class="card-icon-circle">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
          </div>
          <div class="card-text-block">
            <h3>Upload Audio File</h3>
            <span class="card-sub-hint">WAV, MP3, M4A · Drop audio or click to browse</span>
          </div>
        </div>

        <div class="card-action-row">
          <button class="upload-browse-btn" type="button">Choose Audio File</button>
          <button class="sample-take-btn" id="btn-load-demo" type="button" title="Load sample vocal take">✦ Load Sample Take</button>
        </div>

        <div class="file-info" id="file-info">
          <span class="file-dot">●</span>
          <span class="file-name" id="file-name"></span>
          <span class="file-size-badge" id="file-size"></span>
        </div>
      </div>

      <!-- Record Card -->
      <div class="input-card" id="record-card">
        <div class="card-header-row">
          <div class="card-icon-circle">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
          </div>
          <div class="card-text-block">
            <h3>Sing with Microphone</h3>
            <span class="card-sub-hint">Sing in your natural pitch with live swara detection</span>
          </div>
        </div>

        <div class="card-action-row recording-controls" id="recording-controls">
          <button class="record-btn-compact" id="record-btn" title="Start/Stop Recording" type="button"></button>
          
          <div class="recording-meta">
            <span class="recording-timer" id="recording-timer">00:00.00</span>
            <span class="recording-label" id="recording-label">Tap to sing</span>
          </div>

          <div class="live-visualizer-compact" id="live-visualizer">
            <canvas id="live-canvas" width="130" height="24"></canvas>
          </div>

          <div class="live-singing-subtitle" id="live-singing-subtitle">
            <span class="live-sub-tag">LIVE</span>
            <div class="live-sub-content" id="live-sub-content">
              <span class="live-sub-placeholder">—</span>
            </div>
          </div>
        </div>

        <div class="file-info" id="recording-info">
          <span class="file-dot">●</span>
          <span class="file-name" id="recording-name">Vocal Take</span>
          <span class="file-size-badge" id="recording-duration"></span>
        </div>

        <!-- Silent backing elements for liveRun state hooks -->
        <div id="live-accumulating-run" style="display: none;">
          <span id="live-run-counter"></span>
          <div id="live-run-tokens"></div>
        </div>
      </div>
    </section>

    <!-- "Your Sa" Calibration & Tuning Bar -->
    <div class="sa-calibration-bar" id="sa-calibration-bar">
      <div class="sa-calib-label-group">
        <span class="sa-calib-icon">🪕</span>
        <div>
          <div class="sa-calib-title">Your Sa (Adhara Shadja)</div>
          <div class="sa-calib-sub">Set root tonic or use automatic pitch detection</div>
        </div>
      </div>
      <div class="sa-calib-controls">
        <button class="sa-btn-auto active" id="btn-sa-auto" title="Auto-Detect Sa from singing vocal distribution">
          ✦ Auto-Detect Sa
        </button>
        <div class="sa-select-group">
          <select class="sa-select" id="sa-root-select" title="Root Note">
            <option value="C">C</option>
            <option value="C#">C♯</option>
            <option value="D" selected>D</option>
            <option value="D#">D♯</option>
            <option value="E">E</option>
            <option value="F">F</option>
            <option value="F#">F♯</option>
            <option value="G">G</option>
            <option value="G#">G♯</option>
            <option value="A">A</option>
            <option value="A#">A♯</option>
            <option value="B">B</option>
          </select>
          <select class="sa-select" id="sa-octave-select" title="Octave">
            <option value="2">2 (Deep)</option>
            <option value="3" selected>3 (Standard)</option>
            <option value="4">4 (High)</option>
            <option value="5">5 (Upper)</option>
          </select>
        </div>
        <div class="sa-hz-pill" id="sa-hz-indicator">D3 · 146.8 Hz</div>
      </div>
    </div>

    <!-- Action Button -->
    <div class="analyze-zone" id="analyze-zone">
      <button class="analyze-btn" id="analyze-btn">
        ✦ Transcribe Swaras &amp; Inscribe Notation
      </button>
    </div>

    <!-- Processing Overlay -->
    <div class="processing-overlay" id="processing-overlay">
      <div class="processing-content">
        <div class="processing-spinner"></div>
        <div class="processing-title">Transcribing Vocal Take</div>
        <div class="processing-subtitle" id="processing-status">Extracting pitch contour...</div>
        <div class="processing-progress">
          <div class="processing-progress-bar" id="progress-bar"></div>
        </div>
      </div>
    </div>

    <!-- Results View -->
    <section class="analysis-section" id="analysis-section">
      <!-- Auto-Detected Sa Profile -->
      <div class="tonic-banner" id="tonic-banner">
        <div class="tonic-main">
          <span class="tonic-tag">ROOT SHADJA (SA)</span>
          <div class="tonic-value" id="tonic-freq">Sa = — Hz</div>
          <div class="tonic-ref" id="tonic-ref">Western Ref: —</div>
        </div>
        <div class="tonic-stats" id="tonic-stats">
          <div class="tonic-stat-item">
            <span class="label">Swaras Sung</span>
            <span class="val" id="stat-total-notes">0</span>
          </div>
          <div class="tonic-stat-item">
            <span class="label">Vocal Span</span>
            <span class="val" id="stat-vocal-span">—</span>
          </div>
          <div class="tonic-stat-item">
            <span class="label">Predominant Swara</span>
            <span class="val" id="stat-predom">—</span>
          </div>
          <div class="tonic-stat-item">
            <span class="label">Duration</span>
            <span class="val" id="stat-duration">0.0s</span>
          </div>
        </div>
      </div>

      <!-- MASTER CLASSICAL OBSIDIAN STAGE (EXACT MATCH TO REFERENCE ARTWORK) -->
      <div class="classical-obsidian-stage" id="classical-obsidian-stage">
        
        <!-- Stage Masthead Header -->
        <div class="classical-stage-header">
          <div class="stage-title-row">
            <h1 class="classical-stage-title" id="classical-stage-title">Adi Tala — Chaturasra Nadai</h1>
            <div class="tala-selector-wrap">
              <label for="stage-tala-select" class="tala-select-label">Tala:</label>
              <select id="stage-tala-select" class="classical-tala-select" title="Select Carnatic Tala or Auto-Detect">
                <option value="auto">⚡ Auto-Detect</option>
                <option value="adi" selected>Adi Tala (8 beats · 4+2+2)</option>
                <option value="rupaka">Rupaka Tala (6 beats · 2+4)</option>
                <option value="misra_chapu">Misra Chapu (7 beats · 3+4)</option>
                <option value="khanda_chapu">Khanda Chapu (5 beats · 2+3)</option>
                <option value="tisra_triputa">Tisra Triputa (7 beats · 3+2+2)</option>
                <option value="misra_jhampa">Misra Jhampa (10 beats · 7+1+2)</option>
                <option value="khanda_ata">Khanda Ata (14 beats · 5+5+2+2)</option>
                <option value="eka">Chaturasra Eka (4 beats · 4)</option>
              </select>
              <span class="tala-auto-badge" id="tala-auto-badge" style="display: none;">⚡ Auto</span>
            </div>
          </div>
          <div class="classical-stage-subtitle" id="classical-stage-subtitle">RĀ — MID · SA = 146 HZ</div>
        </div>

        <!-- Master Classical Hand Illustration Rig (with Gold Bangles & Kinetic Motion Arcs) -->
        <div class="classical-hand-canvas-wrap" id="classical-hand-canvas-wrap"></div>

        <!-- Golden Architectural Tala Score Grid (4-Line Synchronized Stage) -->
        <div class="classical-score-board" id="classical-score-board">
          <!-- 4-Line Stage Window Controls -->
          <div class="score-window-nav" id="score-window-nav" style="display: none;">
            <button class="score-nav-btn" id="score-prev-page" title="Previous 4 Lines" type="button">◀ Prev 4 Lines</button>
            <div class="score-window-indicator">
              <span class="score-window-text" id="score-window-text">Lines 1–4</span>
              <span class="score-sync-badge" id="score-sync-badge">✦ Auto-Paging in Sync</span>
            </div>
            <button class="score-nav-btn" id="score-next-page" title="Next 4 Lines" type="button">Next 4 Lines ▶</button>
          </div>

          <!-- Laghu (4) | Dhrutam (2) | Dhrutam (2) Anga Header -->
          <div class="score-anga-header">
            <div class="score-row-tag-spacer"></div>
            <div class="score-anga-col laghu-header">Laghu (4)</div>
            <div class="score-col-divider">|</div>
            <div class="score-anga-col dhrutam-header">Dhrutam (2)</div>
            <div class="score-col-divider">|</div>
            <div class="score-anga-col dhrutam-header">Dhrutam (2)</div>
          </div>
          <!-- Golden Separator Horizontal Bar -->
          <div class="score-gold-divider-line"></div>
          <!-- Swaras Rows Container (Exactly 4 Lines in a go) -->
          <div class="score-avartanams-list" id="score-avartanams-list"></div>
        </div>

        <!-- Free-Flow (Alapana) Unmetered Phrases View (when toggled) -->
        <div class="classical-freeflow-board" id="classical-freeflow-board" style="display: none;">
          <div class="freeflow-phrases-container" id="freeflow-phrases-container"></div>
        </div>

        <!-- Luminous Glowing Audio Waveform with Beat Pulse Beads -->
        <div class="classical-waveform-section">
          <div id="classical-luminous-waveform" class="classical-luminous-waveform"></div>
          
          <!-- Minimalist Floating Playback & Mode Controller -->
          <div class="classical-playback-bar">
            <button class="stage-play-btn" id="stage-play-btn" title="Play / Pause">▶</button>
            <div class="stage-timing" id="stage-timing">0:00 / 0:00</div>
            <div class="stage-tempo-pill" id="stage-tempo-pill">109 BPM</div>
            <div class="stage-mode-switcher">
              <button class="stage-mode-btn active" id="btn-stage-tala" title="Structured Adi Tala Grid">Tala Grid</button>
              <button class="stage-mode-btn" id="btn-stage-freeflow" title="Unmetered Alapana / Virutham">Free Flow</button>
            </div>
            <button class="stage-export-btn" id="btn-stage-export" title="Export Manuscript Notation">⤓ Export</button>
          </div>

          <!-- Horizontal Swaras Track (Karaoke-style note timeline) -->
          <div class="horizontal-swaras-section" id="horizontal-swaras-section" style="display:none;">
            <div class="horizontal-swaras-header">
              <span class="horizontal-swaras-title">SVARA TIMELINE</span>
              <span class="horizontal-swaras-hint">Click any swara to seek · Playing note glows</span>
            </div>
            <div class="horizontal-swaras-track-wrapper">
              <div class="horizontal-swaras-track">
                <div id="horizontal-swaras-playhead" class="horizontal-swaras-playhead" style="left:0%;"></div>
                <div id="horizontal-swaras-blocks" class="horizontal-swaras-blocks"></div>
              </div>
              <div id="horizontal-swaras-ruler" class="horizontal-swaras-ruler"></div>
            </div>
          </div>
        </div>

      </div>

      <!-- Cinema Sung Note Subtitle Card & Real-time Intonation HUD -->
      <div class="subtitle-display-card" id="subtitle-display-card">
        <div class="subtitle-card-header">
          <span class="subtitle-badge">✦ Active Swara &amp; Real-time Intonation</span>
          <span class="subtitle-timing" id="sub-pitch-hz">-- Hz</span>
        </div>
        <div class="subtitle-body">
          <div class="subtitle-swara-giant">
            <span class="sub-swara-name" id="sub-swara-name">Sa</span>
            <span class="sub-saptak-dot" id="sub-saptak-dot"></span>
            <span class="sub-swara-devanagari" id="sub-swara-devanagari">सा</span>
          </div>
          <div class="subtitle-details">
            <div class="sub-full-name" id="sub-full-name">Shadja · Madhya Sthayi</div>
            <div class="sub-meta-row">
              <span class="sub-saptak-badge" id="sub-saptak-badge">Madhya Sthayi</span>
              <span class="sub-cents-badge in-tune" id="sub-cents-badge">0¢ In-Tune</span>
              <span class="sub-gamaka-badge" id="sub-gamaka-badge" style="display: none;">Steady</span>
              <span id="sub-western-ref" style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--text-tertiary);">Western: C4</span>
            </div>
            <div class="sub-cents-meter">
              <span class="cents-label">-50¢</span>
              <div class="cents-track">
                <div class="cents-center-zone"></div>
                <div class="cents-needle" id="sub-cents-needle" style="left: 50%;"></div>
              </div>
              <span class="cents-label">+50¢</span>
            </div>
          </div>
        </div>
        <div class="subtitle-ribbon-wrapper">
          <div class="subtitle-ribbon" id="subtitle-ribbon"></div>
        </div>
      </div>

      <!-- Background audio decoding container for WaveSurfer -->
      <div id="waveform" style="display: none;"></div>

      <!-- Action Bar -->
      <div class="results-action-bar">
        <button class="btn" id="new-analysis-btn">
          <span class="btn-glyph">↺</span> Analyze Another Take
        </button>
        <div class="export-group">
          <button class="btn btn-export-midi" id="btn-export-sargam" title="Download Carnatic Notation as Text">
            <span class="btn-glyph">⤓</span> Export Manuscript Notation
          </button>
          <button class="btn btn-export-json" id="btn-export-pitch-csv" title="Download Pitch Trajectory CSV">
            <span class="btn-glyph">⤓</span> Export Pitch CSV
          </button>
        </div>
      </div>
    </section>

    <!-- About Modal -->
    <div class="modal-overlay" id="modal-about">
      <div class="modal-card">
        <div class="modal-header">
          <h3>About NoteScope Vocal &amp; Tala Architecture</h3>
          <button class="modal-close" id="btn-close-about">&times;</button>
        </div>
        <div class="modal-body">
          <h4>Vocal Swara Transcription &amp; Gamaka Tracking</h4>
          <p>NoteScope applies continuous fundamental pitch trajectory tracking specifically calibrated for Indian vocal intonation. Adaptive pitch extraction paired with acoustic onset segmentation captures nuanced gamakas, oscillations, and sustained swaras while eliminating octave ambiguities and fragmented syllables.</p>

          <h4>Automatic Tonic (Sa) Identification</h4>
          <p>Indian classical and playback singing uses a relative tonic called <strong>Sa (Adhara Shadja)</strong>. NoteScope does <strong>not</strong> assume any default pitch: it computes a folded pitch class distribution of the vocal take and correlates it against the Indian harmonic consonance template (Sa, Pa, Ma, Ga) to lock on to your natural root frequency automatically.</p>

          <h4>Horizontal Swara Track & Subtitles</h4>
          <p>Notes are displayed directly beneath the waveform timeline with proportional durations, and active notes are projected in large, glowing <strong>karaoke-style subtitles</strong> with real-time tuning meters.</p>

          <h4>The 12 Indian Swaras</h4>
          <p>All recognized notes are displayed in authentic Indian Sargam notation:</p>
          <div class="tech-badge-list">
            <span class="tech-badge">Sa (सा · Shadja)</span>
            <span class="tech-badge">re (रे · Komal Re)</span>
            <span class="tech-badge">Re (रे · Shuddha Re)</span>
            <span class="tech-badge">ga (ग · Komal Ga)</span>
            <span class="tech-badge">Ga (ग · Shuddha Ga)</span>
            <span class="tech-badge">ma (म · Shuddha Ma)</span>
            <span class="tech-badge">Ma' (म · Teevra Ma)</span>
            <span class="tech-badge">Pa (प · Pancham)</span>
            <span class="tech-badge">dha (ध · Komal Dha)</span>
            <span class="tech-badge">Dha (ध · Shuddha Dha)</span>
            <span class="tech-badge">ni (नि · Komal Ni)</span>
            <span class="tech-badge">Ni (नि · Shuddha Ni)</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Floating Tanpura Drone Widget (EZSwara Style) -->
    <div class="tanpura-widget" id="tanpura-widget">
      <button class="tanpura-fab" id="tanpura-fab" title="Toggle Tanpura Drone (Shruti Box)">
        <span class="tanpura-fab-icon" style="font-size: 1.4rem;">🪕</span>
      </button>

      <div class="tanpura-popover" id="tanpura-popover">
        <div class="tanpura-pop-header">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.2rem;">🪕</span>
            <div>
              <span class="tanpura-pop-title">Authentic Tanpura</span>
              <div style="font-size: 0.68rem; color: var(--gold-primary); font-weight: 500;">Miraj & Tanjore Acoustic Engine</div>
            </div>
          </div>
          <button class="modal-close" id="btn-tanpura-close" style="font-size: 1.2rem; cursor: pointer;">&times;</button>
        </div>

        <!-- Live Pitch & Kattai Status Pill -->
        <div class="tanpura-pitch-status" id="tanpura-pitch-status">
          <div class="tanpura-pitch-main">
            <span id="tanpura-scale-name" style="font-weight: 700; color: var(--text-primary);">D3</span>
            <span class="tanpura-pitch-dot" style="color: var(--text-tertiary); margin: 0 4px;">·</span>
            <span id="tanpura-kattai-badge" style="color: var(--gold-primary); font-weight: 600;">2 Kattai</span>
          </div>
          <span id="tanpura-freq-label" class="tanpura-hz-val">146.8 Hz</span>
        </div>

        <!-- 4-String Rhythmic Activity Meter -->
        <div class="tanpura-strings-meter" id="tanpura-strings-meter" title="4-String Pluck Cycle Sequence">
          <div class="tanpura-str-pill" data-str="0" id="tanpura-str-0"><span class="str-num">1</span> <span class="str-label" id="tanpura-str1-label">Pa</span></div>
          <div class="tanpura-str-pill" data-str="1" id="tanpura-str-1"><span class="str-num">2</span> <span class="str-label">Sȧ</span></div>
          <div class="tanpura-str-pill" data-str="2" id="tanpura-str-2"><span class="str-num">3</span> <span class="str-label">Sȧ₂</span></div>
          <div class="tanpura-str-pill" data-str="3" id="tanpura-str-3"><span class="str-num">4</span> <span class="str-label">Sạ</span></div>
        </div>

        <!-- Scale & Kattai Selection (All 12 Scales) -->
        <div class="tanpura-pop-section">
          <label class="tanpura-pop-label">Shruti / Scale (All 12 Kattais):</label>
          <div class="tanpura-scale-row">
            <select class="tanpura-scale-select" id="tanpura-scale-select">
              <option value="C">C · 1 Kattai (White 1)</option>
              <option value="C#">C# · 1.5 Kattai (Black 1)</option>
              <option value="D" selected>D · 2 Kattai (White 2)</option>
              <option value="D#">D# · 2.5 Kattai (Black 2)</option>
              <option value="E">E · 3 Kattai (White 3)</option>
              <option value="F">F · 4 Kattai (White 4)</option>
              <option value="F#">F# · 4.5 Kattai (Black 3)</option>
              <option value="G">G · 5 Kattai (White 5)</option>
              <option value="G#">G# · 5.5 Kattai (Black 4)</option>
              <option value="A">A · 6 Kattai (White 6)</option>
              <option value="A#">A# · 6.5 Kattai (Black 5)</option>
              <option value="B">B · 7 Kattai (White 7)</option>
            </select>
            <div class="tanpura-octave-toggle" id="tanpura-octave-toggle">
              <button class="tanpura-oct-btn active" data-octave="male" title="Male Pitch Range (Octave 2-3)">♂ Male</button>
              <button class="tanpura-oct-btn" data-octave="female" title="Female Pitch Range (Octave 3-4)">♀ Female</button>
            </div>
          </div>
        </div>

        <!-- String 1 Tuning -->
        <div class="tanpura-pop-section">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <label class="tanpura-pop-label">String 1 Tuning:</label>
            <span style="font-size: 0.70rem; color: var(--text-tertiary);" id="tanpura-tuning-hint">Panchamam (1.5x)</span>
          </div>
          <div class="tanpura-tuning-chips" id="tanpura-tuning-chips" style="width: 100%; display: flex;">
            <button class="tanpura-chip active" data-tuning="Pa" style="flex: 1;">Pa (5th)</button>
            <button class="tanpura-chip" data-tuning="Ma" style="flex: 1;">Ma (4th)</button>
            <button class="tanpura-chip" data-tuning="Ni" style="flex: 1;">Ni (7th)</button>
          </div>
        </div>

        <!-- Timbre & Pluck Tempo -->
        <div class="tanpura-pop-section">
          <div class="tanpura-options-grid">
            <div>
              <label class="tanpura-pop-label">Timbre:</label>
              <div class="tanpura-chip-row" id="tanpura-timbre-chips">
                <button class="tanpura-chip active" data-timbre="miraj" title="Deep, resonant seasoned gourd tone">Miraj</button>
                <button class="tanpura-chip" data-timbre="tanjore" title="Crisp, bright bell-like singing chime">Tanjore</button>
              </div>
            </div>
            <div>
              <label class="tanpura-pop-label">Tempo:</label>
              <div class="tanpura-chip-row" id="tanpura-tempo-chips">
                <button class="tanpura-chip" data-tempo="slow">Slow</button>
                <button class="tanpura-chip active" data-tempo="medium">Med</button>
                <button class="tanpura-chip" data-tempo="fast">Fast</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Microtonal Fine-Tuning Slider -->
        <div class="tanpura-pop-section">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <label class="tanpura-pop-label">Fine Tune (Cents):</label>
            <span id="tanpura-cents-val" style="font-family: var(--font-mono); font-size: 0.74rem; color: var(--gold-primary); font-weight: 600;">0 ¢</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <input type="range" class="tanpura-vol-slider" id="tanpura-cents-slider" min="-50" max="50" step="1" value="0" />
            <button id="btn-tanpura-cents-reset" class="btn-xs" style="font-size: 0.68rem; padding: 2px 6px; cursor: pointer;" title="Reset fine tune to 0 cents">0¢</button>
          </div>
        </div>

        <!-- Volume -->
        <div class="tanpura-pop-section">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <label class="tanpura-pop-label">Acoustic Master Volume:</label>
            <span id="tanpura-vol-val" style="font-family: var(--font-mono); font-size: 0.74rem; color: var(--text-secondary);">65%</span>
          </div>
          <input type="range" class="tanpura-vol-slider" id="tanpura-vol" min="0" max="1" step="0.05" value="0.65" />
        </div>

        <!-- Start/Stop Drone Button -->
        <div style="margin-top: 4px;">
          <button class="btn-primary tanpura-start-btn" id="btn-tanpura-toggle" style="width: 100%; display: flex; align-items: center; justify-content: center; gap: 8px; padding: 8px 14px; font-size: 0.86rem; font-weight: 600;">
            <span id="tanpura-btn-icon">▶</span>
            <span id="tanpura-btn-text">Start Authentic Tanpura</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Carnatic Manuscript Export Modal (EZSwara Style) -->
    <div class="export-modal-backdrop" id="export-modal-backdrop">
      <div class="export-modal">
        <div class="export-modal-header">
          <div class="export-modal-title">Carnatic Manuscript Score</div>
          <button class="modal-close" id="btn-close-export-modal" style="font-size: 1.5rem; cursor: pointer;">&times;</button>
        </div>
        <textarea class="export-text-preview" id="export-text-preview" readonly></textarea>
        <div class="export-actions">
          <button class="btn-secondary" id="btn-copy-export">📋 Copy Notation</button>
          <button class="btn-secondary" id="btn-print-export">🖨️ Print Manuscript</button>
          <button class="btn-primary" id="btn-download-export">⤓ Download .txt</button>
        </div>
      </div>
    </div>

    <!-- Toast -->
    <div class="toast" id="toast"></div>
  `;

  bindEvents();
  initClassicalStageReady();
}

// ── Event Handlers ──────────────────────────────────────────────────────────
function bindEvents() {
  const recordCard = document.getElementById('record-card');
  recordCard.addEventListener('click', (e) => {
    if (e.target.closest('.recording-controls') || e.target.closest('.file-info')) return;
    initRecording();
  });

  document.getElementById('record-btn').addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!state.recorder) {
      await initRecording();
      await toggleRecording();
    } else {
      await toggleRecording();
    }
  });

  const uploadCard = document.getElementById('upload-card');
  uploadCard.addEventListener('click', (e) => {
    if (e.target.closest('.file-info') || e.target.closest('#btn-load-demo') || e.target.closest('.sample-take-btn')) return;
    document.getElementById('file-input').click();
  });

  document.getElementById('file-input').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  uploadCard.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadCard.classList.add('drag-over');
  });
  uploadCard.addEventListener('dragleave', () => {
    uploadCard.classList.remove('drag-over');
  });
  uploadCard.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadCard.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      handleFile(file);
    } else {
      showToast('Please upload an audio file', true);
    }
  });

  // Demo audio
  const btnAbhogi = document.getElementById('btn-load-abhogi-demo');
  if (btnAbhogi) btnAbhogi.addEventListener('click', loadAbhogiDemo);
  document.getElementById('btn-load-demo').addEventListener('click', loadDemoAudio);

  // Analyze
  document.getElementById('analyze-btn').addEventListener('click', startAnalysis);

  // Stage Playback
  const stagePlayBtn = document.getElementById('stage-play-btn');
  if (stagePlayBtn) stagePlayBtn.addEventListener('click', togglePlayback);

  // Stage Mode Switcher
  const btnStageTala = document.getElementById('btn-stage-tala');
  if (btnStageTala) btnStageTala.addEventListener('click', () => setCarnaticMode('tala'));

  const btnStageFreeflow = document.getElementById('btn-stage-freeflow');
  if (btnStageFreeflow) btnStageFreeflow.addEventListener('click', () => setCarnaticMode('freeflow'));

  // 4-Line Stage Pagination Controls
  const btnPrevLines = document.getElementById('score-prev-page');
  if (btnPrevLines) {
    btnPrevLines.addEventListener('click', () => {
      if (state.scoreLineWindow > 0) {
        state.scoreLineWindow--;
        renderTalaScoreBoard();
      }
    });
  }
  const btnNextLines = document.getElementById('score-next-page');
  if (btnNextLines) {
    btnNextLines.addEventListener('click', () => {
      const currentData = state.talaData || state.adiTalaData;
      const total = currentData?.avartanams?.length || 0;
      const maxWindow = Math.max(0, Math.ceil(total / 4) - 1);
      if (state.scoreLineWindow < maxWindow) {
        state.scoreLineWindow++;
        renderTalaScoreBoard();
      }
    });
  }


  // Stage Tala Selector
  const stageTalaSelect = document.getElementById('stage-tala-select');
  if (stageTalaSelect) {
    stageTalaSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (val === 'auto') {
        state.talaMode = 'auto';
        autoDetectAndApplyTala();
      } else {
        state.talaMode = 'manual';
        setCarnaticTala(val, false);
      }
    });
  }

  // Stage Export & Manuscript Export
  const btnStageExport = document.getElementById('btn-stage-export');
  if (btnStageExport) btnStageExport.addEventListener('click', openExportModal);
  document.getElementById('btn-export-sargam').addEventListener('click', openExportModal);

  // "Your Sa" Calibration Bar
  const btnSaAuto = document.getElementById('btn-sa-auto');
  if (btnSaAuto) btnSaAuto.addEventListener('click', activateAutoSa);

  const saRoot = document.getElementById('sa-root-select');
  const saOctave = document.getElementById('sa-octave-select');
  if (saRoot && saOctave) {
    saRoot.addEventListener('change', updateSaFromControls);
    saOctave.addEventListener('change', updateSaFromControls);
  }

  // Tanpura Floating Widget
  const tanpuraFab = document.getElementById('tanpura-fab');
  const tanpuraPopover = document.getElementById('tanpura-popover');
  if (tanpuraFab && tanpuraPopover) {
    tanpuraFab.addEventListener('click', (e) => {
      e.stopPropagation();
      const willOpen = !tanpuraPopover.classList.contains('open');
      tanpuraPopover.classList.toggle('open');
      if (willOpen) {
        updateTanpuraStatusDisplay();
        if (!tanpura.isPlaying) {
          toggleTanpuraDrone();
        }
      }
    });
  }

  const btnTanpuraToggle = document.getElementById('btn-tanpura-toggle');
  if (btnTanpuraToggle) {
    btnTanpuraToggle.addEventListener('click', () => {
      toggleTanpuraDrone();
    });
  }

  const btnTanpuraClose = document.getElementById('btn-tanpura-close');
  if (btnTanpuraClose && tanpuraPopover) {
    btnTanpuraClose.addEventListener('click', (e) => {
      e.stopPropagation();
      tanpuraPopover.classList.remove('open');
    });
  }

  // Scale Select (All 12 Scales)
  const tanpuraScaleSelect = document.getElementById('tanpura-scale-select');
  if (tanpuraScaleSelect) {
    tanpuraScaleSelect.addEventListener('change', (e) => {
      const note = e.target.value;
      tanpura.setScale(note);
      syncTanpuraWithMainSa(note, tanpura.octave);
      showToast(`Tanpura scale set to ${note} (${tanpura.saHz.toFixed(1)} Hz)`);
    });
  }

  // Octave Pill Switcher (Male vs Female)
  document.querySelectorAll('.tanpura-oct-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      document.querySelectorAll('.tanpura-oct-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const oct = btn.dataset.octave;
      tanpura.setOctave(oct);
      syncTanpuraWithMainSa(tanpura.currentScaleNote, oct);
      showToast(`Tanpura octave: ${oct === 'female' ? 'Female (Madhya/Taara)' : 'Male (Mandra/Madhya)'}`);
    });
  });

  // String 1 Tuning (Pa, Ma, Ni)
  document.querySelectorAll('#tanpura-tuning-chips .tanpura-chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      document.querySelectorAll('#tanpura-tuning-chips .tanpura-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const tuning = chip.dataset.tuning;
      tanpura.setFirstString(tuning);
      const str1Label = document.getElementById('tanpura-str1-label');
      if (str1Label) str1Label.textContent = tuning;
      const hintEl = document.getElementById('tanpura-tuning-hint');
      if (hintEl) {
        if (tuning === 'Pa') hintEl.textContent = 'Panchamam (1.5x)';
        else if (tuning === 'Ma') hintEl.textContent = 'Madhyamam (1.33x)';
        else if (tuning === 'Ni') hintEl.textContent = 'Nishadham (1.88x)';
      }
      showToast(`Tanpura String 1 tuned to ${tuning}`);
    });
  });

  // Timbre Chips (Miraj vs Tanjore)
  document.querySelectorAll('#tanpura-timbre-chips .tanpura-chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      document.querySelectorAll('#tanpura-timbre-chips .tanpura-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const timbre = chip.dataset.timbre;
      tanpura.setTimbre(timbre);
      showToast(`Tanpura timbre: ${timbre === 'miraj' ? 'Miraj (Deep Gourd Body)' : 'Tanjore (Bright Chime)'}`);
    });
  });

  // Tempo Chips (Slow, Medium, Fast)
  document.querySelectorAll('#tanpura-tempo-chips .tanpura-chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      document.querySelectorAll('#tanpura-tempo-chips .tanpura-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const tempo = chip.dataset.tempo;
      tanpura.setTempo(tempo);
      showToast(`Pluck Cadence: ${tempo.toUpperCase()}`);
    });
  });

  // Microtonal Fine-Tuning Slider
  const centsSlider = document.getElementById('tanpura-cents-slider');
  const centsVal = document.getElementById('tanpura-cents-val');
  const centsReset = document.getElementById('btn-tanpura-cents-reset');
  if (centsSlider) {
    centsSlider.addEventListener('input', (e) => {
      const cents = parseInt(e.target.value, 10);
      tanpura.setFineTune(cents);
      if (centsVal) centsVal.textContent = `${cents > 0 ? '+' : ''}${cents} ¢`;
      updateTanpuraStatusDisplay();
    });
  }
  if (centsReset) {
    centsReset.addEventListener('click', (e) => {
      e.stopPropagation();
      if (centsSlider) centsSlider.value = '0';
      if (centsVal) centsVal.textContent = '0 ¢';
      tanpura.setFineTune(0);
      updateTanpuraStatusDisplay();
      showToast('Fine-tuning reset to 0 cents');
    });
  }

  // Volume Slider
  const tanpuraVol = document.getElementById('tanpura-vol');
  const volVal = document.getElementById('tanpura-vol-val');
  if (tanpuraVol) {
    tanpuraVol.addEventListener('input', (e) => {
      const vol = parseFloat(e.target.value);
      tanpura.setVolume(vol);
      if (volVal) volVal.textContent = `${Math.round(vol * 100)}%`;
    });
  }

  // Active string real-time indicator callback
  tanpura.setOnPluck((step) => {
    document.querySelectorAll('.tanpura-str-pill').forEach(pill => pill.classList.remove('plucking'));
    if (step >= 0) {
      const pill = document.getElementById(`tanpura-str-${step}`);
      if (pill) {
        pill.classList.add('plucking');
        setTimeout(() => pill.classList.remove('plucking'), 480);
      }
    }
  });

  document.addEventListener('click', (e) => {
    if (tanpuraPopover && !e.target.closest('#tanpura-widget')) {
      tanpuraPopover.classList.remove('open');
    }
  });

  // Export Modal Handlers
  const btnCloseExport = document.getElementById('btn-close-export-modal');
  const exportBackdrop = document.getElementById('export-modal-backdrop');
  if (btnCloseExport && exportBackdrop) {
    btnCloseExport.addEventListener('click', closeExportModal);
    exportBackdrop.addEventListener('click', (e) => {
      if (e.target === exportBackdrop) closeExportModal();
    });
  }

  const btnCopyExport = document.getElementById('btn-copy-export');
  if (btnCopyExport) btnCopyExport.addEventListener('click', copyExportNotation);

  const btnPrintExport = document.getElementById('btn-print-export');
  if (btnPrintExport) btnPrintExport.addEventListener('click', printExportNotation);

  const btnDownloadExport = document.getElementById('btn-download-export');
  if (btnDownloadExport) btnDownloadExport.addEventListener('click', exportSargamFile);

  // Export Pitch CSV
  document.getElementById('btn-export-pitch-csv').addEventListener('click', exportPitchCsv);

  // New Analysis
  document.getElementById('new-analysis-btn').addEventListener('click', resetApp);

  // Modal
  const modal = document.getElementById('modal-about');
  document.getElementById('btn-about').addEventListener('click', () => modal.classList.add('visible'));
  document.getElementById('btn-close-about').addEventListener('click', () => modal.classList.remove('visible'));
  // Top Classical Tala Stage
  initTopTalaStage();
}

function initTopTalaStage() {
  const container = document.getElementById('top-tala-hand-stage');
  if (!container) return;

  state.topHandStage = new CarnaticHandStage('top-tala-hand-stage', {
    isTopStage: true,
    soundEnabled: true,
    bpm: 76,
    onBeatChange: (beat, beatInfo) => {
      // Update top beads
      document.querySelectorAll('.top-tala-beads-wrap .tala-bead').forEach(btn => {
        const b = parseInt(btn.dataset.beat, 10);
        btn.classList.toggle('active', b === beat);
      });

      const fillEl = document.getElementById('top-tala-progress-fill');
      if (fillEl) fillEl.style.width = `${(beat / 8) * 100}%`;
    }
  });

  // Play / Pause Toggle
  const playBtn = document.getElementById('top-tala-play-btn');
  const playIcon = document.getElementById('top-tala-play-icon');
  const playText = document.getElementById('top-tala-play-text');

  if (playBtn) {
    playBtn.addEventListener('click', () => {
      const isPlaying = state.topHandStage.togglePlay();
      if (playIcon) playIcon.textContent = isPlaying ? '❚❚' : '▶';
      if (playText) playText.textContent = isPlaying ? 'Pause Tala' : 'Play Tala';
      playBtn.classList.toggle('playing', isPlaying);
    });
  }

  // Tempo Buttons (Auto, Vilamba 56, Madhyama 76, Dhruta 108)
  document.querySelectorAll('.tala-tempo-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tala-tempo-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const bpmMode = btn.dataset.bpm;
      state.activeTempoMode = bpmMode;
      let targetBpm;
      if (bpmMode === 'auto') {
        targetBpm = state.detectedBpm || 76;
      } else {
        targetBpm = parseInt(bpmMode, 10);
      }
      state.topHandStage.setBpm(targetBpm);
      updateTopTalaTempoLabel(targetBpm);
    });
  });

  // Sound Toggle
  const soundBtn = document.getElementById('top-tala-sound-btn');
  const soundIcon = document.getElementById('top-tala-sound-icon');
  const soundLabel = document.getElementById('top-tala-sound-label');

  if (soundBtn) {
    soundBtn.addEventListener('click', () => {
      const soundOn = state.topHandStage.toggleSound();
      if (soundIcon) soundIcon.textContent = soundOn ? '🔊' : '🔇';
      if (soundLabel) soundLabel.textContent = soundOn ? 'Sound On' : 'Muted';
      soundBtn.classList.toggle('muted', !soundOn);
    });
  }

  // Beads Click to Step
  document.querySelectorAll('.top-tala-beads-wrap .tala-bead').forEach(btn => {
    btn.addEventListener('click', () => {
      const beat = parseInt(btn.dataset.beat, 10);
      state.topHandStage.setBeat(beat, true);
    });
  });

  // Start autonomous playback loop immediately at default/detected speed
  const initialBpm = state.detectedBpm || 76;
  state.topHandStage.startLoop(initialBpm);
  if (playIcon) playIcon.textContent = '❚❚';
  if (playText) playText.textContent = 'Pause Tala';
  playBtn?.classList.add('playing');
}

/**
 * Dynamically updates the Identified Probable BPM and re-times the Tala Hand Stage.
 */
function applyDetectedBpm(bpm, kalaName = null, reason = 'Detected Vocal Tempo') {
  if (!bpm || bpm <= 0) return;
  state.detectedBpm = Math.round(bpm);
  state.detectedKala = kalaName || (state.detectedBpm < 68 ? 'Vilamba Kala' : (state.detectedBpm > 98 ? 'Dhruta Kala' : 'Madhyama Kala'));

  // Update button and badge labels
  const autoBtnLabel = document.getElementById('btn-auto-bpm-label');
  if (autoBtnLabel) autoBtnLabel.textContent = state.detectedBpm;

  const autoBpmEl = document.getElementById('top-tala-detected-bpm');
  if (autoBpmEl) autoBpmEl.textContent = state.detectedBpm;

  const autoKalaEl = document.getElementById('top-tala-detected-kala');
  if (autoKalaEl) autoKalaEl.textContent = state.detectedKala;

  const tempoPill = document.getElementById('top-tala-tempo-pill');
  if (tempoPill) {
    tempoPill.innerHTML = `✦ Probable Tempo: <strong>${state.detectedBpm} BPM</strong> (${state.detectedKala})`;
    tempoPill.classList.remove('highlight-pulse');
    void tempoPill.offsetWidth; // trigger reflow
    tempoPill.classList.add('highlight-pulse');
  }

  // If in Auto mode (default), set top hand stage loop speed immediately
  if (state.activeTempoMode === 'auto') {
    document.querySelectorAll('.tala-tempo-btn').forEach(b => b.classList.remove('active'));
    const autoBtn = document.getElementById('tempo-auto');
    if (autoBtn) autoBtn.classList.add('active');

    if (state.topHandStage) {
      state.topHandStage.setBpm(state.detectedBpm);
    }
  }

  // Also sync secondary hand rig if active
  if (state.handStage) {
    state.handStage.setBpm(state.detectedBpm);
  }
}

function updateTopTalaTempoLabel(bpm) {
  let kala = 'Madhyama Kala';
  if (bpm < 68) kala = 'Vilamba Kala';
  else if (bpm > 98) kala = 'Dhruta Kala';
  
  const autoKalaEl = document.getElementById('top-tala-detected-kala');
  if (autoKalaEl && state.activeTempoMode !== 'auto') {
    autoKalaEl.textContent = `${kala} · Manual`;
  }
}

// ── Permanent Classical Manuscript Theme Initialization ────────────────────
function initTheme() {
  document.documentElement.setAttribute('data-theme', 'manuscript');
}

// ── "Your Sa" Root Calibration ──────────────────────────────────────────────
function calculateNoteFreq(root, octave) {
  const noteOffsets = { 'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11 };
  const semitone = noteOffsets[root] !== undefined ? noteOffsets[root] : 2;
  const midi = (parseInt(octave, 10) + 1) * 12 + semitone;
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function updateSaFromControls() {
  const root = document.getElementById('sa-root-select').value;
  const oct = document.getElementById('sa-octave-select').value;
  const hz = calculateNoteFreq(root, oct);
  state.selectedRoot = root;
  state.selectedOctave = parseInt(oct, 10);
  state.customTonicFreq = Number(hz.toFixed(2));
  state.isAutoSa = false;

  const autoBtn = document.getElementById('btn-sa-auto');
  if (autoBtn) autoBtn.classList.remove('active');

  const hzPill = document.getElementById('sa-hz-indicator');
  if (hzPill) hzPill.textContent = `${root}${oct} · ${hz.toFixed(1)} Hz`;

  // Bidirectional sync into Tanpura
  const octStr = parseInt(oct, 10) >= 4 ? 'female' : 'male';
  tanpura.setScale(root, octStr);
  updateTanpuraStatusDisplay();

  showToast(`Adhara Shadja (Sa) calibrated to ${root}${oct} (${hz.toFixed(1)} Hz)`);
}

function activateAutoSa() {
  state.isAutoSa = true;
  state.customTonicFreq = null;
  const autoBtn = document.getElementById('btn-sa-auto');
  if (autoBtn) autoBtn.classList.add('active');

  const hzPill = document.getElementById('sa-hz-indicator');
  if (hzPill) hzPill.textContent = 'Auto-Detecting ✦';

  showToast('Sa mode: Automatic Harmonic Consonance Calibration');
}

// ── Floating Tanpura Synthesizer ───────────────────────────────────────────
function toggleTanpuraDrone() {
  const isPlaying = tanpura.toggle();
  updateTanpuraUI(isPlaying);
}

function updateTanpuraStatusDisplay() {
  const scaleNameEl = document.getElementById('tanpura-scale-name');
  const kattaiEl = document.getElementById('tanpura-kattai-badge');
  const freqEl = document.getElementById('tanpura-freq-label');
  const scaleSelect = document.getElementById('tanpura-scale-select');

  const curNote = tanpura.currentScaleNote || 'D';
  const curOct = tanpura.octave === 'female' ? 4 : (['A', 'A#', 'B'].includes(curNote) ? 2 : 3);
  const scaleDef = TANPURA_SCALES.find(s => s.note.toUpperCase() === curNote.toUpperCase());

  if (scaleNameEl) scaleNameEl.textContent = `${curNote}${curOct}`;
  if (kattaiEl && scaleDef) kattaiEl.textContent = scaleDef.kattai;
  if (freqEl) freqEl.textContent = `${tanpura.saHz.toFixed(1)} Hz`;
  if (scaleSelect && scaleSelect.value !== curNote) scaleSelect.value = curNote;
}

function syncTanpuraWithMainSa(note, octaveStr) {
  const octNum = octaveStr === 'female' ? 4 : (['A', 'A#', 'B'].includes(note) ? 2 : 3);
  const saRoot = document.getElementById('sa-root-select');
  const saOctave = document.getElementById('sa-octave-select');
  if (saRoot) saRoot.value = note;
  if (saOctave) saOctave.value = String(octNum);

  const hz = calculateNoteFreq(note, octNum);
  state.selectedRoot = note;
  state.selectedOctave = octNum;
  state.customTonicFreq = Number(hz.toFixed(2));
  state.isAutoSa = false;

  const autoBtn = document.getElementById('btn-sa-auto');
  if (autoBtn) autoBtn.classList.remove('active');

  const hzPill = document.getElementById('sa-hz-indicator');
  if (hzPill) hzPill.textContent = `${note}${octNum} · ${hz.toFixed(1)} Hz`;

  updateTanpuraStatusDisplay();
}

function updateTanpuraUI(isPlaying) {
  const fab = document.getElementById('tanpura-fab');
  const btnIcon = document.getElementById('tanpura-btn-icon');
  const btnText = document.getElementById('tanpura-btn-text');
  const toggleBtn = document.getElementById('btn-tanpura-toggle');

  if (fab) {
    if (isPlaying) fab.classList.add('playing');
    else fab.classList.remove('playing');
  }

  if (btnIcon && btnText) {
    btnIcon.textContent = isPlaying ? '⏹' : '▶';
    btnText.textContent = isPlaying ? 'Stop Tanpura Drone' : 'Start Authentic Tanpura';
  } else if (toggleBtn) {
    toggleBtn.textContent = isPlaying ? 'Stop Drone' : 'Start Drone';
  }

  updateTanpuraStatusDisplay();
}

// ── Live Accumulating Swara Run ─────────────────────────────────────────────
let recentLiveSwara = null;
let consecutiveFrames = 0;

function resetLiveRun() {
  state.liveRunSwaras = [];
  recentLiveSwara = null;
  consecutiveFrames = 0;
  updateLiveRunDisplay();
}

function feedLiveSwara(swaraSymbol) {
  if (!swaraSymbol) {
    consecutiveFrames = 0;
    return;
  }
  if (swaraSymbol === recentLiveSwara) {
    consecutiveFrames++;
    if (consecutiveFrames === 3) {
      const lastToken = state.liveRunSwaras[state.liveRunSwaras.length - 1];
      if (lastToken !== swaraSymbol) {
        state.liveRunSwaras.push(swaraSymbol);
        updateLiveRunDisplay();
      }
    }
  } else {
    recentLiveSwara = swaraSymbol;
    consecutiveFrames = 1;
  }
}

function updateLiveRunDisplay() {
  const counter = document.getElementById('live-run-counter');
  const tokensContainer = document.getElementById('live-run-tokens');
  if (counter) {
    counter.textContent = `${state.liveRunSwaras.length} Swara${state.liveRunSwaras.length === 1 ? '' : 's'}`;
  }
  if (tokensContainer) {
    if (state.liveRunSwaras.length === 0) {
      tokensContainer.innerHTML = '<span style="color: var(--text-tertiary); font-family: var(--font-sans); font-size: 0.85rem;">Notes will accumulate here in real-time as you sing...</span>';
    } else {
      tokensContainer.innerHTML = state.liveRunSwaras.map(s => `<span class="live-run-token">${s}</span>`).join('');
      tokensContainer.scrollLeft = tokensContainer.scrollWidth;
    }
  }
}

// ── Carnatic Manuscript Modal & Export ──────────────────────────────────────
function openExportModal() {
  const backdrop = document.getElementById('export-modal-backdrop');
  const preview = document.getElementById('export-text-preview');
  if (!backdrop || !preview) return;

  const text = exportCarnaticTextNotation(state.adiTalaData, state.raga, state.tonic);
  preview.value = text;
  backdrop.classList.add('open');
}

function closeExportModal() {
  const backdrop = document.getElementById('export-modal-backdrop');
  if (backdrop) backdrop.classList.remove('open');
}

function copyExportNotation() {
  const preview = document.getElementById('export-text-preview');
  if (!preview) return;
  navigator.clipboard.writeText(preview.value).then(() => {
    showToast('Carnatic manuscript copied to clipboard! 📋');
  }).catch(() => {
    showToast('Failed to copy to clipboard', true);
  });
}

function printExportNotation() {
  const preview = document.getElementById('export-text-preview');
  if (!preview) return;
  const printWin = window.open('', '_blank');
  if (!printWin) {
    showToast('Popup blocked. Please allow popups to print.', true);
    return;
  }
  printWin.document.write(`
    <html>
      <head>
        <title>Carnatic Manuscript Score — NoteScope</title>
        <style>
          body { font-family: 'Courier New', Courier, monospace; white-space: pre; padding: 30px; font-size: 13px; line-height: 1.45; color: #111; }
        </style>
      </head>
      <body>${preview.value.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</body>
    </html>
  `);
  printWin.document.close();
  printWin.focus();
  setTimeout(() => {
    printWin.print();
  }, 250);
}

async function loadAbhogiDemo() {
  try {
    showToast('Loading Abhogi Varnam master take...');
    const resp = await fetch('/samples/abhogi_varnam_take.wav');
    if (!resp.ok) throw new Error('Fetch failed');
    const blob = await resp.blob();
    const file = new File([blob], 'test_master_take.wav', { type: 'audio/wav' });
    handleFile(file);
    showToast('Master take loaded! Click "Recognize Swaras & Pitch" below.');
  } catch (err) {
    console.error('Abhogi demo loading failed:', err);
    showToast('Failed to load Abhogi demo file', true);
  }
}

// ── Demo Audio ──────────────────────────────────────────────────────────────
async function loadDemoAudio() {
  try {
    showToast('Loading vocal Sargam demo take...');
    const resp = await fetch('/samples/vocal_sargam.wav');
    if (!resp.ok) throw new Error('Fetch failed');
    const blob = await resp.blob();
    const file = new File([blob], 'vocal_sargam_scale.wav', { type: 'audio/wav' });
    handleFile(file);
    showToast('Sample loaded! Click "Recognize Swaras & Pitch" below.');
  } catch (err) {
    console.error('Demo loading failed:', err);
    showToast('Failed to load demo file', true);
  }
}

// ── Recording with Live Pitch & Subtitles ─────────────────────────────────────
async function initRecording() {
  const controls = document.getElementById('recording-controls');
  const recordCard = document.getElementById('record-card');

  if (controls.classList.contains('visible')) return;

  state.recorder = new AudioRecorder();
  const ok = await state.recorder.init();

  if (!ok) {
    showToast('Microphone access denied. Please grant microphone permission.', true);
    return;
  }

  controls.classList.add('visible');
  recordCard.classList.add('active');

  document.getElementById('upload-card').style.opacity = '0.35';
  document.getElementById('upload-card').style.pointerEvents = 'none';
}

async function toggleRecording() {
  const recordBtn = document.getElementById('record-btn');
  const label = document.getElementById('recording-label');
  const timer = document.getElementById('recording-timer');
  const liveSubContent = document.getElementById('live-sub-content');

  if (!state.recorder) return;

  if (state.recorder.isRecording) {
    recordBtn.classList.remove('recording');
    label.textContent = 'Recording captured';

    const blob = await state.recorder.stop();
    state.recorder.stopLiveWaveform();

    if (blob) {
      state.audioBlob = blob;
      const info = document.getElementById('recording-info');
      info.classList.add('visible');
      document.getElementById('recording-name').textContent = 'Live Vocal Take';
      document.getElementById('analyze-zone').classList.add('visible');
    }

    if (liveSubContent) {
      liveSubContent.innerHTML = `<span class="live-sub-placeholder">Take captured! Ready to recognize Swaras.</span>`;
    }
  } else {
    recordBtn.classList.add('recording');
    label.textContent = 'Singing... tap to finish take';
    timer.textContent = '00:00.00';
    resetLiveRun();

    state.recorder.start((time) => {
      timer.textContent = time;
    });

    const canvas = document.getElementById('live-canvas');
    canvas.width = canvas.parentElement.offsetWidth * (window.devicePixelRatio || 1);
    canvas.height = 60 * (window.devicePixelRatio || 1);

    // Live pitch callback to drive real-time singing subtitles & live accumulating run!
    state.recorder.drawLiveWaveform(canvas, ({ pitchHz, rms }) => {
      if (!liveSubContent) return;

      if (pitchHz > 65) {
        const liveTonicHz = state.customTonicFreq || (state.tonic ? state.tonic.tonicFreq : 146.8);
        const liveSwara = pitchToLiveSwara(pitchHz, liveTonicHz);
        if (liveSwara) {
          const dev = liveSwara.centsDeviation;
          const devStr = dev >= 0 ? `+${dev}¢` : `${dev}¢`;
          liveSubContent.innerHTML = `
            <div class="live-swara-giant">
              <span>${liveSwara.formatted}</span>
              <span class="live-swara-devanagari">${liveSwara.devanagari}</span>
            </div>
            <div class="live-swara-meta">
              <span>${liveSwara.hz} Hz</span> · <span>${liveSwara.western}</span> · <span>${devStr}</span>
            </div>
          `;
          feedLiveSwara(liveSwara.formatted);
        }
      } else {
        liveSubContent.innerHTML = `<span class="live-sub-placeholder">Sing into microphone to see notes live...</span>`;
        feedLiveSwara(null);
      }
    });
  }
}

async function handleFile(file) {
  state.audioBlob = file;

  const fileInfo = document.getElementById('file-info');
  fileInfo.classList.add('visible');
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent = formatFileSize(file.size);

  document.getElementById('upload-card').classList.add('active');
  document.getElementById('record-card').style.opacity = '0.35';
  document.getElementById('record-card').style.pointerEvents = 'none';

  document.getElementById('analyze-zone').classList.add('visible');

  // Fast pre-analysis: immediately decode audio and identify probable vocal tempo!
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    state.audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const acoustic = detectAudioBpm(state.audioBuffer);
    if (acoustic && acoustic.bpm) {
      applyDetectedBpm(acoustic.bpm, acoustic.kala, 'Acoustic Onset Analysis');
      showToast(`✦ Probable Tempo Identified: ${acoustic.bpm} BPM (${acoustic.kala}) · Tala speed adjusted!`);
    }
    audioCtx.close();
  } catch (err) {
    console.warn('Acoustic tempo pre-estimation error:', err);
  }
}

// ── Core Analysis Pipeline (Neural + Auto-Sa) ────────────────────────────────
async function startAnalysis() {
  if (!state.audioBlob) return;

  const overlay = document.getElementById('processing-overlay');
  const progressBar = document.getElementById('progress-bar');
  const statusText = document.getElementById('processing-status');

  overlay.classList.add('visible');

  try {
    statusText.textContent = 'Decoding vocal audio...';
    progressBar.style.width = '8%';

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await state.audioBlob.arrayBuffer();
    state.audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Call the Revamped Vocal Recognition Engine (TorchCrepe Neural Server with Client MIR Fallback)
    const result = await recognizeVocalNotes(
      state.audioBlob,
      state.audioBuffer,
      (progress, message) => {
        progressBar.style.width = `${Math.round(progress * 100)}%`;
        if (message) statusText.textContent = message;
      },
      state.customTonicFreq
    );

    state.swaraNotes = result.notes;
    state.timeSteps = result.timeSteps;
    state.pitches = result.pitches;
    state.confidences = result.confidences;
    state.tonic = result.tonic;

    statusText.textContent = 'Rendering waveform, horizontal track & subtitles...';
    progressBar.style.width = '95%';

    if (state.recorder) {
      state.recorder.destroy();
      state.recorder = null;
    }

    await renderResults();

    progressBar.style.width = '100%';
    setTimeout(() => {
      overlay.classList.remove('visible');
    }, 350);

    audioCtx.close();
  } catch (err) {
    console.error('Analysis failed:', err);
    overlay.classList.remove('visible');
    showToast('Analysis failed. Please try another audio file.', true);
  }
}

// ── Results View Rendering ──────────────────────────────────────────────────
async function renderResults() {
  const analysisSection = document.getElementById('analysis-section');
  if (analysisSection) analysisSection.classList.add('visible');

  const tonicBanner = document.getElementById('tonic-banner');
  if (tonicBanner) tonicBanner.classList.add('visible');

  // 1. Reset 4-line window to top and render Carnatic Kriti Notation & Tala Studio
  state.scoreLineWindow = 0;
  renderCarnaticStudio();


  // 2. Populate Tonic & Vocal Stats with Carnatic musicology
  document.getElementById('tonic-freq').textContent = `Sa = ${state.tonic.tonicFreq} Hz`;
  document.getElementById('tonic-ref').textContent = `Western Reference: ${state.tonic.tonicWestern}`;
  document.getElementById('stat-total-notes').textContent = state.swaraNotes.length;

  // Sync Tanpura and Sa Calibration bar with the identified tonic
  if (state.tonic && state.tonic.tonicFreq) {
    tanpura.setSa(state.tonic.tonicFreq);
    const tanpuraFreqLabel = document.getElementById('tanpura-freq-label');
    if (tanpuraFreqLabel) {
      tanpuraFreqLabel.textContent = `${state.tonic.tonicFreq} Hz`;
    }
    const hzPill = document.getElementById('sa-hz-indicator');
    if (hzPill) {
      hzPill.textContent = `${state.tonic.tonicFreq} Hz (${state.tonic.tonicWestern})`;
    }
    if (state.isAutoSa && state.tonic.tonicWestern) {
      const match = state.tonic.tonicWestern.match(/([A-G]#?)([0-9])/);
      if (match) {
        const rootEl = document.getElementById('sa-root-select');
        const octEl = document.getElementById('sa-octave-select');
        if (rootEl) rootEl.value = match[1];
        if (octEl) octEl.value = match[2];
      }
    }
  }

  const duration = state.audioBuffer.duration;
  document.getElementById('stat-duration').textContent = `${duration.toFixed(1)}s`;

  if (state.swaraNotes.length > 0) {
    const swaraCounts = {};
    for (const n of state.swaraNotes) {
      const cSwara = formatCarnaticSwara(n, state.raga);
      swaraCounts[cSwara.symbol] = (swaraCounts[cSwara.symbol] || 0) + 1;
    }
    const predom = Object.entries(swaraCounts).sort((a, b) => b[1] - a[1])[0];
    document.getElementById('stat-predom').textContent = predom ? predom[0] : 'S';

    const saptaks = [...new Set(state.swaraNotes.map(n => {
      const c = formatCarnaticSwara(n, state.raga);
      return `${c.sthayi} Sthayi`;
    }))];
    document.getElementById('stat-vocal-span').textContent = saptaks.join(', ');
  } else {
    document.getElementById('stat-predom').textContent = '—';
    document.getElementById('stat-vocal-span').textContent = '—';
  }

  // 3. Render Luminous Golden Waveform & Audio Player
  await renderWaveform();

  // 4. Render Horizontal Swaras Timeline (below waveform) & subtitle ribbon
  if (state.swaraNotes && state.swaraNotes.length > 0) {
    const swarasSection = document.getElementById('horizontal-swaras-section');
    if (swarasSection) swarasSection.style.display = '';
    renderHorizontalSwarasTrack();
    renderHorizontalTrackRuler(state.audioBuffer.duration);
    renderSubtitleRibbon();
    // Pre-set the subtitle card to the very first note
    if (state.swaraNotes[0]) {
      updateSubtitleDisplay(state.swaraNotes[0], 0);
    }
  }

  // Smoothly scroll down to the stage on the same page
  const stageEl = document.getElementById('classical-obsidian-stage');
  if (stageEl) {
    stageEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

async function renderWaveform() {
  const url = URL.createObjectURL(state.audioBlob);

  if (state.wavesurfer) {
    state.wavesurfer.destroy();
    state.wavesurfer = null;
  }

  // Background WaveSurfer for audio decoding and playback events
  state.wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: 'rgba(253, 203, 110, 0.45)',
    progressColor: '#FDCB6E',
    cursorColor: '#FD79A8',
    cursorWidth: 2,
    height: 60,
    normalize: true,
    interact: true,
    hideScrollbar: true,
    url,
  });

  const dur = state.audioBuffer ? state.audioBuffer.duration : 1;
  const aksharaSec = state.adiTalaData ? state.adiTalaData.aksharaSec : 0.53;
  const totalAksharas = Math.ceil(dur / aksharaSec);
  const beatTimestamps = Array.from({ length: totalAksharas }, (_, i) => i * aksharaSec);

  // Initialize Master Luminous Golden Waveform with Beat Pulse Beads
  state.luminousWaveform = new LuminousWaveform('classical-luminous-waveform', {
    onSeek: (time) => {
      if (state.wavesurfer) {
        if (typeof state.wavesurfer.setTime === 'function') {
          state.wavesurfer.setTime(time);
        } else {
          const totalDur = state.wavesurfer.getDuration() || dur;
          state.wavesurfer.seekTo(time / totalDur);
        }
        state.wavesurfer.play();
      }
    }
  });
  state.luminousWaveform.setAudioBuffer(state.audioBuffer, beatTimestamps);

  state.wavesurfer.on('ready', () => {
    const totalDur = state.wavesurfer.getDuration() || dur;
    const stageTiming = document.getElementById('stage-timing');
    if (stageTiming) stageTiming.textContent = `0:00 / ${formatDuration(totalDur)}`;
  });

  // timeupdate: only update timing label (heavy sync moved to 60FPS RAF loop)
  state.wavesurfer.on('timeupdate', (time) => {
    const totalDur = state.wavesurfer.getDuration() || dur;
    const stageTiming = document.getElementById('stage-timing');
    if (stageTiming) stageTiming.textContent = `${formatDuration(time)} / ${formatDuration(totalDur)}`;
  });

  state.wavesurfer.on('play', () => {
    state.isPlaying = true;
    const stagePlayBtn = document.getElementById('stage-play-btn');
    if (stagePlayBtn) stagePlayBtn.textContent = '⏸';
    // Stop top hand autonomous loop — audio sync takes over
    if (state.topHandStage && state.topHandStage.isPlaying) {
      state.topHandStage.stopLoop();
      const playIcon = document.getElementById('top-tala-play-icon');
      const playText = document.getElementById('top-tala-play-text');
      const playBtn = document.getElementById('top-tala-play-btn');
      if (playIcon) playIcon.textContent = '▶';
      if (playText) playText.textContent = 'Play Tala';
      playBtn?.classList.remove('playing');
    }
    // Reset sync counters so beat transitions fire immediately
    if (state.topHandStage) state.topHandStage.resetSync();
    if (state.handStage) state.handStage.resetSync();
    state.activeNoteIndex = -1;
    startPlaybackSyncLoop();
  });

  state.wavesurfer.on('pause', () => {
    state.isPlaying = false;
    stopPlaybackSyncLoop();
    const stagePlayBtn = document.getElementById('stage-play-btn');
    if (stagePlayBtn) stagePlayBtn.textContent = '▶';
  });

  state.wavesurfer.on('finish', () => {
    state.isPlaying = false;
    stopPlaybackSyncLoop();
    const stagePlayBtn = document.getElementById('stage-play-btn');
    if (stagePlayBtn) stagePlayBtn.textContent = '▶';
    if (state.luminousWaveform) state.luminousWaveform.setTime(0);
    if (state.handStage) state.handStage.animateToKriya(1, false);
    if (state.topHandStage) state.topHandStage.setBeat(1, false);
    // Clear all active highlights
    document.querySelectorAll('.score-cell.active-akshara, .score-cell.active-score-beat, .score-cell.active-sung-note').forEach(el => {
      el.classList.remove('active-akshara', 'active-score-beat', 'active-sung-note');
    });
    document.querySelectorAll('.horizontal-swara-block.active-sung-note').forEach(el => el.classList.remove('active-sung-note'));
    document.querySelectorAll('.sub-ribbon-item.active').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.score-avartanam-row.active-avartanam').forEach(el => el.classList.remove('active-avartanam'));
    state.activeNoteIndex = -1;
  });
}


// ── 60 FPS Playback Sync Engine ──────────────────────────────────────────────
function startPlaybackSyncLoop() {
  stopPlaybackSyncLoop();

  const loop = () => {
    if (!state.isPlaying || !state.wavesurfer) return;

    const time = state.wavesurfer.getCurrentTime();
    const totalDur = state.wavesurfer.getDuration() || (state.audioBuffer?.duration || 1);
    const aksharaSec = (state.talaData || state.adiTalaData)?.aksharaSec || (60 / (state.detectedBpm || 76));

    // 1. Sync luminous waveform playhead
    if (state.luminousWaveform) {
      state.luminousWaveform.setTime(time);
    }

    // 2. Sync horizontal swaras track playhead needle
    const playhead = document.getElementById('horizontal-swaras-playhead');
    if (playhead && totalDur > 0) {
      playhead.style.left = `${(time / totalDur) * 100}%`;
    }

    // 3. Sync Top Tala Hand Stage with audio
    if (state.topHandStage) {
      state.topHandStage.syncWithTime(time, aksharaSec);
    }

    // 4. Sync Bottom Tala Hand Stage with audio (all talas, not just adi)
    if (state.carnaticMode === 'tala' && state.handStage && (state.talaData || state.adiTalaData)) {
      state.handStage.syncWithTime(time, aksharaSec);
    }

    // 5. Sync Tala Score Grid active akshara cell + avartanam row
    if (state.carnaticMode === 'tala') {
      updateActiveAksharaAtTime(time);
    }

    // 6. Sync active swara note, subtitle ribbon, and cinema subtitle card
    updateActiveNoteAtTime(time);

    // 7. Sync free-flow phrases if in that mode
    if (state.carnaticMode === 'freeflow') {
      updateActiveFreeflowPhraseAtTime(time);
    }

    state.rafLoopId = requestAnimationFrame(loop);
  };

  state.rafLoopId = requestAnimationFrame(loop);
}

function stopPlaybackSyncLoop() {
  if (state.rafLoopId) {
    cancelAnimationFrame(state.rafLoopId);
    state.rafLoopId = null;
  }
}

// ── Render Horizontal Swaras Track (Directly Under Waveform) ────────────────
function renderHorizontalSwarasTrack() {
  const blocksContainer = document.getElementById('horizontal-swaras-blocks');
  if (!blocksContainer) return;

  blocksContainer.innerHTML = '';
  const totalDur = state.audioBuffer.duration || 1;

  state.swaraNotes.forEach((note, idx) => {
    const leftPct = (note.startTime / totalDur) * 100;
    const widthPct = Math.max(1.2, (note.duration / totalDur) * 100);

    const cSwara = formatCarnaticSwara(note, state.raga);

    // Determine Swara color category
    let typeClass = 'swara-type-shuddha';
    if (note.swara.symbol === 'S') typeClass = 'swara-type-sa';
    else if (note.swara.symbol === 'P') typeClass = 'swara-type-pa';
    else if (note.swara.type === 'komal') typeClass = 'swara-type-komal';
    else if (note.swara.type === 'teevra') typeClass = 'swara-type-teevra';

    const saptakDot = cSwara.sthayi === 'Taar' ? '̇' : (cSwara.sthayi === 'Mandra' ? '̣' : '');

    const block = document.createElement('div');
    block.className = `horizontal-swara-block ${typeClass}`;
    block.id = `swara-block-${idx}`;
    block.dataset.index = idx;
    block.dataset.start = note.startTime;
    block.dataset.end = note.endTime;
    block.style.left = `${leftPct}%`;
    block.style.width = `${widthPct}%`;

    const gamakaGlyph = note.gamaka?.type === 'kampitam' ? '〰' : (note.gamaka?.type === 'jaaru' ? (note.gamaka.glideDirection === 'ascending' ? '↗' : '↘') : '');

    block.innerHTML = `
      <div class="swara-block-top">
        <span class="swara-block-name">${cSwara.symbol}</span>
        ${gamakaGlyph ? `<span class="swara-block-gamaka" title="${note.gamaka.label}">${gamakaGlyph}</span>` : ''}
      </div>
      <div class="swara-block-hz">${note.meanHz}Hz</div>
    `;

    block.title = `${cSwara.fullName} (${cSwara.symbol})\nTime: ${note.startTime}s - ${note.endTime}s (${note.duration}s)\nPitch: ${note.meanHz}Hz (${note.centsDeviation >= 0 ? '+' : ''}${note.centsDeviation}¢)${note.gamaka?.label ? '\nGamaka: ' + note.gamaka.label : ''}\nClick to jump here`;

    // Click note block to jump audio & subtitles directly
    block.addEventListener('click', (e) => {
      e.stopPropagation();
      seekToNote(note, idx);
    });

    blocksContainer.appendChild(block);
  });
}

function renderHorizontalTrackRuler(duration) {
  const ruler = document.getElementById('horizontal-swaras-ruler');
  if (!ruler) return;
  ruler.innerHTML = '';

  const step = duration > 30 ? 5 : (duration > 10 ? 2 : 1);
  for (let t = 0; t <= duration; t += step) {
    const pct = (t / duration) * 100;
    const isMajor = t % (step * 2) === 0;

    const tick = document.createElement('div');
    tick.className = `ruler-tick${isMajor ? ' major' : ''}`;
    tick.style.left = `${pct}%`;
    ruler.appendChild(tick);

    if (isMajor || t === 0 || t + step > duration) {
      const label = document.createElement('div');
      label.className = 'ruler-label';
      label.style.left = `${pct}%`;
      label.textContent = `${t}s`;
      ruler.appendChild(label);
    }
  }
}

// ── Subtitle Display & Active Note Tracker ──────────────────────────────────
function updateActiveNoteAtTime(time) {
  if (!state.swaraNotes || state.swaraNotes.length === 0) return;

  // Primary: find exact note containing current time
  let noteIdx = state.swaraNotes.findIndex(n => time >= n.startTime && time <= n.endTime);

  // Fallback: during micro-gaps between notes, use the most recently passed note
  // so the subtitle card doesn't go blank mid-playback
  if (noteIdx === -1) {
    for (let i = state.swaraNotes.length - 1; i >= 0; i--) {
      if (time > state.swaraNotes[i].endTime) {
        // Only sustain if within 0.35s of note end (typical inter-note gap)
        if (time - state.swaraNotes[i].endTime < 0.35) {
          noteIdx = i;
        }
        break;
      }
    }
  }

  if (noteIdx !== -1 && noteIdx !== state.activeNoteIndex) {
    state.activeNoteIndex = noteIdx;
    const note = state.swaraNotes[noteIdx];

    // Highlight horizontal block
    document.querySelectorAll('.horizontal-swara-block.active-sung-note').forEach(b => b.classList.remove('active-sung-note'));
    const activeBlock = document.getElementById(`swara-block-${noteIdx}`);
    if (activeBlock) {
      activeBlock.classList.add('active-sung-note');
      activeBlock.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }

    // Highlight subtitle ribbon
    document.querySelectorAll('.sub-ribbon-item.active').forEach(r => r.classList.remove('active'));
    const activeRibbon = document.getElementById(`sub-ribbon-${noteIdx}`);
    if (activeRibbon) {
      activeRibbon.classList.add('active');
      activeRibbon.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }

    // Update Cinema Subtitle Display Card
    updateSubtitleDisplay(note, time);
  }
}

function updateSubtitleDisplay(note, currentTime) {
  if (!note) return;

  const cSwara = formatCarnaticSwara(note, state.raga);
  const saptakDot = cSwara.sthayi === 'Taar' ? '̇' : (cSwara.sthayi === 'Mandra' ? '̣' : '');

  const elSwara = document.getElementById('sub-swara-name');
  if (elSwara) elSwara.textContent = cSwara.symbol;
  const elDeva = document.getElementById('sub-swara-devanagari');
  if (elDeva) elDeva.textContent = cSwara.fullName;
  const elDot = document.getElementById('sub-saptak-dot');
  if (elDot) elDot.textContent = saptakDot;

  const elFullName = document.getElementById('sub-full-name');
  if (elFullName) elFullName.textContent = `${cSwara.fullName} · ${cSwara.sthayi} Sthayi`;
  const elSaptak = document.getElementById('sub-saptak-badge');
  if (elSaptak) elSaptak.textContent = `${cSwara.sthayi} Sthayi`;
  const elPitch = document.getElementById('sub-pitch-hz');
  if (elPitch) elPitch.textContent = `${note.meanHz} Hz`;

  const western = hzToWesternNote(note.meanHz);
  const elWest = document.getElementById('sub-western-ref');
  if (elWest) elWest.textContent = `Western: ${western.noteStr}`;

  // Intonation Needle & Badge
  const dev = note.centsDeviation;
  const needle = document.getElementById('sub-cents-needle');
  const needlePct = Math.max(0, Math.min(100, 50 + (dev / 50) * 50));
  if (needle) needle.style.left = `${needlePct}%`;

    const badge = document.getElementById('sub-cents-badge');
  if (badge) {
    badge.className = 'sub-cents-badge';
    if (Math.abs(dev) <= 12) {
      badge.classList.add('in-tune');
      badge.textContent = `${dev >= 0 ? '+' : ''}${dev}¢ In-Tune`;
    } else if (dev < -12) {
      badge.classList.add('flat');
      badge.textContent = `${dev}¢ Flat`;
    } else {
      badge.classList.add('sharp');
      badge.textContent = `+${dev}¢ Sharp`;
    }
  }

  // Dual-Stream Gamaka Badge
  const gamakaBadge = document.getElementById('sub-gamaka-badge');
  if (gamakaBadge) {
    if (note.gamaka && note.gamaka.label) {
      gamakaBadge.style.display = 'inline-flex';
      gamakaBadge.textContent = note.gamaka.label;
      gamakaBadge.className = `sub-gamaka-badge gamaka-${note.gamaka.type || 'steady'}`;
    } else {
      gamakaBadge.style.display = 'none';
    }
  }
}

function renderSubtitleRibbon() {
  const ribbon = document.getElementById('subtitle-ribbon');
  if (!ribbon) return;

  ribbon.innerHTML = state.swaraNotes.map((n, i) => {
    const cSwara = formatCarnaticSwara(n, state.raga);
    return `
      <div class="sub-ribbon-item${i === 0 ? ' active' : ''}" id="sub-ribbon-${i}" data-index="${i}">
        <strong>${cSwara.symbol}</strong>
        <span>${cSwara.fullName}</span>
        <small style="opacity: 0.6;">${n.startTime}s</small>
      </div>
    `;
  }).join('');

  ribbon.querySelectorAll('.sub-ribbon-item').forEach(item => {
    item.addEventListener('click', () => {
      const idx = parseInt(item.dataset.index);
      if (state.swaraNotes[idx]) {
        seekToNote(state.swaraNotes[idx], idx);
      }
    });
  });
}

function seekToNote(note, index) {
  if (!state.wavesurfer) return;
  const dur = state.wavesurfer.getDuration();
  if (dur > 0) {
    if (typeof state.wavesurfer.setTime === 'function') {
      state.wavesurfer.setTime(note.startTime);
    } else {
      state.wavesurfer.seekTo(note.startTime / dur);
    }
    state.wavesurfer.play();
  }
  state.activeNoteIndex = index;
  const playhead = document.getElementById('horizontal-swaras-playhead');
  if (playhead && dur > 0) {
    playhead.style.left = `${(note.startTime / dur) * 100}%`;
  }
  document.querySelectorAll('.horizontal-swara-block').forEach(b => b.classList.remove('active-sung-note'));
  const activeBlock = document.getElementById(`swara-block-${index}`);
  if (activeBlock) {
    activeBlock.classList.add('active-sung-note');
  }
  updateSubtitleDisplay(note, note.startTime);
}

// ── Carnatic Kriti Notation & Master Tala Studio ─────────────────────────
function renderCarnaticStudio() {
  const dur = state.audioBuffer ? state.audioBuffer.duration : 1;
  state.beatAnalysis = detectBeatStructure(state.swaraNotes, dur, state.detectedBpm);

  // 1. Identify Raga first so all swarasthanas are raga-aware
  const raga = identifyCarnaticRaga(state.swaraNotes);
  state.raga = raga;

  // 2. Auto-detect Tala from musical onsets, Samam downbeat alignment & periodicity
  const detected = detectCarnaticTala(state.swaraNotes, dur, state.beatAnalysis.aksharaSec);
  state.autoDetectedTala = detected;

  // Choose Tala based on auto vs manual user setting
  const activeTalaToApply = state.talaMode === 'auto' ? detected.detectedTalaId : state.activeTalaId;
  state.activeTalaId = activeTalaToApply;
  state.talaData = buildTalaGrid(state.swaraNotes, dur, state.beatAnalysis.aksharaSec, raga, activeTalaToApply);
  state.adiTalaData = state.talaData;

  state.freeFlowData = buildFreeFlowPhrases(state.swaraNotes, raga);

  // Automatically update and lock the Top Tala Stage to the identified probable BPM!
  const finalBpm = state.talaData.bpm || state.beatAnalysis.estimatedBpm;
  applyDetectedBpm(finalBpm, state.beatAnalysis.kala, 'Refined Swara Beat Quantization');

  // Auto-detect beat structure vs free-flow per user requirement
  if (state.beatAnalysis.hasBeatStructure) {
    state.carnaticMode = 'tala';
  } else {
    state.carnaticMode = 'freeflow';
  }

  // Update Master Masthead Header
  const titleEl = document.getElementById('classical-stage-title');
  const subtitleEl = document.getElementById('classical-stage-subtitle');
  const talaDef = CARNATIC_TALAS[state.activeTalaId] || CARNATIC_TALAS.adi;

  if (titleEl) {
    titleEl.textContent = state.carnaticMode === 'tala' 
      ? `${talaDef.name} — ${talaDef.nadai}` 
      : 'Alapana & Virutham — Free Flow';
  }

  if (subtitleEl) {
    const saHz = state.tonic ? Math.round(state.tonic.tonicFreq) : 146;
    const westernRef = state.tonic?.tonicWestern ? state.tonic.tonicWestern.split('/')[0] : 'D';
    const ragaUpper = (raga.name || 'MAYAMALAVAGOWLA').toUpperCase();
    subtitleEl.textContent = `${ragaUpper} — MID · SA = ${saHz} HZ (${westernRef})`;
  }

  // Update Tempo Pill
  const tempoPill = document.getElementById('stage-tempo-pill');
  if (tempoPill && state.talaData) {
    tempoPill.textContent = `${state.talaData.bpm} BPM`;
  }

  // Initialize Master Classical Hand Illustration Rig
  if (state.handStage) {
    state.handStage.stopLoop();
  }
  state.handStage = new CarnaticHandStage('classical-hand-canvas-wrap', {
    talaId: state.activeTalaId,
  });

  // Sync Tala select dropdown & auto badge
  setCarnaticTala(state.activeTalaId, state.talaMode === 'auto');

  // Render Golden Architectural Tala Score Grid
  renderTalaScoreBoard();

  // Render Free-flow phrases
  renderFreeFlowPhrases();

  // Apply active mode
  setCarnaticMode(state.carnaticMode);
}

function setCarnaticTala(talaId, isAuto = false) {
  const talaDef = CARNATIC_TALAS[talaId] || CARNATIC_TALAS.adi;
  state.activeTalaId = talaDef.id;

  // Sync dropdown
  const selectEl = document.getElementById('stage-tala-select');
  if (selectEl) {
    if (state.talaMode === 'auto' && isAuto) {
      selectEl.value = 'auto';
    } else {
      selectEl.value = talaDef.id;
    }
  }

  // Update auto badge
  const autoBadge = document.getElementById('tala-auto-badge');
  if (autoBadge) {
    if (isAuto && state.autoDetectedTala) {
      const confPct = Math.round(state.autoDetectedTala.confidence * 100);
      autoBadge.textContent = `⚡ Auto: ${talaDef.shortName} (${confPct}%)`;
      autoBadge.style.display = 'inline-flex';
    } else {
      autoBadge.style.display = 'none';
    }
  }

  // Update hand stage
  if (state.handStage) {
    state.handStage.setTala(talaDef.id);
  }

  // If notes exist, re-quantize into this Tala
  if (state.swaraNotes && state.swaraNotes.length > 0) {
    const dur = state.audioBuffer ? state.audioBuffer.duration : 1;
    const aksharaSec = state.beatAnalysis ? state.beatAnalysis.aksharaSec : null;
    state.talaData = buildTalaGrid(state.swaraNotes, dur, aksharaSec, state.raga, talaDef.id);
    state.adiTalaData = state.talaData;

    // Update Tempo Pill
    const tempoPill = document.getElementById('stage-tempo-pill');
    if (tempoPill) tempoPill.textContent = `${state.talaData.bpm} BPM`;

    renderTalaScoreBoard();
  } else {
    // Initial ready state
    renderReadyTalaGrid();
  }

  // Update title if in tala mode
  const titleEl = document.getElementById('classical-stage-title');
  if (titleEl && state.carnaticMode === 'tala') {
    titleEl.textContent = `${talaDef.name} — ${talaDef.nadai}`;
  }
}

function autoDetectAndApplyTala() {
  const dur = state.audioBuffer ? state.audioBuffer.duration : 1;
  const aksharaSec = state.beatAnalysis ? state.beatAnalysis.aksharaSec : null;
  const detected = detectCarnaticTala(state.swaraNotes, dur, aksharaSec);
  state.autoDetectedTala = detected;
  setCarnaticTala(detected.detectedTalaId, true);
}

function setCarnaticMode(mode) {
  state.carnaticMode = mode;
  const scoreBoard = document.getElementById('classical-score-board');
  const freeflowBoard = document.getElementById('classical-freeflow-board');
  const btnTala = document.getElementById('btn-stage-tala');
  const btnFreeflow = document.getElementById('btn-stage-freeflow');
  const titleEl = document.getElementById('classical-stage-title');
  const talaDef = CARNATIC_TALAS[state.activeTalaId] || CARNATIC_TALAS.adi;

  if (mode === 'tala') {
    if (scoreBoard) scoreBoard.style.display = 'block';
    if (freeflowBoard) freeflowBoard.style.display = 'none';
    if (btnTala) btnTala.classList.add('active');
    if (btnFreeflow) btnFreeflow.classList.remove('active');
    if (titleEl) titleEl.textContent = `${talaDef.name} — ${talaDef.nadai}`;
  } else {
    if (scoreBoard) scoreBoard.style.display = 'none';
    if (freeflowBoard) freeflowBoard.style.display = 'block';
    if (btnTala) btnTala.classList.remove('active');
    if (btnFreeflow) btnFreeflow.classList.add('active');
    if (titleEl) titleEl.textContent = 'Alapana & Virutham — Free Flow';
  }
}

function renderTalaScoreBoard() {
  const container = document.getElementById('score-avartanams-list');
  const currentData = state.talaData || state.adiTalaData;
  if (!container || !currentData) return;

  const tala = CARNATIC_TALAS[state.activeTalaId] || CARNATIC_TALAS.adi;

  // Update Anga Header
  const angaHeader = document.querySelector('.score-anga-header');
  if (angaHeader) {
    angaHeader.innerHTML = `
      <div class="score-row-tag-spacer"></div>
      ${tala.angas.map((anga, aIdx) => {
        const divider = (aIdx < tala.angas.length - 1) ? '<div class="score-col-divider">|</div>' : '';
        return `<div class="score-anga-col" style="flex: ${anga.count};">${anga.label}</div>${divider}`;
      }).join('')}
    `;
  }

  const avartanams = currentData.avartanams || [];
  const totalAvartanams = avartanams.length;
  const linesPerPage = 4;
  const maxWindow = Math.max(0, Math.ceil(totalAvartanams / linesPerPage) - 1);
  if (state.scoreLineWindow > maxWindow) state.scoreLineWindow = maxWindow;
  if (state.scoreLineWindow < 0) state.scoreLineWindow = 0;

  const startIdx = state.scoreLineWindow * linesPerPage;
  const endIdx = Math.min(totalAvartanams, startIdx + linesPerPage);

  // Update Window Navigation label & button states
  const winText = document.getElementById('score-window-text');
  const prevBtn = document.getElementById('score-prev-page');
  const nextBtn = document.getElementById('score-next-page');
  const navWrap = document.getElementById('score-window-nav');
  
  if (totalAvartanams > 0) {
    if (navWrap) navWrap.style.display = totalAvartanams > 4 ? 'flex' : 'none';
    if (winText) winText.textContent = `Lines ${startIdx + 1}–${endIdx} of ${totalAvartanams}`;
    if (prevBtn) prevBtn.disabled = (state.scoreLineWindow <= 0);
    if (nextBtn) nextBtn.disabled = (state.scoreLineWindow >= maxWindow);
  } else if (navWrap) {
    navWrap.style.display = 'none';
  }

  container.innerHTML = '';

  for (let avIdx = startIdx; avIdx < endIdx; avIdx++) {
    const av = avartanams[avIdx];
    const row = document.createElement('div');
    row.className = 'score-avartanam-row';
    row.id = `score-avartanam-${avIdx}`;

    // Line number badge
    const tag = document.createElement('div');
    tag.className = 'score-row-tag';
    tag.textContent = `Line ${avIdx + 1}`;
    row.appendChild(tag);

    let cellOffset = 0;
    const angasHtml = tala.angas.map((anga, aIdx) => {
      const angaCells = av.cells.slice(cellOffset, cellOffset + anga.count);
      cellOffset += anga.count;
      const divider = (aIdx < tala.angas.length - 1) ? '<div class="score-col-divider">|</div>' : '';
      return `
        <div class="score-anga-cells" style="flex: ${anga.count};">
          ${angaCells.map(c => renderScoreCellHtml(c)).join('')}
        </div>
        ${divider}
      `;
    }).join('');

    const rowContent = document.createElement('div');
    rowContent.className = 'score-row-content';
    rowContent.innerHTML = angasHtml;
    row.appendChild(rowContent);

    // Click handler for cells to seek audio
    row.querySelectorAll('.score-cell').forEach(cellEl => {
      cellEl.addEventListener('click', (e) => {
        e.stopPropagation();
        const start = parseFloat(cellEl.dataset.start);
        if (!isNaN(start) && state.wavesurfer) {
          if (typeof state.wavesurfer.setTime === 'function') {
            state.wavesurfer.setTime(start);
          } else {
            const dur = state.wavesurfer.getDuration() || state.audioBuffer.duration;
            state.wavesurfer.seekTo(start / dur);
          }
          state.wavesurfer.play();
        }
      });
    });

    container.appendChild(row);
  }
}

function renderScoreCellHtml(cell) {
  let swaraClass = '';
  if (cell.display === ',' || cell.display === '·') swaraClass += ' karvai';
  else if (cell.display === ';') swaraClass += ' semicolon';
  if (cell.isDoubleOverline) swaraClass += ' double-overline';
  else if (cell.isOverline) swaraClass += ' overline';

  const kriyaShort = cell.kriya ? (cell.kriya.name || cell.kriya.label || '') : '';
  const kriyaSymbol = cell.kriya ? (cell.kriya.symbol || '') : '';

  return `
    <div class="score-cell${swaraClass}" id="score-cell-${cell.aksharaNumber}" data-akshara="${cell.aksharaNumber}" data-beat="${cell.beat}" data-start="${cell.startTime}" data-end="${cell.endTime}" title="Akshara ${cell.aksharaNumber} (Beat ${cell.beat}) · ${kriyaShort}\n${cell.display} · Time: ${cell.startTime}s - ${cell.endTime}s\nClick to seek audio">
      <div class="score-cell-top-meta">
        <span class="score-beat-indicator">B${cell.beat}</span>
        ${kriyaSymbol ? `<span class="score-kriya-symbol">${kriyaSymbol}</span>` : ''}
      </div>
      <span class="score-swara-text">${cell.display}</span>
      ${kriyaShort ? `<span class="score-kriya-badge">${kriyaShort.split(' ')[0]}</span>` : ''}
    </div>
  `;
}

function renderFreeFlowPhrases() {
  const container = document.getElementById('freeflow-phrases-container');
  if (!container || !state.freeFlowData) return;

  container.innerHTML = '';

  if (state.freeFlowData.length === 0) {
    container.innerHTML = '<div style="color: var(--text-tertiary); text-align: center; padding: 20px;">No musical phrases detected.</div>';
    return;
  }

  // Windowing 4 phrases at a time for unmetered flow as well
  const totalPhrases = state.freeFlowData.length;
  const startIdx = state.scoreLineWindow * 4;
  const endIdx = Math.min(totalPhrases, startIdx + 4);

  for (let pIdx = startIdx; pIdx < endIdx; pIdx++) {
    const phrase = state.freeFlowData[pIdx];
    const card = document.createElement('div');
    card.className = 'freeflow-phrase-card';
    card.id = `freeflow-phrase-${pIdx}`;

    card.innerHTML = `
      <div class="phrase-header">
        <span class="phrase-index">Line ${pIdx + 1} · Phrase ${phrase.phraseNumber} (Prayoga)</span>
        <span>${phrase.startTime.toFixed(2)}s – ${phrase.endTime.toFixed(2)}s (${phrase.duration}s)</span>
      </div>
      <div class="phrase-sargam-box">${phrase.sargamText}</div>
      <div class="phrase-chips">
        ${phrase.notes.map((n, nIdx) => `
          <span class="phrase-chip" data-start="${n.startTime}" data-end="${n.endTime}" title="${n.swara?.fullName || ''} (${n.meanHz}Hz)">
            <strong>${n.cFmt.symbol}</strong>
            <small style="opacity: 0.7;">${n.startTime}s</small>
          </span>
        `).join('')}
      </div>
    `;

    // Click handler for phrases
    card.addEventListener('click', () => {
      if (state.wavesurfer) {
        if (typeof state.wavesurfer.setTime === 'function') {
          state.wavesurfer.setTime(phrase.startTime);
        } else {
          const dur = state.wavesurfer.getDuration() || state.audioBuffer.duration;
          state.wavesurfer.seekTo(phrase.startTime / dur);
        }
        state.wavesurfer.play();
      }
    });

    // Click on individual chips within phrase
    card.querySelectorAll('.phrase-chip').forEach(chip => {
      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        const start = parseFloat(chip.dataset.start);
        if (!isNaN(start) && state.wavesurfer) {
          if (typeof state.wavesurfer.setTime === 'function') {
            state.wavesurfer.setTime(start);
          } else {
            const dur = state.wavesurfer.getDuration() || state.audioBuffer.duration;
            state.wavesurfer.seekTo(start / dur);
          }
          state.wavesurfer.play();
        }
      });
    });

    container.appendChild(card);
  }
}

function updateActiveAksharaAtTime(time) {
  const currentTalaData = state.talaData || state.adiTalaData;
  if (!currentTalaData) return;
  const aksharaSec = currentTalaData.aksharaSec || 0.53;
  const totalBeats = currentTalaData.totalBeats || 8;
  const currentAksharaNum = Math.floor(time / aksharaSec) + 1;
  const currentAvartanamIdx = Math.floor((currentAksharaNum - 1) / totalBeats);

  // Auto-page 4-line window if current line moves beyond visible 4 lines!
  const totalAvartanams = currentTalaData.avartanams ? currentTalaData.avartanams.length : 1;
  const targetWindow = Math.floor(currentAvartanamIdx / 4);
  const maxWindow = Math.max(0, Math.ceil(totalAvartanams / 4) - 1);
  const boundedTarget = Math.min(maxWindow, Math.max(0, targetWindow));

  if (boundedTarget !== state.scoreLineWindow && currentAvartanamIdx < totalAvartanams) {
    state.scoreLineWindow = boundedTarget;
    renderTalaScoreBoard();
  }

  // Highlight active line
  document.querySelectorAll('.score-avartanam-row.active-avartanam').forEach(el => el.classList.remove('active-avartanam'));
  const activeRow = document.getElementById(`score-avartanam-${currentAvartanamIdx}`);
  if (activeRow) {
    activeRow.classList.add('active-avartanam');
  }

  // Highlight active beat / note cell
  document.querySelectorAll('.score-cell.active-akshara, .score-cell.active-sung-note, .score-cell.active-score-beat').forEach(el => {
    el.classList.remove('active-akshara', 'active-sung-note', 'active-score-beat');
  });
  const activeCell = document.getElementById(`score-cell-${currentAksharaNum}`);
  if (activeCell) {
    activeCell.classList.add('active-akshara', 'active-sung-note', 'active-score-beat');
  }
}

function updateActiveFreeflowPhraseAtTime(time) {
  if (!state.freeFlowData) return;
  const phraseIdx = state.freeFlowData.findIndex(p => time >= p.startTime && time <= p.endTime);
  document.querySelectorAll('.freeflow-phrase-card.active-phrase').forEach(el => el.classList.remove('active-phrase'));
  if (phraseIdx !== -1) {
    const card = document.getElementById(`freeflow-phrase-${phraseIdx}`);
    if (card) card.classList.add('active-phrase');
  }
}

function togglePlayback() {
  if (state.wavesurfer) {
    state.wavesurfer.playPause();
  } else if (state.handStage) {
    const isPlaying = state.handStage.togglePlay();
    const playBtn = document.getElementById('stage-play-btn');
    if (playBtn) playBtn.textContent = isPlaying ? '❚❚' : '▶';
  }
}

// ── Initial Classical Stage Ready State (1-Page Synchronizer) ─────────────
function initClassicalStageReady() {
  const handWrap = document.getElementById('classical-hand-canvas-wrap');
  if (!handWrap) return;

  state.activeTalaId = 'adi';
  const tala = CARNATIC_TALAS[state.activeTalaId];

  // Initialize CarnaticHandStage with autonomous beat synchronization
  state.handStage = new CarnaticHandStage('classical-hand-canvas-wrap', {
    bpm: 88,
    talaId: state.activeTalaId,
    onBeatChange: (beat) => {
      // Highlight ready cell across Line 1 if playing autonomous loop before audio load
      if (!state.wavesurfer) {
        document.querySelectorAll('.score-cell').forEach(c => c.classList.remove('active-beat-cell', 'active-sung-note', 'active-akshara'));
        document.querySelectorAll('.score-avartanam-row').forEach(r => r.classList.remove('active-avartanam'));
        const activeRow = document.getElementById('score-avartanam-ready-0');
        if (activeRow) activeRow.classList.add('active-avartanam');
        const cell = document.getElementById(`score-cell-ready-${beat - 1}`);
        if (cell) cell.classList.add('active-sung-note');
      }
    }
  });

  // Set default stage masthead subtitle
  const subtitleEl = document.getElementById('classical-stage-subtitle');
  if (subtitleEl) {
    subtitleEl.textContent = `${tala.name.toUpperCase()} · ${tala.nadai.toUpperCase()} · 88 BPM · SA = 146.8 HZ (D3)`;
  }

  // Set tempo pill
  const tempoPill = document.getElementById('stage-tempo-pill');
  if (tempoPill) {
    tempoPill.textContent = '88 BPM';
  }

  // Render initial ready 4-Line Tala Score Grid
  renderReadyTalaGrid();

  // Set mode to tala
  setCarnaticMode('tala');
}

function renderReadyTalaGrid() {
  const container = document.getElementById('score-avartanams-list');
  if (!container) return;

  const tala = CARNATIC_TALAS[state.activeTalaId] || CARNATIC_TALAS.adi;

  // Update Anga Header
  const angaHeader = document.querySelector('.score-anga-header');
  if (angaHeader) {
    angaHeader.innerHTML = `
      <div class="score-row-tag-spacer"></div>
      ${tala.angas.map((anga, aIdx) => {
        const divider = (aIdx < tala.angas.length - 1) ? '<div class="score-col-divider">|</div>' : '';
        return `<div class="score-anga-col" style="flex: ${anga.count};">${anga.label}</div>${divider}`;
      }).join('')}
    `;
  }

  // Classical 4-line Sarali Varisai exercise for initial ready state
  const readyLines = [
    { num: 1, swaras: ['S', 'R', 'G', 'M', 'P', 'D', 'N', 'Ṡ'] },
    { num: 2, swaras: ['Ṡ', 'N', 'D', 'P', 'M', 'G', 'R', 'S'] },
    { num: 3, swaras: ['S', 'R', 'S', 'R', 'S', 'R', 'G', 'M'] },
    { num: 4, swaras: ['S', 'R', 'G', 'M', 'P', 'D', 'N', 'Ṡ'] },
  ];

  container.innerHTML = readyLines.map((line) => {
    let cellOffset = 0;
    const angasHtml = tala.angas.map((anga, aIdx) => {
      const beats = tala.beats.slice(cellOffset, cellOffset + anga.count);
      const swarasInAnga = line.swaras.slice(cellOffset, cellOffset + anga.count);
      const startIdx = cellOffset;
      cellOffset += anga.count;
      const divider = (aIdx < tala.angas.length - 1) ? '<div class="score-col-divider">|</div>' : '';

      return `
        <div class="score-anga-cells" style="flex: ${anga.count};">
          ${beats.map((b, bi) => {
            const sw = swarasInAnga[bi] || 'S';
            const beatGlobal = (line.num - 1) * tala.totalBeats + (startIdx + bi);
            return `
              <div class="score-cell" id="score-cell-ready-${beatGlobal}" data-beat="${b.beat}" title="Beat ${b.beat} (${b.name}) · ${sw}">
                <div class="score-cell-top-meta">
                  <span class="score-beat-indicator">B${b.beat}</span>
                  <span class="score-kriya-symbol">${b.symbol || ''}</span>
                </div>
                <span class="score-swara-text">${sw}</span>
                <span class="score-kriya-badge">${b.name.split(' ')[0]}</span>
              </div>
            `;
          }).join('')}
        </div>
        ${divider}
      `;
    }).join('');

    return `
      <div class="score-avartanam-row" id="score-avartanam-ready-${line.num - 1}">
        <div class="score-row-tag">Line ${line.num}</div>
        <div class="score-row-content">${angasHtml}</div>
      </div>
    `;
  }).join('');
}

// ── Export Utilities ────────────────────────────────────────────────────────
function exportSargamFile() {
  if (state.swaraNotes.length === 0) {
    showToast('No Swaras detected to export', true);
    return;
  }

  const raga = state.raga || identifyCarnaticRaga(state.swaraNotes);
  const activeTala = CARNATIC_TALAS[state.activeTalaId] || CARNATIC_TALAS.adi;
  const currentTalaData = state.talaData || state.adiTalaData;

  let text = `=================================================================\n`;
  text += ` NOTESCOPE — CARNATIC CLASSICAL KRITI NOTATION MANUSCRIPT\n`;
  text += `=================================================================\n\n`;
  text += `Raga:             ${raga.name}\n`;
  text += `Arohana/Avarohana:${raga.swarasthanas.join(' ')} Ṡ\n`;
  text += `Tala:             ${activeTala.name} (${activeTala.sign})\n`;
  text += `Tempo / Akshara:  ${currentTalaData ? `${currentTalaData.bpm} BPM (${currentTalaData.aksharaSec}s / Akshara)` : '111 BPM'}\n`;
  text += `Auto-Detected Sa: ${state.tonic.tonicFreq} Hz (${state.tonic.tonicWestern})\n`;
  text += `Total Swaras:     ${state.swaraNotes.length}\n`;
  text += `Audio Duration:   ${state.audioBuffer.duration.toFixed(2)} seconds\n`;
  text += `Beat Structure:   ${state.beatAnalysis ? (state.beatAnalysis.hasBeatStructure ? `Identified (${state.beatAnalysis.estimatedBpm} BPM)` : 'Free Flow / Alapana') : activeTala.name}\n`;
  text += `-----------------------------------------------------------------\n\n`;

  if (currentTalaData && currentTalaData.avartanams.length > 0) {
    text += `CARNATIC KRITI NOTATION — ${activeTala.name.toUpperCase()} (${activeTala.sign}):\n\n`;
    text += `Angas:     |  ${activeTala.angas.map(a => a.label.padEnd(a.count * 6, ' ')).join('|  ')} ||\n`;
    text += `Aksharas:  |  ${activeTala.beats.map(b => String(b.beat).padEnd(5, ' ')).join('')} ||\n`;
    text += `Kriyas:    |  ${activeTala.beats.map(b => (b.symbol || '·').padEnd(5, ' ')).join('')} ||\n`;
    text += `------------------------------------------------------------------\n`;

    currentTalaData.avartanams.forEach(av => {
      let cellOffset = 0;
      const angaStrs = activeTala.angas.map(a => {
        const slice = av.cells.slice(cellOffset, cellOffset + a.count);
        cellOffset += a.count;
        return slice.map(c => {
          let d = c.display || ',';
          if (c.isDoubleOverline) d = `[[${d}]]`;
          else if (c.isOverline) d = `[${d}]`;
          return d.padEnd(5, ' ');
        }).join('');
      });
      text += `Av. ${String(av.avartanamIndex).padStart(2, ' ')}:    |  ${angaStrs.join('|  ')} ||\n`;
    });
    text += `\nLegend: , = 1 akshara hold; ; = 2 akshara hold; [S R] = 2nd speed; [[S R G M]] = 3rd speed; · = rest\n\n`;
  }

  if (state.freeFlowData && state.freeFlowData.length > 0) {
    text += `FREE FLOW PHRASES (ALAPANA / PRAYOGAS):\n`;
    state.freeFlowData.forEach(p => {
      text += `Phrase ${p.phraseNumber} [${p.startTime.toFixed(2)}s - ${p.endTime.toFixed(2)}s]: ${p.sargamText}\n`;
    });
    text += `\n`;
  }

  text += `CARNATIC SARGAM PHRASE SEQUENCE:\n`;
  text += `${state.swaraNotes.map(n => formatCarnaticSwara(n, raga).symbol).join('  —  ')}\n\n`;

  text += `DETAILED SWARASTHANA & PITCH AUDIT:\n`;
  text += `Time (s)\tSwara\tSwarasthana Name\tSthayi\tMean Pitch (Hz)\tCents Dev\tWestern\tDuration (s)\n`;

  for (const n of state.swaraNotes) {
    const cFmt = formatCarnaticSwara(n, raga);
    const dev = n.centsDeviation >= 0 ? `+${n.centsDeviation}¢` : `${n.centsDeviation}¢`;
    const western = n.western ? n.western.noteStr : '';
    text += `${n.startTime.toFixed(2)}\t${cFmt.symbol}\t${cFmt.fullName}\t${cFmt.sthayi} Sthayi\t${n.meanHz}\t${dev}\t${western}\t${n.duration.toFixed(2)}\n`;
  }

  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  downloadBlob(blob, 'notescope_carnatic_notation.txt');
  showToast('Carnatic manuscript notation downloaded');
}

function exportPitchCsv() {
  if (state.pitches.length === 0) {
    showToast('No pitch data to export', true);
    return;
  }

  let csv = `Time_Seconds,Pitch_Hz,Detected_Sa_Hz,Cents_From_Sa,Carnatic_Swara,Swarasthana_Name,Sthayi\n`;
  const tonic = state.tonic.tonicFreq;
  const raga = state.raga;

  for (let i = 0; i < state.pitches.length; i++) {
    const t = state.timeSteps[i].toFixed(3);
    const p = state.pitches[i];
    if (p > 65) {
      const cents = (1200 * Math.log2(p / tonic)).toFixed(1);
      const swara = state.swaraNotes.find(n => state.timeSteps[i] >= n.startTime && state.timeSteps[i] <= n.endTime);
      const cFmt = swara ? formatCarnaticSwara(swara, raga) : null;
      const swaraSymbol = cFmt ? cFmt.symbol : '';
      const swaraFullName = cFmt ? cFmt.fullName : '';
      const sthayi = cFmt ? `${cFmt.sthayi} Sthayi` : '';
      csv += `${t},${p.toFixed(2)},${tonic},${cents},${swaraSymbol},${swaraFullName},${sthayi}\n`;
    }
  }

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  downloadBlob(blob, 'notescope_pitch_data.csv');
  showToast('Vocal pitch CSV downloaded');
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Reset ───────────────────────────────────────────────────────────────────
function resetApp() {
  stopPlaybackSyncLoop();
  if (state.wavesurfer) {
    state.wavesurfer.destroy();
    state.wavesurfer = null;
  }
  if (state.recorder) {
    state.recorder.destroy();
    state.recorder = null;
  }
  tanpura.stop();
  state.audioBlob = null;
  state.audioBuffer = null;
  state.timeSteps = [];
  state.pitches = [];
  state.swaraNotes = [];
  state.tonic = null;
  if (state.topHandStage) {
    state.topHandStage.stopLoop();
    state.topHandStage = null;
  }
  state.isPlaying = false;
  state.activeNoteIndex = -1;
  state.handStage = null;
  state.luminousWaveform = null;
  state.talaHandAnimator = null;
  state.adiTalaData = null;
  state.freeFlowData = null;
  state.beatAnalysis = null;
  state.carnaticMode = 'tala';
  state.liveRunSwaras = [];

  createApp();
  initTheme();
}

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message, isError = false) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = message;
  toast.className = `toast visible${isError ? ' error' : ''}`;
  setTimeout(() => toast.classList.remove('visible'), 3500);
}

// ── Start App ───────────────────────────────────────────────────────────────
createApp();
initTheme();
