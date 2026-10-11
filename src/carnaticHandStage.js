/**
 * NoteScope — Classical Tala Hand Rig & GSAP Choreography Engine
 * 
 * Authentic Carnatic Hastha Mudra Choreography:
 * - Clean anatomical pencil/ink sketch aesthetics (parchment cream in dark mode, rich kumkum/madder ink in light mode)
 * - 5 Distinct Classical Vector Sketch Poses:
 *   1. Thattu (Beats 1, 5, 7): Downward flat palm strike (Pataka Mudra) with kinetic shockwave
 *   2. Chinna Viral (Beat 2): Thumb meets little finger (Kanishthika Mudra) with contact spark
 *   3. Mothira Viral (Beat 3): Thumb meets ring finger (Anamika Mudra) with contact spark
 *   4. Nadu Viral (Beat 4): Thumb meets middle finger (Madhyama / Sandamsha Mudra) with contact spark
 *   5. Veechu (Beats 6, 8): Palm waves skyward (Ardhachandra / Uthana Mudra) with floating ribbons
 * - Dual Ornate Gold Wrist Bangles (Valaiyal / Kangan) with metallic luster
 * - Traditional Red Alta (Maruthani / Sempanchu) accents on fingertip pads and palm
 * - Self-contained Web Audio Talam click synthesizer
 * - Autonomous rhythmic playback loop (Vilamba 56, Madhyama 76, Dhruta 108 BPM) & external audio sync
 */

import { gsap } from 'gsap';
import { CARNATIC_TALAS } from './carnaticNotation.js';

export const ADI_TALA_BEATS = CARNATIC_TALAS.adi.beats;

export class CarnaticHandStage {
  constructor(containerId, options = {}) {
    this.container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!this.container) return;

    this.options = {
      isTopStage: false,
      soundEnabled: options.soundEnabled !== undefined ? options.soundEnabled : true,
      bpm: options.bpm || 76,
      onBeatChange: options.onBeatChange || null,
      ...options,
    };

    this.talaId = this.options.talaId || 'adi';
    this.talaDef = CARNATIC_TALAS[this.talaId] || CARNATIC_TALAS.adi;
    this.beats = this.talaDef.beats;
    this.totalBeats = this.talaDef.totalBeats;

    this.currentBeat = 1;
    this.currentKriya = 'thattu';
    this.isPlaying = false;
    this.timerId = null;
    this.audioCtx = null;
    this.lastAksharaTotal = -1;

    this.initSvg();
    this.initElements();
    this.setBeat(1, false);
  }

  resetSync() {
    this.lastAksharaTotal = -1;
  }

  setTala(talaId) {
    if (CARNATIC_TALAS[talaId]) {
      this.talaId = talaId;
      this.talaDef = CARNATIC_TALAS[talaId];
      this.beats = this.talaDef.beats;
      this.totalBeats = this.talaDef.totalBeats;
      this.lastAksharaTotal = -1;
      if (this.currentBeat > this.totalBeats) {
        this.currentBeat = 1;
      }
      this.setBeat(this.currentBeat, false);
      if (this.isPlaying) {
        this.startLoop(this.options.bpm);
      }
    }
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

  initSvg() {
    this.container.innerHTML = `
      <div class="carnatic-stage-wrapper ${this.options.isTopStage ? 'top-showcase-stage' : ''}">
        <svg class="carnatic-hand-svg" viewBox="0 0 1200 896" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Classical Carnatic Tala Hand Sketch Rig">
          <defs>
            <!-- Sacred Stage Vignette Mask -->
            <radialGradient id="stageVignetteGrad" cx="50%" cy="50%" r="50%">
              <stop offset="65%" stop-color="#ffffff" stop-opacity="1" />
              <stop offset="92%" stop-color="#ffffff" stop-opacity="0.85" />
              <stop offset="100%" stop-color="#ffffff" stop-opacity="0" />
            </radialGradient>
            <mask id="stageVignetteMask">
              <rect x="0" y="0" width="1200" height="896" rx="28" fill="url(#stageVignetteGrad)" />
            </mask>

            <!-- Glowing VFX Gradients & Filters -->
            <linearGradient id="goldWaveGrad" x1="0%" y1="0%" x2="100%" y2="50%">
              <stop offset="0%" stop-color="rgba(212, 175, 55, 0.95)" />
              <stop offset="50%" stop-color="rgba(180, 20, 36, 0.65)" />
              <stop offset="100%" stop-color="rgba(180, 20, 36, 0.0)" />
            </linearGradient>

            <!-- 22k Temple Gold Gradients -->
            <linearGradient id="templeGoldShine" x1="0%" y1="0%" x2="100%" y2="60%">
              <stop offset="0%" stop-color="#FFF8D6" />
              <stop offset="25%" stop-color="#F5D061" />
              <stop offset="50%" stop-color="#D4AF37" />
              <stop offset="75%" stop-color="#A67C1E" />
              <stop offset="100%" stop-color="#FFEBB0" />
            </linearGradient>

            <linearGradient id="templeGoldBevel" x1="0%" y1="100%" x2="0%" y2="0%">
              <stop offset="0%" stop-color="#6E4805" />
              <stop offset="30%" stop-color="#B88A22" />
              <stop offset="65%" stop-color="#F2D572" />
              <stop offset="100%" stop-color="#FFF9DE" />
            </linearGradient>

            <linearGradient id="kanganFiligree" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#C2932E" />
              <stop offset="50%" stop-color="#FFEAA8" />
              <stop offset="100%" stop-color="#996E14" />
            </linearGradient>

            <!-- Ruby & Emerald Gemstones -->
            <radialGradient id="gemRuby" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stop-color="#FF6B7A" />
              <stop offset="40%" stop-color="#C41E3A" />
              <stop offset="85%" stop-color="#7A0018" />
              <stop offset="100%" stop-color="#40000B" />
            </radialGradient>

            <radialGradient id="gemEmerald" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stop-color="#80FFAE" />
              <stop offset="40%" stop-color="#097942" />
              <stop offset="85%" stop-color="#024222" />
              <stop offset="100%" stop-color="#012412" />
            </radialGradient>

            <filter id="goldGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3.8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <filter id="sparkleGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4.8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            
            <filter id="thattuGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="5.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <!-- Ambient Sacred Stage Aura -->
          <ellipse cx="600" cy="450" rx="520" ry="340" class="stage-ambient-aura" />

          <!-- Poses Container Group (Animated with GSAP and Softly Vignetted) -->
          <g id="hand-poses-container" mask="url(#stageVignetteMask)">
            <!-- 1. POSE: THATTU (Downward Flat Palm Strike · Pataka Mudra) -->
            <g id="pose-thattu" class="hand-pose-sketch" opacity="1">
              <image class="mudra-art" href="/assets/mudras/light_thattu.webp" x="0" y="0" width="1200" height="896" preserveAspectRatio="xMidYMid meet" />
            </g>

            <!-- 2. POSE: CHINNA VIRAL (Little Finger Tap · Kanishthika Mudra) -->
            <g id="pose-chinna" class="hand-pose-sketch" opacity="0">
              <image class="mudra-art" href="/assets/mudras/light_chinna.webp" x="0" y="0" width="1200" height="896" preserveAspectRatio="xMidYMid meet" />
            </g>

            <!-- 3. POSE: MOTHIRA VIRAL (Ring Finger Tap · Anamika Mudra) -->
            <g id="pose-mothira" class="hand-pose-sketch" opacity="0">
              <image class="mudra-art" href="/assets/mudras/light_mothira.webp" x="0" y="0" width="1200" height="896" preserveAspectRatio="xMidYMid meet" />
            </g>

            <!-- 4. POSE: NADU VIRAL (Middle Finger Tap · Madhyama / Sandamsha Mudra) -->
            <g id="pose-nadu" class="hand-pose-sketch" opacity="0">
              <image class="mudra-art" href="/assets/mudras/light_nadu.webp" x="0" y="0" width="1200" height="896" preserveAspectRatio="xMidYMid meet" />
            </g>

            <!-- 5. POSE: VEECHU (Upward Open Palm Wave · Ardhachandra / Uthana Mudra) -->
            <g id="pose-veechu" class="hand-pose-sketch" opacity="0">
              <image class="mudra-art" href="/assets/mudras/light_veechu.webp" x="0" y="0" width="1200" height="896" preserveAspectRatio="xMidYMid meet" />
            </g>
          </g>

          <!-- VFX Group 1: Thattu Downward Impact Shockwave Rings -->
          <g id="vfx-thattu-ripple" opacity="0" filter="url(#thattuGlow)">
            <ellipse cx="680" cy="480" rx="200" ry="48" fill="none" stroke="rgba(180, 20, 36, 0.9)" stroke-width="3.8" />
            <ellipse cx="680" cy="480" rx="310" ry="72" fill="none" stroke="rgba(212, 175, 55, 0.85)" stroke-width="2.8" stroke-dasharray="12, 8" />
            <ellipse cx="680" cy="480" rx="420" ry="98" fill="none" stroke="rgba(180, 20, 36, 0.45)" stroke-width="2.0" />
            <circle cx="640" cy="460" r="5" fill="#B41424" />
            <circle cx="720" cy="470" r="4.5" fill="#D4AF37" />
            <circle cx="590" cy="490" r="4" fill="#B41424" />
            <circle cx="770" cy="500" r="4" fill="#D4AF37" />
          </g>

          <!-- VFX Group 2: Contact Spark Starburst (Finger Counting Kriyas) -->
          <g id="vfx-contact-spark" opacity="0" filter="url(#sparkleGlow)">
            <circle cx="0" cy="0" r="9.5" fill="#D4AF37" />
            <circle cx="0" cy="0" r="20" fill="none" stroke="#B41424" stroke-width="2.6" opacity="0.9" />
            <line x1="-38" y1="0" x2="38" y2="0" stroke="#B41424" stroke-width="3.8" stroke-linecap="round" />
            <line x1="0" y1="-38" x2="0" y2="38" stroke="#B41424" stroke-width="3.8" stroke-linecap="round" />
            <line x1="-26" y1="-26" x2="26" y2="26" stroke="#D4AF37" stroke-width="2.8" stroke-linecap="round" />
            <line x1="-26" y1="26" x2="26" y2="-26" stroke="#D4AF37" stroke-width="2.8" stroke-linecap="round" />
          </g>

          <!-- VFX Group 3: Veechu Flowing Golden Wave Ribbons -->
          <g id="vfx-veechu-ribbons" opacity="0">
            <path d="M 400,260 Q 520,160 660,220 T 840,190 T 1000,230" fill="none" stroke="url(#goldWaveGrad)" stroke-width="5.5" stroke-linecap="round" />
            <path d="M 360,320 Q 480,220 620,280 T 800,260 T 960,290" fill="none" stroke="url(#goldWaveGrad)" stroke-width="3.8" stroke-linecap="round" stroke-dasharray="10, 8" />
            <path d="M 440,200 Q 560,100 700,160 T 880,140" fill="none" stroke="url(#goldWaveGrad)" stroke-width="2.8" stroke-linecap="round" />
          </g>
        </svg>

        <!-- Kriya Badge & Action Description (Embedded in Stage) -->
        <div class="kriya-subtitle-bar">
          <div class="kriya-tag-badge" id="kriya-badge">Akshara 1 · Thattu (தட்டு)</div>
          <div class="kriya-action-desc" id="kriya-desc">Downward Flat Palm Strike on thigh · Chaturasra Laghu Initial Beat</div>
        </div>
      </div>
    `;
  }

  initElements() {
    this.stageWrapper = this.container.querySelector('.carnatic-stage-wrapper');
    this.svg = this.container.querySelector('.carnatic-hand-svg');
    this.posesContainer = this.container.querySelector('#hand-poses-container');

    // 5 Distinct Poses
    this.poses = {
      thattu: this.container.querySelector('#pose-thattu'),
      chinna: this.container.querySelector('#pose-chinna'),
      mothira: this.container.querySelector('#pose-mothira'),
      nadu: this.container.querySelector('#pose-nadu'),
      veechu: this.container.querySelector('#pose-veechu'),
    };

    // VFX Elements
    this.vfxThattuRipple = this.container.querySelector('#vfx-thattu-ripple');
    this.vfxContactSpark = this.container.querySelector('#vfx-contact-spark');
    this.vfxVeechuRibbons = this.container.querySelector('#vfx-veechu-ribbons');

    // Subtitles
    this.kriyaBadge = this.container.querySelector('#kriya-badge');
    this.kriyaDesc = this.container.querySelector('#kriya-desc');
  }

  animateToKriya(beat, animated = true) {
    return this.setBeat(beat, animated);
  }

  /**
   * Smoothly transitions the classical hand rig to the specified Adi Tala beat (1 to 8).
   * @param {number} beat 1 to 8
   * @param {boolean} animated whether to play the dynamic physical bounce and sound
   */
  setBeat(beat, animated = true) {
    if (beat < 1 || beat > this.totalBeats) beat = 1;
    this.currentBeat = beat;
    const beatInfo = this.beats[beat - 1] || this.beats[0];
    const kriya = beatInfo.kriya;
    this.currentKriya = kriya;

    // Update labels
    if (this.kriyaBadge) {
      this.kriyaBadge.textContent = `Akshara ${beat} · ${beatInfo.name} (${beatInfo.tamil})`;
    }
    if (this.kriyaDesc) {
      this.kriyaDesc.textContent = beatInfo.desc;
    }

    // Play subtle wooden talam click
    if (animated) {
      this.playTalaSound(kriya);
    }

    // Cross-fade the clean sketches
    const dur = animated ? 0.24 : 0;

    Object.keys(this.poses).forEach((key) => {
      const poseEl = this.poses[key];
      if (!poseEl) return;
      if (key === kriya) {
        gsap.to(poseEl, {
          opacity: 1,
          duration: dur,
          ease: 'power2.out',
          overwrite: 'auto',
        });
      } else {
        gsap.to(poseEl, {
          opacity: 0,
          duration: dur * 0.7,
          ease: 'power1.in',
          overwrite: 'auto',
        });
      }
    });

    if (!animated) return;

    // Pose-specific anatomical and kinetic animations
    if (kriya === 'thattu') {
      this.animateThattuKinetic(dur);
    } else if (kriya === 'veechu') {
      this.animateVeechuKinetic(dur);
    } else {
      this.animateFingerTapKinetic(dur, beatInfo.sparkPos);
    }

    // Notify external listeners
    if (typeof this.options.onBeatChange === 'function') {
      this.options.onBeatChange(beat, beatInfo);
    }
  }

  animateThattuKinetic(dur) {
    const target = this.posesContainer || this.svg;
    gsap.killTweensOf([target, this.vfxThattuRipple, this.vfxContactSpark, this.vfxVeechuRibbons]);

    gsap.to(target, {
      y: 18,
      rotation: 1.2,
      transformOrigin: '950px 720px',
      duration: dur * 0.35,
      ease: 'power2.in',
      onComplete: () => {
        gsap.to(target, {
          y: 0,
          rotation: 0,
          duration: dur * 0.85,
          ease: 'elastic.out(1.2, 0.45)',
        });
      },
    });

    if (this.vfxThattuRipple) {
      gsap.set(this.vfxThattuRipple, { opacity: 1, scale: 0.65, transformOrigin: '680px 480px' });
      gsap.to(this.vfxThattuRipple, {
        opacity: 0,
        scale: 1.35,
        duration: 0.45,
        ease: 'power2.out',
      });
    }

    if (this.vfxVeechuRibbons) gsap.to(this.vfxVeechuRibbons, { opacity: 0, duration: 0.1 });
    if (this.vfxContactSpark) gsap.to(this.vfxContactSpark, { opacity: 0, duration: 0.05 });
  }

  animateFingerTapKinetic(dur, sparkPos) {
    const target = this.posesContainer || this.svg;
    gsap.killTweensOf([target, this.vfxContactSpark, this.vfxThattuRipple, this.vfxVeechuRibbons]);

    gsap.to(target, {
      y: -9,
      rotation: -0.6,
      transformOrigin: '950px 720px',
      duration: dur * 0.4,
      ease: 'back.out(1.4)',
      onComplete: () => {
        gsap.to(target, {
          y: 0,
          rotation: 0,
          duration: dur * 0.7,
          ease: 'power1.out',
        });
      },
    });

    if (this.vfxContactSpark && sparkPos) {
      gsap.set(this.vfxContactSpark, {
        x: sparkPos.x,
        y: sparkPos.y,
        scale: 0.3,
        opacity: 1,
        transformOrigin: '0px 0px',
      });
      gsap.to(this.vfxContactSpark, {
        scale: 1.45,
        opacity: 0,
        duration: 0.38,
        ease: 'power2.out',
      });
    }

    if (this.vfxThattuRipple) gsap.to(this.vfxThattuRipple, { opacity: 0, duration: 0.1 });
    if (this.vfxVeechuRibbons) gsap.to(this.vfxVeechuRibbons, { opacity: 0, duration: 0.1 });
  }

  animateVeechuKinetic(dur) {
    const target = this.posesContainer || this.svg;
    gsap.killTweensOf([target, this.vfxVeechuRibbons, this.vfxThattuRipple, this.vfxContactSpark]);

    gsap.to(target, {
      y: -24,
      rotation: -2.8,
      transformOrigin: '950px 720px',
      duration: dur * 0.65,
      ease: 'sine.out',
      onComplete: () => {
        gsap.to(target, {
          y: 0,
          rotation: 0,
          duration: dur * 0.85,
          ease: 'sine.inOut',
        });
      },
    });

    if (this.vfxVeechuRibbons) {
      gsap.set(this.vfxVeechuRibbons, { opacity: 0.9, y: 15 });
      gsap.to(this.vfxVeechuRibbons, {
        y: -35,
        opacity: 0,
        duration: 0.65,
        ease: 'power1.out',
      });
    }

    if (this.vfxThattuRipple) gsap.to(this.vfxThattuRipple, { opacity: 0, duration: 0.1 });
    if (this.vfxContactSpark) gsap.to(this.vfxContactSpark, { opacity: 0, duration: 0.05 });
  }

  playTalaSound(kriya) {
    if (!this.options.soundEnabled) return;
    try {
      const ctx = this.getAudioContext();
      if (!ctx) return;
      if (ctx.state === 'suspended') ctx.resume();

      const now = ctx.currentTime;

      if (kriya === 'thattu') {
        // Resonant classical bronze/wood Talam strike
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(480, now);
        osc.frequency.exponentialRampToValueAtTime(120, now + 0.08);

        gain.gain.setValueAtTime(0.25, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.13);

        // Sub thud
        const subOsc = ctx.createOscillator();
        const subGain = ctx.createGain();
        subOsc.type = 'sine';
        subOsc.frequency.setValueAtTime(140, now);
        subOsc.frequency.exponentialRampToValueAtTime(45, now + 0.07);
        subGain.gain.setValueAtTime(0.18, now);
        subGain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
        subOsc.connect(subGain);
        subGain.connect(ctx.destination);
        subOsc.start(now);
        subOsc.stop(now + 0.09);
      } else if (kriya === 'veechu') {
        // Gentle airy whoosh
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(320, now);
        osc.frequency.linearRampToValueAtTime(520, now + 0.1);

        gain.gain.setValueAtTime(0.05, now);
        gain.gain.linearRampToValueAtTime(0.001, now + 0.14);

        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.15);
      } else {
        // Delicate wooden finger tap click (chinna, mothira, nadu)
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(760, now);
        osc.frequency.exponentialRampToValueAtTime(280, now + 0.04);

        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);

        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.07);
      }
    } catch {
      // Audio autoplay policy catch
    }
  }

  /**
   * Starts autonomous rhythmic playback through all 8 Adi Tala aksharas.
   */
  startLoop(bpm) {
    if (bpm) this.options.bpm = bpm;
    this.stopLoop();
    this.isPlaying = true;

    const intervalMs = (60 / this.options.bpm) * 1000;

    // Immediately play next beat
    this.timerId = setInterval(() => {
      const nextBeat = (this.currentBeat % this.totalBeats) + 1;
      this.setBeat(nextBeat, true);
    }, intervalMs);
  }

  stopLoop() {
    this.isPlaying = false;
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  togglePlay() {
    if (this.isPlaying) {
      this.stopLoop();
    } else {
      this.startLoop();
    }
    return this.isPlaying;
  }

  setBpm(bpm) {
    this.options.bpm = bpm;
    if (this.isPlaying) {
      this.startLoop(bpm);
    }
  }

  toggleSound() {
    this.options.soundEnabled = !this.options.soundEnabled;
    return this.options.soundEnabled;
  }

  /**
   * Synchronize hand directly with audio playback time and Akshara beat duration.
   */
  syncWithTime(currentTime, aksharaSec) {
    if (!aksharaSec || aksharaSec <= 0) return;
    const currentAksharaTotal = Math.floor(currentTime / aksharaSec);
    const beatInAvartanam = (currentAksharaTotal % this.totalBeats) + 1;

    if (beatInAvartanam !== this.currentBeat) {
      this.setBeat(beatInAvartanam, true);
    }
  }
}
