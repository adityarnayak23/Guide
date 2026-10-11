/**
 * NoteScope — Editorial Classical Tala Hand Component (Direction B)
 * 
 * Aesthetic: Darbar / ECM luxury classical archive with artisanal fine-line
 * anatomical illustration and antique gold kinetic motion strokes.
 * 
 * Traditional Carnatic Tala Kriyas (Adi Tala 4 + 2 + 2):
 * - Beat 1: Thattu (தட்டு · Palm Strike on Lap)
 * - Beat 2: Chinna Viral (சின்ன விரல் · Little Finger Count)
 * - Beat 3: Mothira Viral (மோதிர விரல் · Ring Finger Count)
 * - Beat 4: Nadu Viral (நடு விரல் · Middle Finger Count)
 * - Beat 5: Thattu (தட்டு · Palm Strike)
 * - Beat 6: Veechu (வீச்சு · Wave, Palm Turned Upward)
 * - Beat 7: Thattu (தட்டு · Palm Strike)
 * - Beat 8: Veechu (வீச்சு · Wave, Palm Turned Upward)
 */

import { ADI_TALA_KRIYAS } from './carnaticNotation.js';

export class TalaHandAnimator {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.currentBeat = 1;
    this.activeKriya = ADI_TALA_KRIYAS[0];
    this.render();
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="classical-tala-widget" id="classical-tala-widget">
        <!-- Top Editorial Header -->
        <div class="tala-widget-header">
          <div class="tala-heading-group">
            <span class="tala-meta-label">TALA CHOREOGRAPHY</span>
            <h4 class="tala-classical-title">Adi Tala — Chaturasra Nadai</h4>
          </div>

          <!-- Kriya Announcement -->
          <div class="tala-kriya-display" id="tala-kriya-display">
            <span class="kriya-script-tamil" id="kriya-tamil">தட்டு</span>
            <span class="kriya-divider">—</span>
            <span class="kriya-name-english" id="kriya-english">Thattu · Palm Strike</span>
          </div>

          <!-- Anga & Bead Progress Meter -->
          <div class="tala-meter-group">
            <div class="tala-anga-caption" id="tala-anga-badge">Laghu (Beat 1 / 4)</div>
            <div class="tala-bead-meter" id="tala-bead-meter">
              <span class="tala-bead active" data-step="1" title="Beat 1: Thattu"></span>
              <span class="tala-bead" data-step="2" title="Beat 2: Chinna Viral"></span>
              <span class="tala-bead" data-step="3" title="Beat 3: Mothira Viral"></span>
              <span class="tala-bead" data-step="4" title="Beat 4: Nadu Viral"></span>
              <span class="bead-sep">|</span>
              <span class="tala-bead" data-step="5" title="Beat 5: Thattu"></span>
              <span class="tala-bead" data-step="6" title="Beat 6: Veechu"></span>
              <span class="bead-sep">|</span>
              <span class="tala-bead" data-step="7" title="Beat 7: Thattu"></span>
              <span class="tala-bead" data-step="8" title="Beat 8: Veechu"></span>
              <span class="bead-sep">||</span>
            </div>
          </div>
        </div>

        <!-- Center Classical Fine-Line Illustration Canvas -->
        <div class="tala-illustration-stage">
          <svg id="carnatic-hand-svg" viewBox="0 0 320 180" class="classical-hand-svg thattu-state">
            <defs>
              <filter id="goldCandleGlow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="3.5" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            <!-- Subtle Lap Horizon Line -->
            <line x1="40" y1="148" x2="280" y2="148" stroke="rgba(212, 175, 55, 0.18)" stroke-width="1" stroke-dasharray="4 6" />

            <!-- Kinetic Gesture Arcs (Active per kriya state) -->
            <g class="kinetic-motion-group" id="kinetic-motion-group">
              <!-- Strike (Thattu) impact ripple -->
              <g class="motion-strike">
                <path class="kriya-arc strike-arc-1" d="M 60 148 Q 110 164 165 148" fill="none" stroke="#D4AF37" stroke-width="1.6" />
                <path class="kriya-arc strike-arc-2" d="M 78 148 Q 110 158 148 148" fill="none" stroke="#D4AF37" stroke-width="1" opacity="0.6" stroke-dasharray="2 3" />
                <circle class="strike-touch-point" cx="110" cy="154" r="3" fill="#D4AF37" filter="url(#goldCandleGlow)" />
              </g>

              <!-- Finger Count Arcs -->
              <g class="motion-pinky">
                <path class="kriya-arc" d="M 62 128 Q 78 142 96 130" fill="none" stroke="#D4AF37" stroke-width="1.5" stroke-dasharray="2 3" />
                <circle cx="80" cy="138" r="2.5" fill="#D4AF37" filter="url(#goldCandleGlow)" />
              </g>

              <g class="motion-ring">
                <path class="kriya-arc" d="M 54 114 Q 72 128 92 116" fill="none" stroke="#D4AF37" stroke-width="1.5" stroke-dasharray="2 3" />
                <circle cx="74" cy="124" r="2.5" fill="#D4AF37" filter="url(#goldCandleGlow)" />
              </g>

              <g class="motion-middle">
                <path class="kriya-arc" d="M 48 98 Q 68 114 88 100" fill="none" stroke="#D4AF37" stroke-width="1.5" stroke-dasharray="2 3" />
                <circle cx="68" cy="110" r="2.5" fill="#D4AF37" filter="url(#goldCandleGlow)" />
              </g>

              <!-- Wave (Veechu) upward flow -->
              <g class="motion-veechu">
                <path class="kriya-arc" d="M 45 125 Q 95 55 160 82" fill="none" stroke="#D4AF37" stroke-width="1.6" stroke-dasharray="4 4" />
                <path class="kriya-arc" d="M 60 110 Q 100 48 150 72" fill="none" stroke="#D4AF37" stroke-width="1.1" stroke-dasharray="2 3" opacity="0.6" />
                <circle cx="108" cy="56" r="3" fill="#D4AF37" filter="url(#goldCandleGlow)" />
              </g>
            </g>

            <!-- Classical Fine-Line Anatomical Hand -->
            <g id="classical-hand-group" class="classical-hand-group">
              <!-- Forearm and Wrist Lines -->
              <path class="hand-contour" d="M 245 138 C 220 134, 198 128, 182 118" fill="none" stroke="#EAE4DC" stroke-width="1.6" stroke-linecap="round" />
              <path class="hand-contour" d="M 252 110 C 228 104, 206 98, 190 88" fill="none" stroke="#EAE4DC" stroke-width="1.6" stroke-linecap="round" />

              <!-- Antique Gold Bangle (Kankana) on Wrist -->
              <ellipse class="gold-bangle bangle-1" cx="212" cy="112" rx="4.5" ry="17" fill="none" stroke="#D4AF37" stroke-width="2.2" transform="rotate(-18 212 112)" filter="url(#goldCandleGlow)" />
              <ellipse class="gold-bangle bangle-2" cx="221" cy="115" rx="4.5" ry="17" fill="none" stroke="#D4AF37" stroke-width="1.6" transform="rotate(-18 221 115)" />
              <ellipse class="gold-bangle bangle-3" cx="230" cy="118" rx="4.5" ry="17" fill="none" stroke="#C2932E" stroke-width="1.2" transform="rotate(-18 230 118)" stroke-dasharray="3 2" />

              <!-- Palm Base & Thumb (Angushtam) -->
              <path id="hand-thumb" class="hand-contour" d="M 182 118 C 168 112, 154 114, 142 122 C 128 132, 114 140, 98 134 C 88 130, 90 118, 105 114 C 122 110, 138 102, 148 90" fill="none" stroke="#EAE4DC" stroke-width="1.6" stroke-linecap="round" />

              <!-- Index Finger (Tarjani) -->
              <path id="finger-index" class="hand-contour" d="M 148 90 C 134 80, 116 74, 96 72 C 78 70, 68 74, 62 80 C 58 84, 64 88, 76 88 C 96 88, 118 92, 134 100" fill="none" stroke="#EAE4DC" stroke-width="1.5" stroke-linecap="round" />

              <!-- Middle Finger (Nadu Viral / Madhyama) -->
              <path id="finger-middle" class="hand-contour" d="M 134 100 C 120 93, 102 88, 84 86 C 66 84, 56 88, 52 94 C 48 98, 56 102, 68 102 C 88 102, 110 106, 126 112" fill="none" stroke="#EAE4DC" stroke-width="1.5" stroke-linecap="round" />

              <!-- Ring Finger (Mothira Viral / Anamika) with Gold Ring (Mothiram) -->
              <path id="finger-ring" class="hand-contour" d="M 126 112 C 114 106, 98 102, 82 102 C 68 102, 60 106, 58 112 C 55 116, 62 120, 74 120 C 92 120, 110 122, 122 126" fill="none" stroke="#EAE4DC" stroke-width="1.5" stroke-linecap="round" />
              <ellipse class="gold-ring ring-1" cx="94" cy="111" rx="2.8" ry="6" fill="none" stroke="#D4AF37" stroke-width="1.8" filter="url(#goldCandleGlow)" />
              <circle cx="94" cy="111" r="1.6" fill="#C41E3A" />

              <!-- Little Finger (Chinna Viral / Kanishthika) -->
              <path id="finger-pinky" class="hand-contour" d="M 122 126 C 112 122, 100 118, 88 118 C 76 118, 70 122, 68 128 C 66 132, 74 134, 84 134 C 98 134, 115 132, 138 124" fill="none" stroke="#EAE4DC" stroke-width="1.5" stroke-linecap="round" />

              <!-- Traditional Red Alta (Maruthani) Dipped Fingertip Accents -->
              <circle cx="98" cy="134" r="5" fill="rgba(185, 22, 38, 0.75)" filter="url(#goldCandleGlow)" />
              <circle cx="64" cy="81" r="4.5" fill="rgba(185, 22, 38, 0.75)" />
              <circle cx="54" cy="95" r="4.5" fill="rgba(185, 22, 38, 0.75)" />
              <circle cx="60" cy="113" r="4.5" fill="rgba(185, 22, 38, 0.75)" />
              <circle cx="70" cy="129" r="4.2" fill="rgba(185, 22, 38, 0.75)" />

              <!-- Subtle Life Line Contours -->
              <path d="M 156 108 C 148 114, 140 118, 134 120" fill="none" stroke="rgba(212, 175, 55, 0.35)" stroke-width="0.9" />
            </g>
          </svg>
        </div>

        <!-- Bottom Action Subtext -->
        <div class="tala-action-subtext" id="kriya-action-desc">
          Palm downward strike on lap surface
        </div>
      </div>
    `;

    this.updateBeat(1);
  }

  updateBeat(beatNumber) {
    this.currentBeat = Math.max(1, Math.min(8, beatNumber));
    const kriya = ADI_TALA_KRIYAS[this.currentBeat - 1];
    this.activeKriya = kriya;

    const svg = document.getElementById('carnatic-hand-svg');
    const angaBadge = document.getElementById('tala-anga-badge');
    const kriyaTamil = document.getElementById('kriya-tamil');
    const kriyaEnglish = document.getElementById('kriya-english');
    const kriyaDesc = document.getElementById('kriya-action-desc');

    if (!svg) return;

    // Reset state classes
    svg.classList.remove('thattu-state', 'pinky-state', 'ring-state', 'middle-state', 'veechu-state');
    svg.classList.add(`${kriya.action}-state`);

    // Update Text
    if (angaBadge) angaBadge.textContent = `${kriya.anga} · Beat ${kriya.angaBeat} / ${kriya.anga === 'Laghu' ? '4' : '2'} (Akshara ${kriya.akshara} of 8)`;
    if (kriyaTamil) kriyaTamil.textContent = kriya.tamil;
    if (kriyaEnglish) kriyaEnglish.textContent = `${kriya.label} · ${kriya.description}`;
    if (kriyaDesc) kriyaDesc.textContent = kriya.description;

    // Step dots / beads
    document.querySelectorAll('.tala-bead').forEach(bead => {
      const step = parseInt(bead.dataset.step, 10);
      if (step === this.currentBeat) {
        bead.classList.add('active');
      } else {
        bead.classList.remove('active');
      }
    });
  }

  syncWithTime(currentTime, aksharaSec = 0.53) {
    if (!aksharaSec || aksharaSec <= 0) return;
    const totalAksharas = Math.floor(currentTime / aksharaSec);
    const beatInAvartanam = (totalAksharas % 8) + 1;
    if (beatInAvartanam !== this.currentBeat) {
      this.updateBeat(beatInAvartanam);
    }
  }
}
