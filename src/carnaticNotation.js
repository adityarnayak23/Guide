/**
 * NoteScope — Authentic Carnatic Music Notation & Tala Engine
 * 
 * Implements the standard Carnatic musicological framework (Prof. P. Sambamoorthy / K.V. Narayanaswamy format):
 * 1. 16 Carnatic Swarasthanas mapped across the 12 semitones:
 *    - S (Shadjam)
 *    - R₁ (Shuddha Rishabham), R₂ (Chatushruti Rishabham / G₁ Shuddha Gandharam), R₃ (Shatshruti Rishabham / G₂ Sadharana Gandharam)
 *    - G₂ (Sadharana Gandharam), G₃ (Antara Gandharam)
 *    - M₁ (Shuddha Madhyamam), M₂ (Prati Madhyamam)
 *    - P (Panchamam)
 *    - D₁ (Shuddha Dhaivatham), D₂ (Chatushruti Dhaivatham / N₁ Shuddha Nishadham), D₃ (Shatshruti Dhaivatham / N₂ Kaisiki Nishadham)
 *    - N₂ (Kaisiki Nishadham), N₃ (Kakali Nishadham)
 * 2. Adi Tala Grid (Chaturasra Jati Triputa Tala: Laghu 4 + Dhrutam 2 + Dhrutam 2 = 8 Aksharas)
 * 3. Authentic Karvai & Speed Notation:
 *    - Single Swara = 1 Akshara (Prathama Kaalam / 1st speed)
 *    - Comma (,) = 1 Akshara extension of sustained note
 *    - Semicolon (;) = 2 Aksharas extension of sustained note
 *    - Rest (·) = Silence / Vishranti (unvoiced beat pause)
 *    - Overline = 2nd speed (Dvitiya Kaalam: 2 swaras in 1 akshara)
 *    - Double overline = 3rd speed (Tritiya Kaalam: 4 swaras in 1 akshara)
 * 4. Sthayi (Octave):
 *    - Taara Sthayi dot above (Ṡ, Ṙ₂, Ġ₃...)
 *    - Mandra Sthayi dot below (Ṣ, Ṛ₂, G̣₃...)
 *    - Madhya Sthayi (S, R₂, G₃...)
 * 5. Free-Flow (Alapana / Virutham) Mode: Unmetered musical phrases (Prayogas / Vakyas)
 * 6. Adi Tala Kriyas (Choreographed hand actions):
 *    - 1: Thattu (Palm Clap)
 *    - 2: Chinna Viral (Pinky Finger Tap)
 *    - 3: Mothira Viral (Ring Finger Tap)
 *    - 4: Nadu Viral (Middle Finger Tap)
 *    - 5: Thattu (Palm Clap)
 *    - 6: Veechu (Palm Wave Upwards)
 *    - 7: Thattu (Palm Clap)
 *    - 8: Veechu (Palm Wave Upwards)
 */

export const CARNATIC_SWARASTHANAS = [
  { index: 0,  symbol: 'S',  name: 'Shadjam',               subscript: '',   cents: 0,    ratio: 1.0,    western: '1 / Tonic' },
  { index: 1,  symbol: 'R₁', name: 'Shuddha Rishabham',     subscript: '₁',  cents: 100,  ratio: 16/15,  western: 'b2 / Minor 2nd' },
  { index: 2,  symbol: 'R₂', name: 'Chatushruti Rishabham',  subscript: '₂',  cents: 200,  ratio: 9/8,    western: '2 / Major 2nd', enharmonic: 'G₁' },
  { index: 3,  symbol: 'G₂', name: 'Sadharana Gandharam',    subscript: '₂',  cents: 300,  ratio: 6/5,    western: 'b3 / Minor 3rd', enharmonic: 'R₃' },
  { index: 4,  symbol: 'G₃', name: 'Antara Gandharam',       subscript: '₃',  cents: 400,  ratio: 5/4,    western: '3 / Major 3rd' },
  { index: 5,  symbol: 'M₁', name: 'Shuddha Madhyamam',      subscript: '₁',  cents: 500,  ratio: 4/3,    western: '4 / Perfect 4th' },
  { index: 6,  symbol: 'M₂', name: 'Prati Madhyamam',        subscript: '₂',  cents: 600,  ratio: 45/32,  western: '#4 / Augmented 4th' },
  { index: 7,  symbol: 'P',  name: 'Panchamam',              subscript: '',   cents: 700,  ratio: 3/2,    western: '5 / Perfect 5th' },
  { index: 8,  symbol: 'D₁', name: 'Shuddha Dhaivatham',    subscript: '₁',  cents: 800,  ratio: 8/5,    western: 'b6 / Minor 6th' },
  { index: 9,  symbol: 'D₂', name: 'Chatushruti Dhaivatham', subscript: '₂',  cents: 900,  ratio: 5/3,    western: '6 / Major 6th', enharmonic: 'N₁' },
  { index: 10, symbol: 'N₂', name: 'Kaisiki Nishadham',      subscript: '₂',  cents: 1000, ratio: 9/5,    western: 'b7 / Minor 7th', enharmonic: 'D₃' },
  { index: 11, symbol: 'N₃', name: 'Kakali Nishadham',       subscript: '₃',  cents: 1100, ratio: 15/8,   western: '7 / Major 7th' },
];

export const MELAKARTA_RAGAS = [
  { name: 'Dheerasankarabharanam (29th Melakarta)', swarasthanas: ['S', 'R₂', 'G₃', 'M₁', 'P', 'D₂', 'N₃'], semitones: [0, 2, 4, 5, 7, 9, 11], western: 'Major / Ionian' },
  { name: 'Mayamalavagowla (15th Melakarta)', swarasthanas: ['S', 'R₁', 'G₃', 'M₁', 'P', 'D₁', 'N₃'], semitones: [0, 1, 4, 5, 7, 8, 11], western: 'Double Harmonic' },
  { name: 'Mechakalyani (65th Melakarta)', swarasthanas: ['S', 'R₂', 'G₃', 'M₂', 'P', 'D₂', 'N₃'], semitones: [0, 2, 4, 6, 7, 9, 11], western: 'Lydian' },
  { name: 'Kharaharapriya (22nd Melakarta)', swarasthanas: ['S', 'R₂', 'G₂', 'M₁', 'P', 'D₂', 'N₂'], semitones: [0, 2, 3, 5, 7, 9, 10], western: 'Dorian' },
  { name: 'Hanumatodi (8th Melakarta)', swarasthanas: ['S', 'R₁', 'G₂', 'M₁', 'P', 'D₁', 'N₂'], semitones: [0, 1, 3, 5, 7, 8, 10], western: 'Phrygian' },
  { name: 'Harikambhoji (28th Melakarta)', swarasthanas: ['S', 'R₂', 'G₃', 'M₁', 'P', 'D₂', 'N₂'], semitones: [0, 2, 4, 5, 7, 9, 10], western: 'Mixolydian' },
  { name: 'Natabhairavi (20th Melakarta)', swarasthanas: ['S', 'R₂', 'G₂', 'M₁', 'P', 'D₁', 'N₂'], semitones: [0, 2, 3, 5, 7, 8, 10], western: 'Natural Minor / Aeolian' },
  { name: 'Charukesi (26th Melakarta)', swarasthanas: ['S', 'R₂', 'G₃', 'M₁', 'P', 'D₁', 'N₂'], semitones: [0, 2, 4, 5, 7, 8, 10], western: 'Melodic Minor 5th' },
  { name: 'Shanmukhapriya (56th Melakarta)', swarasthanas: ['S', 'R₂', 'G₂', 'M₂', 'P', 'D₁', 'N₂'], semitones: [0, 2, 3, 6, 7, 8, 10], western: 'Hungarian Minor' },
  { name: 'Simhendramadhyamam (57th Melakarta)', swarasthanas: ['S', 'R₂', 'G₂', 'M₂', 'P', 'D₁', 'N₃'], semitones: [0, 2, 3, 6, 7, 8, 11], western: 'Harmonic Minor #4' },
  { name: 'Chakravakam (16th Melakarta)', swarasthanas: ['S', 'R₁', 'G₃', 'M₁', 'P', 'D₂', 'N₂'], semitones: [0, 1, 4, 5, 7, 9, 10], western: 'Major Phrygian Dominant' },
  { name: 'Kamavardhani / Pantuvarali (51st Melakarta)', swarasthanas: ['S', 'R₁', 'G₃', 'M₂', 'P', 'D₁', 'N₃'], semitones: [0, 1, 4, 6, 7, 8, 11], western: 'Purvi Thaat' },
  { name: 'Mohanam (Janya of Harikambhoji)', swarasthanas: ['S', 'R₂', 'G₃', 'P', 'D₂'], semitones: [0, 2, 4, 7, 9], western: 'Major Pentatonic' },
  { name: 'Hamsadhwani (Janya of Shankarabharanam)', swarasthanas: ['S', 'R₂', 'G₃', 'P', 'N₃'], semitones: [0, 2, 4, 7, 11], western: 'Pentatonic Major (no 4/6)' },
  { name: 'Hindolam (Janya of Natabhairavi)', swarasthanas: ['S', 'G₂', 'M₁', 'D₁', 'N₂'], semitones: [0, 3, 5, 8, 10], western: 'Minor Pentatonic (no 2/5)' },
  { name: 'Shuddha Saveri (Janya of Shankarabharanam)', swarasthanas: ['S', 'R₂', 'M₁', 'P', 'D₂'], semitones: [0, 2, 5, 7, 9], western: 'Ritusen Pentatonic' },
  { name: 'Malahari (Janya of Mayamalavagowla)', swarasthanas: ['S', 'R₁', 'M₁', 'P', 'D₁'], semitones: [0, 1, 5, 7, 8], western: 'Geetham scale' },
  { name: 'Madhyamavati (Janya of Kharaharapriya)', swarasthanas: ['S', 'R₂', 'M₁', 'P', 'N₂'], semitones: [0, 2, 5, 7, 10], western: 'Auspicious Mangalam scale' },
  { name: 'Abhogi (Janya of Kharaharapriya)', swarasthanas: ['S', 'R₂', 'G₂', 'M₁', 'D₂'], semitones: [0, 2, 3, 5, 9], western: 'Pentatonic Dorian' },
  { name: 'Nattai (Janya of Chalanata)', swarasthanas: ['S', 'R₃', 'G₃', 'M₁', 'P', 'N₃'], semitones: [0, 3, 4, 5, 7, 11], western: 'Gambheera scale' },
  { name: 'Revati (Janya of Mayamalavagowla)', swarasthanas: ['S', 'R₁', 'M₁', 'P', 'N₂'], semitones: [0, 1, 5, 7, 10], western: 'Bairagi Bhairav' },
  { name: 'Bilahari (Janya of Shankarabharanam)', swarasthanas: ['S', 'R₂', 'G₃', 'P', 'D₂'], semitones: [0, 2, 4, 7, 9], western: 'Alhaiya Bilawal' },
  { name: 'Suddha Dhanyasi (Janya of Kharaharapriya)', swarasthanas: ['S', 'G₂', 'M₁', 'P', 'N₂'], semitones: [0, 3, 5, 7, 10], western: 'Dhani Thaat' },
];

export const CARNATIC_TALAS = {
  adi: {
    id: 'adi',
    name: 'Adi Tala (Chaturasra Jati Triputa Tala)',
    shortName: 'Adi Tala (8)',
    nadai: 'Chaturasra Nadai',
    sign: 'I₄ O O',
    totalBeats: 8,
    angas: [
      { name: 'Laghu', count: 4, label: 'Laghu (4)' },
      { name: 'Dhrutam', count: 2, label: 'Dhrutam (2)' },
      { name: 'Dhrutam', count: 2, label: 'Dhrutam (2)' },
    ],
    prior: 0.65,
    beats: [
      { beat: 1, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Downward flat palm strike on thigh (Pataka mudra) · Initial strike of Laghu', sparkPos: null },
      { beat: 2, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Little Finger Tap', desc: 'Thumb tip touches little finger tip (Kanishthika mudra)', sparkPos: { x: 609, y: 324 } },
      { beat: 3, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Ring Finger Tap', desc: 'Thumb tip touches ring finger tip (Anamika mudra)', sparkPos: { x: 505, y: 270 } },
      { beat: 4, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'nadu', name: 'Nadu Viral', symbol: '·', tamil: 'நடுவிரல்', action: 'Middle Finger Tap', desc: 'Thumb tip touches middle finger tip (Madhyama mudra) completing Laghu', sparkPos: { x: 560, y: 300 } },
      { beat: 5, anga: 'Dhrutam 1 (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Downward flat palm strike on thigh · Initial strike of 1st Dhrutam', sparkPos: null },
      { beat: 6, anga: 'Dhrutam 1 (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'Hand turns over, palm facing skyward in an upward wave gesture', sparkPos: null },
      { beat: 7, anga: 'Dhrutam 2 (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Downward flat palm strike on thigh · Initial strike of 2nd Dhrutam', sparkPos: null },
      { beat: 8, anga: 'Dhrutam 2 (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'Hand turns over, palm facing skyward · Completing the Avartan', sparkPos: null },
    ],
  },
  rupaka: {
    id: 'rupaka',
    name: 'Rupaka Tala (Chaturasra Jati Rupaka Tala)',
    shortName: 'Rupaka Tala (6)',
    nadai: 'Chaturasra Nadai',
    sign: 'O I₄',
    totalBeats: 6,
    angas: [
      { name: 'Dhrutam', count: 2, label: 'Dhrutam (2)' },
      { name: 'Laghu', count: 4, label: 'Laghu (4)' },
    ],
    prior: 0.16,
    beats: [
      { beat: 1, anga: 'Dhrutam (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Downward flat palm strike on thigh · Dhrutam strike', sparkPos: null },
      { beat: 2, anga: 'Dhrutam (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'Hand turns over, palm facing skyward (Dhrutam wave)', sparkPos: null },
      { beat: 3, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Downward flat palm strike · Beginning of Chaturasra Laghu', sparkPos: null },
      { beat: 4, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Little Finger Tap', desc: 'Thumb tip touches little finger tip (Kanishthika mudra)', sparkPos: { x: 609, y: 324 } },
      { beat: 5, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Ring Finger Tap', desc: 'Thumb tip touches ring finger tip (Anamika mudra)', sparkPos: { x: 505, y: 270 } },
      { beat: 6, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'nadu', name: 'Nadu Viral', symbol: '·', tamil: 'நடுவிரல்', action: 'Middle Finger Tap', desc: 'Thumb tip touches middle finger tip (Madhyama mudra) · Completing Rupaka Avartan', sparkPos: { x: 560, y: 300 } },
    ],
  },
  misra_chapu: {
    id: 'misra_chapu',
    name: 'Misra Chapu Tala',
    shortName: 'Misra Chapu (7)',
    nadai: 'Chapu (3+4)',
    sign: '3 + 4',
    totalBeats: 7,
    angas: [
      { name: 'Tisra Anga', count: 3, label: 'Krama 1 (3)' },
      { name: 'Chaturasra Anga', count: 4, label: 'Krama 2 (4)' },
    ],
    prior: 0.08,
    beats: [
      { beat: 1, anga: 'Krama 1 (3)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Primary downbeat strike of Misra Chapu (Samam)', sparkPos: null },
      { beat: 2, anga: 'Krama 1 (3)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'First finger count of Tisra section', sparkPos: { x: 609, y: 324 } },
      { beat: 3, anga: 'Krama 1 (3)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Second finger count of Tisra section', sparkPos: { x: 505, y: 270 } },
      { beat: 4, anga: 'Krama 2 (4)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Secondary accent strike initiating 4-count section', sparkPos: null },
      { beat: 5, anga: 'Krama 2 (4)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'Counting little finger', sparkPos: { x: 609, y: 324 } },
      { beat: 6, anga: 'Krama 2 (4)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Counting ring finger', sparkPos: { x: 505, y: 270 } },
      { beat: 7, anga: 'Krama 2 (4)', angaType: 'laghu', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'Upward wave resolving the 7-beat cycle', sparkPos: null },
    ],
  },
  khanda_chapu: {
    id: 'khanda_chapu',
    name: 'Khanda Chapu Tala',
    shortName: 'Khanda Chapu (5)',
    nadai: 'Chapu (2+3)',
    sign: '2 + 3',
    totalBeats: 5,
    angas: [
      { name: 'Dhruta Anga', count: 2, label: 'Krama 1 (2)' },
      { name: 'Tisra Anga', count: 3, label: 'Krama 2 (3)' },
    ],
    prior: 0.05,
    beats: [
      { beat: 1, anga: 'Krama 1 (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Primary strike initiating Khanda Chapu', sparkPos: null },
      { beat: 2, anga: 'Krama 1 (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'Wave gesture completing 2-count', sparkPos: null },
      { beat: 3, anga: 'Krama 2 (3)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Strike initiating 3-count section', sparkPos: null },
      { beat: 4, anga: 'Krama 2 (3)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'Little finger count', sparkPos: { x: 609, y: 324 } },
      { beat: 5, anga: 'Krama 2 (3)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Ring finger count resolving 5-beat cycle', sparkPos: { x: 505, y: 270 } },
    ],
  },
  tisra_triputa: {
    id: 'tisra_triputa',
    name: 'Tisra Triputa Tala (Tisra Jati Triputa Tala)',
    shortName: 'Tisra Triputa (7)',
    nadai: 'Tisra Nadai',
    sign: 'I₃ O O',
    totalBeats: 7,
    angas: [
      { name: 'Laghu', count: 3, label: 'Laghu (3)' },
      { name: 'Dhrutam', count: 2, label: 'Dhrutam (2)' },
      { name: 'Dhrutam', count: 2, label: 'Dhrutam (2)' },
    ],
    prior: 0.02,
    beats: [
      { beat: 1, anga: 'Laghu (3)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Strike of Tisra Laghu', sparkPos: null },
      { beat: 2, anga: 'Laghu (3)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'Little finger tap', sparkPos: { x: 609, y: 324 } },
      { beat: 3, anga: 'Laghu (3)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Ring finger tap completing Tisra Laghu', sparkPos: { x: 505, y: 270 } },
      { beat: 4, anga: 'Dhrutam 1 (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'First Dhrutam strike', sparkPos: null },
      { beat: 5, anga: 'Dhrutam 1 (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'First Dhrutam wave', sparkPos: null },
      { beat: 6, anga: 'Dhrutam 2 (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Second Dhrutam strike', sparkPos: null },
      { beat: 7, anga: 'Dhrutam 2 (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'Second Dhrutam wave completing 7-beat cycle', sparkPos: null },
    ],
  },
  misra_jhampa: {
    id: 'misra_jhampa',
    name: 'Misra Jhampa Tala (Misra Jati Jhampa Tala)',
    shortName: 'Misra Jhampa (10)',
    nadai: 'Misra Nadai',
    sign: 'I₇ U O',
    totalBeats: 10,
    angas: [
      { name: 'Laghu', count: 7, label: 'Laghu (7)' },
      { name: 'Anudhrutam', count: 1, label: 'Anudhrutam (1)' },
      { name: 'Dhrutam', count: 2, label: 'Dhrutam (2)' },
    ],
    prior: 0.02,
    beats: [
      { beat: 1, anga: 'Laghu (7)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Initial strike of Misra Laghu', sparkPos: null },
      { beat: 2, anga: 'Laghu (7)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'Little finger tap', sparkPos: { x: 609, y: 324 } },
      { beat: 3, anga: 'Laghu (7)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Ring finger tap', sparkPos: { x: 505, y: 270 } },
      { beat: 4, anga: 'Laghu (7)', angaType: 'laghu', kriya: 'nadu', name: 'Nadu Viral', symbol: '·', tamil: 'நடுவிரல்', action: 'Finger Tap', desc: 'Middle finger tap', sparkPos: { x: 560, y: 300 } },
      { beat: 5, anga: 'Laghu (7)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: '5th count strike', sparkPos: null },
      { beat: 6, anga: 'Laghu (7)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: '6th count little finger tap', sparkPos: { x: 609, y: 324 } },
      { beat: 7, anga: 'Laghu (7)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: '7th count completing Misra Laghu', sparkPos: { x: 505, y: 270 } },
      { beat: 8, anga: 'Anudhrutam (1)', angaType: 'anudhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Single Anudhrutam strike', sparkPos: null },
      { beat: 9, anga: 'Dhrutam (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Dhrutam strike', sparkPos: null },
      { beat: 10, anga: 'Dhrutam (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: 'Dhrutam wave completing 10-beat cycle', sparkPos: null },
    ],
  },
  khanda_ata: {
    id: 'khanda_ata',
    name: 'Khanda Ata Tala (Khanda Jati Ata Tala)',
    shortName: 'Khanda Ata (14)',
    nadai: 'Khanda Nadai',
    sign: 'I₅ I₅ O O',
    totalBeats: 14,
    angas: [
      { name: 'Laghu 1', count: 5, label: 'Laghu 1 (5)' },
      { name: 'Laghu 2', count: 5, label: 'Laghu 2 (5)' },
      { name: 'Dhrutam 1', count: 2, label: 'Dhrutam 1 (2)' },
      { name: 'Dhrutam 2', count: 2, label: 'Dhrutam 2 (2)' },
    ],
    prior: 0.015,
    beats: [
      { beat: 1, anga: 'Laghu 1 (5)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: '1st Laghu strike', sparkPos: null },
      { beat: 2, anga: 'Laghu 1 (5)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'Little finger tap', sparkPos: { x: 609, y: 324 } },
      { beat: 3, anga: 'Laghu 1 (5)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Ring finger tap', sparkPos: { x: 505, y: 270 } },
      { beat: 4, anga: 'Laghu 1 (5)', angaType: 'laghu', kriya: 'nadu', name: 'Nadu Viral', symbol: '·', tamil: 'நடுவிரல்', action: 'Finger Tap', desc: 'Middle finger tap', sparkPos: { x: 560, y: 300 } },
      { beat: 5, anga: 'Laghu 1 (5)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: '5th count of 1st Laghu', sparkPos: { x: 609, y: 324 } },
      { beat: 6, anga: 'Laghu 2 (5)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: '2nd Laghu strike', sparkPos: null },
      { beat: 7, anga: 'Laghu 2 (5)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'Little finger tap', sparkPos: { x: 609, y: 324 } },
      { beat: 8, anga: 'Laghu 2 (5)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Ring finger tap', sparkPos: { x: 505, y: 270 } },
      { beat: 9, anga: 'Laghu 2 (5)', angaType: 'laghu', kriya: 'nadu', name: 'Nadu Viral', symbol: '·', tamil: 'நடுவிரல்', action: 'Finger Tap', desc: 'Middle finger tap', sparkPos: { x: 560, y: 300 } },
      { beat: 10, anga: 'Laghu 2 (5)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: '10th count completing 2nd Laghu', sparkPos: { x: 609, y: 324 } },
      { beat: 11, anga: 'Dhrutam 1 (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: '1st Dhrutam strike', sparkPos: null },
      { beat: 12, anga: 'Dhrutam 1 (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: '1st Dhrutam wave', sparkPos: null },
      { beat: 13, anga: 'Dhrutam 2 (2)', angaType: 'dhrutam', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: '2nd Dhrutam strike', sparkPos: null },
      { beat: 14, anga: 'Dhrutam 2 (2)', angaType: 'dhrutam', kriya: 'veechu', name: 'Veechu', symbol: '~', tamil: 'வீச்சு', action: 'Wave Upward', desc: '2nd Dhrutam wave completing 14-beat cycle', sparkPos: null },
    ],
  },
  eka: {
    id: 'eka',
    name: 'Chaturasra Eka Tala',
    shortName: 'Eka Tala (4)',
    nadai: 'Chaturasra Nadai',
    sign: 'I₄',
    totalBeats: 4,
    angas: [
      { name: 'Laghu', count: 4, label: 'Laghu (4)' },
    ],
    prior: 0.005,
    beats: [
      { beat: 1, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'thattu', name: 'Thattu', symbol: '▾', tamil: 'தட்டு', action: 'Palm Strike', desc: 'Laghu strike', sparkPos: null },
      { beat: 2, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'chinna', name: 'Chinna Viral', symbol: '·', tamil: 'சிறுவிரல்', action: 'Finger Tap', desc: 'Little finger tap', sparkPos: { x: 609, y: 324 } },
      { beat: 3, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'mothira', name: 'Mothira Viral', symbol: '·', tamil: 'மோதிரவிரல்', action: 'Finger Tap', desc: 'Ring finger tap', sparkPos: { x: 505, y: 270 } },
      { beat: 4, anga: 'Laghu (4)', angaType: 'laghu', kriya: 'nadu', name: 'Nadu Viral', symbol: '·', tamil: 'நடுவிரல்', action: 'Finger Tap', desc: 'Middle finger tap completing 4-beat cycle', sparkPos: { x: 560, y: 300 } },
    ],
  },
};

export const ADI_TALA_KRIYAS = CARNATIC_TALAS.adi.beats.map(b => ({
  akshara: b.beat,
  anga: b.anga,
  angaBeat: b.beat <= 4 ? b.beat : (b.beat <= 6 ? b.beat - 4 : b.beat - 6),
  action: b.kriya,
  label: b.name,
  symbol: b.symbol,
  tamil: b.tamil,
  description: b.desc,
}));

export function getSwaraIndex(n) {
  if (!n) return 0;
  if (n.swara && n.swara.index !== undefined) return n.swara.index;
  if (n.swara && n.swara.cents !== undefined) return Math.round(n.swara.cents / 100) % 12;
  const name = n.swaraName || (n.swara ? n.swara.name : '') || n.carnatic || n.formatted || '';
  const cleanName = name.replace(/[\u0307\u0323\u0308]/g, '').replace(/1/g, '₁').replace(/2/g, '₂').replace(/3/g, '₃');
  const found = CARNATIC_SWARASTHANAS.findIndex(s => s.symbol === cleanName);
  if (found !== -1) return found;
  if (name.includes('R2') || name.includes('R₂')) return 2;
  if (name.includes('R1') || name.includes('R₁')) return 1;
  if (name.includes('G2') || name.includes('G₂')) return 3;
  if (name.includes('G3') || name.includes('G₃')) return 4;
  if (name.includes('M1') || name.includes('M₁')) return 5;
  if (name.includes('M2') || name.includes('M₂')) return 6;
  if (name.includes('P')) return 7;
  if (name.includes('D1') || name.includes('D₁')) return 8;
  if (name.includes('D2') || name.includes('D₂')) return 9;
  if (name.includes('N2') || name.includes('N₂')) return 10;
  if (name.includes('N3') || name.includes('N₃')) return 11;
  if (name.includes('S')) return 0;
  return 0;
}

/**
 * Converts Swara object into pure Carnatic letter with Sthayi dot and Swarasthana.
 * e.g., Sa -> S, Re -> R₂, Ni in Taar Saptak -> Ṅ₃, Sa in Taar -> Ṡ
 * Supports raga-aware enharmonic resolution (e.g. R₃ in Nattai, G₁ in Ganamurti).
 */
export function formatCarnaticSwara(note, raga = null) {
  if (!note) return { symbol: '—', rawSymbol: '—', sthayi: 'Madhya', subscript: '', fullName: '—' };
  
  const swIdx = getSwaraIndex(note);
  let swDef = CARNATIC_SWARASTHANAS[swIdx] || CARNATIC_SWARASTHANAS[0];
  const saptak = note.saptak || 'Madhya';

  // Check if active raga uses an enharmonic representation (e.g. R₃ instead of G₂)
  let baseLetter = swDef.symbol.charAt(0);
  let sub = swDef.subscript;
  let fullName = swDef.name;

  if (raga && Array.isArray(raga.swarasthanas)) {
    // Check semitone 3: could be R₃
    if (swIdx === 3 && raga.swarasthanas.includes('R₃')) {
      baseLetter = 'R';
      sub = '₃';
      fullName = 'Shatshruti Rishabham';
    }
    // Check semitone 2: could be G₁
    else if (swIdx === 2 && raga.swarasthanas.includes('G₁')) {
      baseLetter = 'G';
      sub = '₁';
      fullName = 'Shuddha Gandharam';
    }
    // Check semitone 10: could be D₃
    else if (swIdx === 10 && raga.swarasthanas.includes('D₃')) {
      baseLetter = 'D';
      sub = '₃';
      fullName = 'Shatshruti Dhaivatham';
    }
    // Check semitone 9: could be N₁
    else if (swIdx === 9 && raga.swarasthanas.includes('N₁')) {
      baseLetter = 'N';
      sub = '₁';
      fullName = 'Shuddha Nishadham';
    }
  }

  // Sthayi dot placement:
  // Dot MUST be directly on base letter so subscript appears cleanly beside it (e.g. Ṙ₂, G̣₃)
  let sthayiSymbol = baseLetter;
  if (saptak === 'Taar') {
    sthayiSymbol = `${baseLetter}\u0307`; // combining dot above: Ṡ
  } else if (saptak === 'Mandra') {
    sthayiSymbol = `${baseLetter}\u0323`; // combining dot below: Ṣ
  } else if (saptak === 'AtiTaar') {
    sthayiSymbol = `${baseLetter}\u0308`; // diaeresis (two dots above): S̈
  }

  return {
    symbol: `${sthayiSymbol}${sub}`,
    rawLetter: baseLetter,
    sthayiSymbol,
    subscript: sub,
    sthayi: saptak,
    fullName,
    centsDeviation: note.centsDeviation || 0,
  };
}

/**
 * Fast Acoustic Onset & Autocorrelation Beat/Tempo Detector.
 * Estimates probable Carnatic vocal BPM directly from raw AudioBuffer waveforms.
 * Works immediately upon audio load or microphone capture before or in parallel with neural note recognition.
 */
export function detectAudioBpm(audioBuffer) {
  if (!audioBuffer) {
    return { bpm: 76, confidence: 0.5, aksharaSec: 0.79, kala: 'Madhyama Kala' };
  }
  
  const sr = audioBuffer.sampleRate;
  const channelData = audioBuffer.getChannelData(0);
  const duration = audioBuffer.duration;
  
  if (duration < 0.8) {
    return { bpm: 76, confidence: 0.5, aksharaSec: 0.79, kala: 'Madhyama Kala' };
  }

  // Downsample to 100 Hz energy frames (10ms hop, 25ms window)
  const hopSize = Math.max(1, Math.floor(sr * 0.010)); // 10ms
  const winSize = Math.max(1, Math.floor(sr * 0.025)); // 25ms
  const numFrames = Math.floor((channelData.length - winSize) / hopSize);
  
  if (numFrames < 50) {
    return { bpm: 76, confidence: 0.5, aksharaSec: 0.79, kala: 'Madhyama Kala' };
  }

  const envelope = new Float32Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    let sumSq = 0;
    // Step by 4 for high-speed calculation
    for (let j = 0; j < winSize; j += 4) {
      const s = channelData[start + j] || 0;
      sumSq += s * s;
    }
    envelope[i] = Math.sqrt(sumSq / (winSize / 4));
  }

  // Onset novelty curve: half-wave rectified first difference
  const onsetFlux = new Float32Array(numFrames - 1);
  let meanFlux = 0;
  for (let i = 0; i < numFrames - 1; i++) {
    const diff = envelope[i + 1] - envelope[i];
    onsetFlux[i] = diff > 0 ? diff : 0;
    meanFlux += onsetFlux[i];
  }
  meanFlux /= (numFrames - 1);

  // Mean-center the onset novelty flux
  for (let i = 0; i < numFrames - 1; i++) {
    onsetFlux[i] -= meanFlux;
  }

  // Evaluate candidate BPMs from 48 to 150 BPM
  // tau in seconds = 60 / bpm -> lag in frames = 6000 / bpm
  let bestBpm = 76;
  let maxScore = -Infinity;

  for (let bpm = 48; bpm <= 150; bpm++) {
    const lag = Math.round(6000 / bpm);
    if (lag >= onsetFlux.length - 10) continue;

    let sum = 0;
    const maxI = Math.min(onsetFlux.length - lag, 3000); // sample up to 30s
    for (let i = 0; i < maxI; i += 2) {
      sum += onsetFlux[i] * onsetFlux[i + lag];
    }
    const corr = sum / (maxI / 2);

    // Carnatic tempo prior favoring natural 65-120 range
    const prior = Math.exp(-0.5 * Math.pow((bpm - 85) / 38, 2));
    const score = corr * (1.0 + 0.35 * prior);

    if (score > maxScore) {
      maxScore = score;
      bestBpm = bpm;
    }
  }

  const aksharaSec = Number((60 / bestBpm).toFixed(2));
  let kala = 'Madhyama Kala';
  if (bestBpm < 68) kala = 'Vilamba Kala';
  else if (bestBpm > 98) kala = 'Dhruta Kala';

  return {
    bpm: bestBpm,
    aksharaSec,
    kala,
    confidence: Number(Math.min(1.0, Math.max(0.4, (maxScore > 0 ? 0.85 : 0.5))).toFixed(2)),
  };
}

/**
 * Automatically evaluates whether vocal recording has a rhythmic beat structure
 * or is free-flowing (Alapana / Virutham), finding the optimal Carnatic Kala BPM.
 */
export function detectBeatStructure(notes, audioDuration, audioEstimatedBpm = null) {
  if (!notes || notes.length < 4) {
    const fallbackBpm = audioEstimatedBpm || 76;
    const aksharaSec = Number((60 / fallbackBpm).toFixed(2));
    return {
      hasBeatStructure: false,
      estimatedBpm: fallbackBpm,
      aksharaSec,
      coeffVar: 1.0,
      kala: fallbackBpm < 68 ? 'Vilamba Kala' : (fallbackBpm > 98 ? 'Dhruta Kala' : 'Madhyama Kala')
    };
  }

  // 1. Calculate Inter-Onset Intervals (IOIs)
  const iois = [];
  const noteStarts = [];
  for (let i = 0; i < notes.length; i++) {
    noteStarts.push(notes[i].startTime);
    if (i < notes.length - 1) {
      const diff = notes[i + 1].startTime - notes[i].startTime;
      if (diff > 0.15 && diff < 3.0) {
        iois.push(diff);
      }
    }
  }

  // 2. Multi-Candidate Grid Quantization & Subharmonic Optimization
  let bestCandidate = null;
  let minPhaseError = Infinity;

  const testBpms = [];
  for (let b = 48; b <= 145; b++) testBpms.push(b);
  if (audioEstimatedBpm && !testBpms.includes(audioEstimatedBpm)) {
    testBpms.push(audioEstimatedBpm);
  }

  for (const testBpm of testBpms) {
    const akshara = 60 / testBpm;
    let totalPhaseErr = 0;
    for (const t of noteStarts) {
      const phase = (t / akshara) % 1.0;
      const err = Math.min(phase, 1.0 - phase);
      totalPhaseErr += err;
    }
    const meanErr = totalPhaseErr / noteStarts.length;

    // Slight bonus if it matches acoustic audio onset BPM or octave harmonic
    let acousticBonus = 0;
    if (audioEstimatedBpm) {
      const ratio = testBpm / audioEstimatedBpm;
      if (Math.abs(ratio - 1) < 0.08 || Math.abs(ratio - 0.5) < 0.06 || Math.abs(ratio - 2) < 0.12) {
        acousticBonus = 0.035;
      }
    }

    const effectiveScore = meanErr - acousticBonus;
    if (effectiveScore < minPhaseError) {
      minPhaseError = effectiveScore;
      bestCandidate = { bpm: testBpm, aksharaSec: akshara, meanErr };
    }
  }

  // 3. Fallback / validation with median IOI
  const sorted = [...iois].sort((a, b) => a - b);
  const medianIoi = sorted[Math.floor(sorted.length / 2)] || 0.79;
  const meanIoi = iois.reduce((a, b) => a + b, 0) / (iois.length || 1);
  const variance = iois.reduce((acc, val) => acc + Math.pow(val - meanIoi, 2), 0) / (iois.length || 1);
  const stdDev = Math.sqrt(variance);
  const coeffVar = stdDev / (meanIoi || 1);

  let estimatedBpm = bestCandidate ? bestCandidate.bpm : Math.round(60 / medianIoi);
  let aksharaSec = Number((60 / estimatedBpm).toFixed(2));
  aksharaSec = Math.max(0.35, Math.min(1.25, aksharaSec));
  estimatedBpm = Math.round(60 / aksharaSec);

  const isRhythmic = coeffVar < 0.55 || (bestCandidate && bestCandidate.meanErr < 0.18);

  let kala = 'Madhyama Kala';
  if (estimatedBpm < 68) kala = 'Vilamba Kala';
  else if (estimatedBpm > 98) kala = 'Dhruta Kala';

  return {
    hasBeatStructure: isRhythmic,
    estimatedBpm,
    aksharaSec,
    coeffVar: Number(coeffVar.toFixed(2)),
    phaseError: bestCandidate ? Number(bestCandidate.meanErr.toFixed(3)) : 0.5,
    kala,
  };
}

/**
 * Universal Carnatic Tala Quantizer.
 * Maps sung notes into authentic Avartanams for any of the 8 major Carnatic Talas.
 * 
 * Quality Guarantees:
 * 1. Onset-primary assignment: notes belong to the beat where they strike,
 *    eliminating false 2nd-speed (dvitiya) overlines caused by subtle boundary spillover.
 * 2. True 2nd speed overlines: only applied when 2 genuine onsets occur within 1 beat.
 * 3. Authentic Karvai (hold): commas (,) only indicate actual sustained notes from previous beats.
 * 4. Vishranti (silence): empty beats are marked as subtle rests (·), NOT as commas or semicolons.
 * 5. Avartanam pruning: avoids generating empty phantom rows beyond the song's ending.
 */
export function buildTalaGrid(notes, audioDuration, customAksharaSec = null, raga = null, talaId = 'adi') {
  const talaDef = CARNATIC_TALAS[talaId] || CARNATIC_TALAS.adi;
  const totalBeats = talaDef.totalBeats;

  // Determine tempo / akshara duration
  const { aksharaSec: autoAkshara } = detectBeatStructure(notes, audioDuration);
  const aksharaSec = customAksharaSec || autoAkshara || 0.54;

  if (!notes || notes.length === 0) {
    return {
      talaId: talaDef.id,
      talaName: talaDef.name,
      shortName: talaDef.shortName,
      nadai: talaDef.nadai,
      sign: talaDef.sign,
      totalBeats,
      angas: talaDef.angas,
      aksharaSec,
      bpm: Math.round(60 / aksharaSec),
      avartanams: [],
    };
  }

  // Determine musical boundary (do not create trailing avartanams past the music)
  const lastNote = notes[notes.length - 1];
  const lastNoteEndTime = lastNote ? lastNote.endTime : audioDuration;
  const totalAksharasNeeded = Math.max(totalBeats, Math.round(lastNoteEndTime / aksharaSec));
  const numAvartanams = Math.ceil(totalAksharasNeeded / totalBeats);

  const avartanams = [];
  let noteCursor = 0;

  for (let aIdx = 0; aIdx < numAvartanams; aIdx++) {
    const startAkshara = aIdx * totalBeats;
    const aksharaCells = [];

    for (let beat = 1; beat <= totalBeats; beat++) {
      const aksharaNumber = startAkshara + beat;
      const beatStartTime = (aksharaNumber - 1) * aksharaSec;
      const beatEndTime = aksharaNumber * aksharaSec;
      const onsetTolerance = aksharaSec * 0.22; // 22% tolerance for singer rhythm variability

      // Collect note onsets belonging to this beat
      const onsetsInBeat = [];
      while (noteCursor < notes.length) {
        const n = notes[noteCursor];
        if (n.startTime < beatEndTime - onsetTolerance) {
          onsetsInBeat.push(n);
          noteCursor++;
        } else {
          break;
        }
      }

      let cellDisplay = '·';
      let cellType = 'rest';
      let isOverline = false;
      let isDoubleOverline = false;
      let activeNote = null;

      if (onsetsInBeat.length === 1) {
        // Prathama Kaalam (1st speed): 1 note in 1 akshara
        const n = onsetsInBeat[0];
        cellDisplay = formatCarnaticSwara(n, raga).symbol;
        cellType = 'swara';
        activeNote = n;
      } else if (onsetsInBeat.length >= 4) {
        // Tritiya Kaalam (3rd speed): 4 notes in 1 akshara with double overline
        cellDisplay = onsetsInBeat.map(n => formatCarnaticSwara(n, raga).symbol).join(' ');
        cellType = 'tritiya';
        isOverline = true;
        isDoubleOverline = true;
        activeNote = onsetsInBeat[0];
      } else if (onsetsInBeat.length >= 2) {
        // Dvitiya Kaalam (2nd speed): 2 notes in 1 akshara with overline
        cellDisplay = onsetsInBeat.map(n => formatCarnaticSwara(n, raga).symbol).join(' ');
        cellType = 'dvitiya';
        isOverline = true;
        isDoubleOverline = false;
        activeNote = onsetsInBeat[0];
      } else {
        // No new onset: check if previous note is sustained across into this beat
        const prevNote = noteCursor > 0 ? notes[noteCursor - 1] : null;
        if (prevNote && prevNote.endTime > beatStartTime + (aksharaSec * 0.35)) {
          // Sustained note extension (Karvai = 1 akshara hold)
          cellDisplay = ',';
          cellType = 'karvai';
          activeNote = prevNote;
        } else {
          // Silence / breath pause / rest (Vishranti)
          cellDisplay = '·';
          cellType = 'rest';
          activeNote = null;
        }
      }

      // Anga assignment per active Tala
      let cumulative = 0;
      let currentAnga = talaDef.angas[0];
      let divider = '';
      for (let i = 0; i < talaDef.angas.length; i++) {
        cumulative += talaDef.angas[i].count;
        if (beat <= cumulative) {
          currentAnga = talaDef.angas[i];
          if (beat === cumulative) {
            divider = (i === talaDef.angas.length - 1) ? '||' : '|';
          }
          break;
        }
      }

      const kriya = talaDef.beats[beat - 1] || talaDef.beats[0];

      aksharaCells.push({
        aksharaNumber,
        beat,
        anga: currentAnga.label,
        divider,
        display: cellDisplay,
        type: cellType,
        isOverline,
        isDoubleOverline,
        startTime: Number(beatStartTime.toFixed(2)),
        endTime: Number(beatEndTime.toFixed(2)),
        kriya,
        note: activeNote,
      });
    }

    avartanams.push({
      avartanamIndex: aIdx + 1,
      startTime: aksharaCells[0].startTime,
      endTime: aksharaCells[totalBeats - 1].endTime,
      cells: aksharaCells,
    });
  }

  // Prune any empty trailing avartanams that contain only rests (no swaras or karvais)
  while (avartanams.length > 1) {
    const lastAv = avartanams[avartanams.length - 1];
    const hasMusicalContent = lastAv.cells.some(c => c.type === 'swara' || c.type === 'dvitiya' || c.type === 'tritiya' || c.type === 'karvai');
    if (!hasMusicalContent) {
      avartanams.pop();
    } else {
      break;
    }
  }

  return {
    talaId: talaDef.id,
    talaName: talaDef.name,
    shortName: talaDef.shortName,
    nadai: talaDef.nadai,
    sign: talaDef.sign,
    totalBeats,
    angas: talaDef.angas,
    aksharaSec,
    bpm: Math.round(60 / aksharaSec),
    avartanams,
  };
}

/**
 * Backward compatibility alias for Adi Tala Grid.
 */
export function buildAdiTalaGrid(notes, audioDuration, customAksharaSec = null, raga = null) {
  return buildTalaGrid(notes, audioDuration, customAksharaSec, raga, 'adi');
}

/**
 * Automatically evaluates recorded audio against Carnatic Talas.
 * Uses:
 * 1. Samam (Downbeat) Phase Alignment & Pulse Coherence
 * 2. Musical phrase boundary modulo & Karvai alignment
 * 3. Total cycle duration fit
 * 4. Carnatic repertory prior weighting
 */
export function detectCarnaticTala(notes, audioDuration, customAksharaSec = null) {
  if (!notes || notes.length < 3) {
    return {
      detectedTalaId: 'adi',
      tala: CARNATIC_TALAS.adi,
      confidence: 0.85,
      kala: 'Madhyama Kala',
      bpm: 76,
      reason: 'Standard Carnatic Tala default (Insufficient onsets for distinctive periodicity)',
      rankings: Object.values(CARNATIC_TALAS).map(t => ({ id: t.id, name: t.name, score: t.prior })),
    };
  }

  // Beat structure and tempo
  const beatInfo = detectBeatStructure(notes, audioDuration);
  const aksharaSec = customAksharaSec || beatInfo.aksharaSec || 0.54;
  const bpm = Math.round(60 / aksharaSec);

  // Extract onsets and weights
  const noteStarts = [];
  const weights = [];
  for (let i = 0; i < notes.length; i++) {
    const n = notes[i];
    noteStarts.push(n.startTime);
    // Weight notes by duration and stability (tonic / fifth notes get higher weight)
    const dur = Math.max(0.1, n.endTime - n.startTime);
    let swaraWeight = 1.0;
    const swIdx = getSwaraIndex(n);
    if (swIdx === 0 || swIdx === 7) swaraWeight = 1.4; // Sa or Pa
    weights.push(dur * swaraWeight);
  }
  const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;

  // Evaluate candidate Talas
  const candidateKeys = Object.keys(CARNATIC_TALAS);
  const scores = [];

  for (const key of candidateKeys) {
    const tala = CARNATIC_TALAS[key];
    const N = tala.totalBeats;
    const cycleSec = N * aksharaSec;

    // 1. Downbeat (Samam) phase alignment search across cycle
    let bestPhaseScore = 0;
    const numPhases = 16;
    for (let p = 0; p < numPhases; p++) {
      const phaseOffset = (p / numPhases) * cycleSec;
      let phaseEnergy = 0;
      for (let i = 0; i < noteStarts.length; i++) {
        const t = noteStarts[i];
        const cyclePos = ((t - phaseOffset) % cycleSec + cycleSec) % cycleSec;
        const normalizedPos = cyclePos / cycleSec; // 0.0 to 1.0
        
        // Circular distance to nearest beat onset
        const beatPos = (normalizedPos * N) % 1.0;
        const beatDist = Math.min(beatPos, 1.0 - beatPos);
        const beatMatch = Math.max(0, 1.0 - (beatDist * 3.5));

        // Extra bonus for hitting the Samam (cycle start)
        const samamDist = Math.min(normalizedPos, 1.0 - normalizedPos);
        const samamMatch = Math.max(0, 1.0 - (samamDist * 4.0));

        phaseEnergy += weights[i] * (beatMatch * 0.7 + samamMatch * 0.8);
      }
      if (phaseEnergy > bestPhaseScore) {
        bestPhaseScore = phaseEnergy;
      }
    }
    const normalizedPhaseScore = bestPhaseScore / totalWeight;

    // 2. Phrase boundary / silence alignment
    let phraseAlignScore = 0;
    let phraseCount = 0;
    for (let i = 0; i < notes.length - 1; i++) {
      const pause = notes[i + 1].startTime - notes[i].endTime;
      if (pause > 0.18) {
        phraseCount++;
        const endBeat = notes[i].endTime / aksharaSec;
        const rem = endBeat % N;
        const distToBoundary = Math.min(rem, N - rem);
        if (distToBoundary < 0.6) {
          phraseAlignScore += 1.0;
        }
      }
    }
    const phraseBonus = phraseCount > 0 ? (phraseAlignScore / phraseCount) * 0.25 : 0.1;

    // 3. Audio duration fit: does total duration fit an integer number of avartanams?
    const lastTime = notes[notes.length - 1].endTime;
    const cyclesFit = lastTime / cycleSec;
    const cycleFitRem = Math.abs(cyclesFit - Math.round(cyclesFit));
    const durationFitScore = Math.max(0, 1.0 - cycleFitRem * 2.5) * 0.15;

    // 4. Carnatic Prior
    const priorBonus = Math.log(1 + tala.prior * 4) * 0.35;

    // Total composite score
    const totalScore = (normalizedPhaseScore * 0.45) + phraseBonus + durationFitScore + priorBonus;

    scores.push({
      id: tala.id,
      name: tala.name,
      shortName: tala.shortName,
      totalBeats: tala.totalBeats,
      tala,
      score: Number(totalScore.toFixed(4)),
    });
  }

  // Sort by score descending
  scores.sort((a, b) => b.score - a.score);

  const topCandidate = scores[0];
  const runnerUp = scores[1];
  const margin = runnerUp ? topCandidate.score - runnerUp.score : 0.3;
  const confidence = Math.min(0.98, Math.max(0.65, Number((0.72 + margin * 0.45).toFixed(2))));

  return {
    detectedTalaId: topCandidate.id,
    tala: topCandidate.tala,
    confidence,
    bpm,
    aksharaSec,
    kala: beatInfo.kala,
    rankings: scores,
  };
}

/**
 * Groups notes into natural musical phrases (Prayogas / Vakyas) for Free-Flow (Alapana) mode.
 */
export function buildFreeFlowPhrases(notes, raga = null) {
  if (!notes || notes.length === 0) return [];

  const phrases = [];
  let currentPhrase = [];
  const breathPauseThreshold = 0.20; // 200ms pause indicates breath / phrase boundary

  for (let i = 0; i < notes.length; i++) {
    const note = notes[i];
    const cFmt = formatCarnaticSwara(note, raga);

    if (currentPhrase.length === 0) {
      currentPhrase.push({ ...note, cFmt });
    } else {
      const prevNote = currentPhrase[currentPhrase.length - 1];
      const pause = note.startTime - prevNote.endTime;

      if (pause >= breathPauseThreshold) {
        phrases.push(finishPhrase(currentPhrase, phrases.length + 1));
        currentPhrase = [{ ...note, cFmt }];
      } else {
        currentPhrase.push({ ...note, cFmt });
      }
    }
  }

  if (currentPhrase.length > 0) {
    phrases.push(finishPhrase(currentPhrase, phrases.length + 1));
  }

  return phrases;
}

function finishPhrase(notes, phraseNumber) {
  const start = notes[0].startTime;
  const end = notes[notes.length - 1].endTime;
  const dur = Number((end - start).toFixed(2));

  // Build Carnatic sargam representation with karvais based on duration
  const tokens = [];
  for (const n of notes) {
    tokens.push(n.cFmt.symbol);
    // Corrected Karvai logic: strictly mutually exclusive
    if (n.duration >= 0.85) {
      tokens.push(';'); // 2 akshara hold
    } else if (n.duration >= 0.50) {
      tokens.push(','); // 1 akshara hold
    }
  }

  return {
    phraseNumber,
    startTime: start,
    endTime: end,
    duration: dur,
    notes,
    sargamText: tokens.join('  '),
    swaraCount: notes.length,
  };
}

/**
 * Algorithmic Carnatic Raga Identification from active Swarasthanas.
 * Matches the sung pitch classes against the 72 Melakartas and prominent Janyas.
 */
export function identifyCarnaticRaga(notes) {
  if (!notes || notes.length === 0) return MELAKARTA_RAGAS[0];

  // Map each sung note to its semitone (0 to 11) relative to tonic Sa
  const sungSemitones = new Set();
  const semitoneWeights = {};

  notes.forEach(n => {
    const swIdx = getSwaraIndex(n);
    sungSemitones.add(swIdx);
    semitoneWeights[swIdx] = (semitoneWeights[swIdx] || 0) + (n.duration || 0.3);
  });

  let bestRaga = MELAKARTA_RAGAS[0];
  let highestScore = -1.0;

  for (const raga of MELAKARTA_RAGAS) {
    const ragaSet = new Set(raga.semitones);
    let matchWeight = 0;
    let totalWeight = 0;

    for (const [semiStr, weight] of Object.entries(semitoneWeights)) {
      const semi = Number(semiStr);
      totalWeight += weight;
      if (ragaSet.has(semi)) {
        matchWeight += weight;
      }
    }

    // Penalize notes outside the raga scale
    const outsidePenalty = (totalWeight - matchWeight) * 1.5;
    // Reward matching expected swaras
    const completeness = raga.semitones.filter(s => sungSemitones.has(s)).length / raga.semitones.length;
    const score = (matchWeight - outsidePenalty) * (0.5 + 0.5 * completeness);

    if (score > highestScore) {
      highestScore = score;
      bestRaga = raga;
    }
  }

  return bestRaga;
}

/**
 * Generates classical Carnatic manuscript text notation (formatted with Tala barlines).
 */
export function exportCarnaticTextNotation(adiTalaData, raga, tonic) {
  if (!adiTalaData || !adiTalaData.avartanams || adiTalaData.avartanams.length === 0) {
    return 'No Carnatic Tala notation available.';
  }

  const ragaName = raga ? (raga.name || raga) : 'Abhogi';
  const saHz = tonic ? (tonic.tonicFreq || tonic.saHz || 147.2) : 147.2;
  const saNote = tonic ? (tonic.tonicWestern || tonic.note || 'D3') : 'D3';

  let out = `Rāgam: ${ragaName}\nThāḷam: Ādi (Chaturasra Jāti Triputa — 8 Aksharas)\nŚruti: ${saNote} (${saHz} Hz)\n`;
  out += `=============================================================================\n`;
  out += `   1       2       3       4     |    5       6     |    7       8\n`;
  out += ` Laghu (4)                       |  Dhrutam (2)     |  Dhrutam (2)\n`;
  out += `-----------------------------------------------------------------------------\n`;

  adiTalaData.avartanams.forEach((av, idx) => {
    const cells = av.cells || av.aksharas || [];
    const laghuAksharas = cells.slice(0, 4);
    const dhrutam1Aksharas = cells.slice(4, 6);
    const dhrutam2Aksharas = cells.slice(6, 8);

    const fmtAkshara = (ak) => {
      if (!ak) return '·'.padEnd(7);
      let text = ak.display || (ak.swaras ? ak.swaras.map(s => s.symbol).join(' ') : '·');
      return text.padEnd(7);
    };

    const lText = laghuAksharas.map(fmtAkshara).join(' ');
    const d1Text = dhrutam1Aksharas.map(fmtAkshara).join(' ');
    const d2Text = dhrutam2Aksharas.map(fmtAkshara).join(' ');

    out += `| ${lText} |  ${d1Text} |  ${d2Text} ‖   (Avartanam ${idx + 1})\n`;
  });

  out += `=============================================================================\n`;
  return out;
}

