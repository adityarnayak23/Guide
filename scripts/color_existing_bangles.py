#!/usr/bin/env python3
"""
Color the existing hand-drawn bangles in all Carnatic mudra illustrations
using rich, luminous 22k South Indian temple gold across all 5 gestures.

Enhances the golden warmth, luster, and yellow-amber saturation while preserving
all dark ink engraving hatch lines, line weights, and paper backgrounds.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUDRAS_DIR = os.path.join(BASE_DIR, 'public', 'assets', 'mudras')

def apply_enhanced_gold(image_rgb, mask_float, is_nadu=False):
    """
    Applies rich, luminous 22k temple gold coloration to the masked area,
    preserving dark ink lines (L < 50) and gating out parchment paper background.
    """
    arr = np.array(image_rgb)
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Gate out paper background (fade to 0 between 215 and 232 luminance)
    gate = np.clip((232.0 - lum) / (232.0 - 215.0), 0.0, 1.0)
    final_mask = mask_float * gate
    
    # In Nadu viral, protect the natural crimson rubies from being tinted gold
    if is_nadu:
        ruby_mask = (r > 100) & (g < 65) & (b < 85)
        final_mask = final_mask * (1.0 - ruby_mask.astype(float))
        
    dl = np.clip(lum / 50.0, 0.0, 1.0)
    l_norm = np.clip(lum / 255.0, 0.0, 1.0)
    
    # Enhanced 22k South Indian Temple Gold Formula:
    # Rich warm amber undertones, vibrant gold midtones, lustrous yellow-gold highlights
    gb_r = np.clip((55 + 200 * (l_norm ** 0.72)) * dl + lum * (1 - dl), 0, 255)
    gb_g = np.clip((35 + 175 * (l_norm ** 0.82)) * dl + lum * (1 - dl), 0, 255)
    gb_b = np.clip((5 + 65 * (l_norm ** 1.50)) * dl + lum * (1 - dl), 0, 255)
    
    r_out = r * (1 - final_mask) + gb_r * final_mask
    g_out = g * (1 - final_mask) + gb_g * final_mask
    b_out = b * (1 - final_mask) + gb_b * final_mask
    
    return Image.fromarray(np.stack([r_out, g_out, b_out], axis=2).astype(np.uint8))

def build_thattu_mask():
    mask = Image.new('L', (1200, 896), 0)
    d = ImageDraw.Draw(mask)
    x_b, y_b = 710, 220
    poly_outer = [
        (x_b + 0, y_b + 70), (x_b + 20, y_b + 38), (x_b + 50, y_b + 15), (x_b + 85, y_b + 4),
        (x_b + 120, y_b + 10), (x_b + 145, y_b + 30), (x_b + 165, y_b + 60), (x_b + 168, y_b + 115),
        (x_b + 155, y_b + 155), (x_b + 138, y_b + 185), (x_b + 112, y_b + 215), (x_b + 85, y_b + 235),
        (x_b + 55, y_b + 242), (x_b + 25, y_b + 210), (x_b + 0, y_b + 140)
    ]
    d.polygon(poly_outer, fill=255)
    poly_inner = [
        (x_b + 5, y_b + 78), (x_b + 30, y_b + 115), (x_b + 55, y_b + 160),
        (x_b + 65, y_b + 195), (x_b + 45, y_b + 230), (x_b + 0, y_b + 230), (x_b + 0, y_b + 80)
    ]
    d.polygon(poly_inner, fill=0)
    poly_back = [
        (x_b + 48, y_b + 165), (x_b + 70, y_b + 195), (x_b + 65, y_b + 225),
        (x_b + 35, y_b + 235), (x_b + 25, y_b + 215), (x_b + 40, y_b + 180)
    ]
    d.polygon(poly_back, fill=255)
    return np.array(mask.filter(ImageFilter.GaussianBlur(radius=1.0))) / 255.0

def build_mothira_mask():
    mask = Image.new('L', (1200, 896), 0)
    d = ImageDraw.Draw(mask)
    x_b, y_b = 300, 520
    poly_outer = [
        (x_b + 165, y_b + 0), (x_b + 165, y_b + 38), (x_b + 145, y_b + 25), (x_b + 125, y_b + 28),
        (x_b + 100, y_b + 45), (x_b + 75, y_b + 62), (x_b + 50, y_b + 80), (x_b + 42, y_b + 115),
        (x_b + 42, y_b + 155), (x_b + 52, y_b + 195), (x_b + 68, y_b + 235), (x_b + 95, y_b + 270),
        (x_b + 130, y_b + 295), (x_b + 165, y_b + 300), (x_b + 205, y_b + 290), (x_b + 240, y_b + 270),
        (x_b + 275, y_b + 245), (x_b + 275, y_b + 160), (x_b + 200, y_b + 160)
    ]
    d.polygon(poly_outer, fill=255)
    poly_inner = [
        (x_b + 165, y_b + 0), (x_b + 165, y_b + 38), (x_b + 152, y_b + 75), (x_b + 142, y_b + 115),
        (x_b + 130, y_b + 150), (x_b + 120, y_b + 185), (x_b + 115, y_b + 215), (x_b + 135, y_b + 225),
        (x_b + 165, y_b + 200), (x_b + 200, y_b + 175), (x_b + 275, y_b + 155), (x_b + 275, y_b + 0)
    ]
    d.polygon(poly_inner, fill=0)
    return np.array(mask.filter(ImageFilter.GaussianBlur(radius=1.0))) / 255.0

def build_nadu_mask():
    mask = Image.new('L', (1200, 896), 0)
    d = ImageDraw.Draw(mask)
    x_b, y_b = 260, 530
    poly_outer = [
        (x_b + 185, y_b + 5), (x_b + 215, y_b + 10), (x_b + 230, y_b + 45), (x_b + 240, y_b + 95),
        (x_b + 240, y_b + 150), (x_b + 225, y_b + 210), (x_b + 205, y_b + 260), (x_b + 175, y_b + 295),
        (x_b + 140, y_b + 295), (x_b + 115, y_b + 260), (x_b + 95, y_b + 195), (x_b + 92, y_b + 125),
        (x_b + 105, y_b + 65), (x_b + 130, y_b + 25), (x_b + 160, y_b + 5), (x_b + 185, y_b + 5)
    ]
    d.polygon(poly_outer, fill=255)
    poly_inner = [
        (x_b + 185, y_b + 15), (x_b + 205, y_b + 20), (x_b + 205, y_b + 55), (x_b + 200, y_b + 100),
        (x_b + 195, y_b + 150), (x_b + 180, y_b + 200), (x_b + 155, y_b + 235), (x_b + 145, y_b + 205),
        (x_b + 155, y_b + 140), (x_b + 155, y_b + 80), (x_b + 165, y_b + 40), (x_b + 185, y_b + 15)
    ]
    d.polygon(poly_inner, fill=0)
    poly_back = [
        (x_b + 145, y_b + 205), (x_b + 155, y_b + 235), (x_b + 180, y_b + 260),
        (x_b + 225, y_b + 220), (x_b + 200, y_b + 200)
    ]
    d.polygon(poly_back, fill=255)
    poly_baju = [
        (x_b + 0, y_b + 35), (x_b + 60, y_b + 35), (x_b + 95, y_b + 55), (x_b + 80, y_b + 85),
        (x_b + 45, y_b + 115), (x_b + 0, y_b + 130), (x_b + 0, y_b + 35)
    ]
    d.polygon(poly_baju, fill=255)
    return np.array(mask.filter(ImageFilter.GaussianBlur(radius=1.0))) / 255.0

def build_veechu_mask():
    mask = Image.new('L', (1200, 896), 0)
    d = ImageDraw.Draw(mask)
    x_b, y_b = 680, 450
    poly_v1 = [
        (x_b + 80, y_b + 5), (x_b + 115, y_b + 18), (x_b + 145, y_b + 55), (x_b + 158, y_b + 95),
        (x_b + 158, y_b + 130), (x_b + 148, y_b + 165), (x_b + 128, y_b + 205), (x_b + 105, y_b + 230),
        (x_b + 75, y_b + 242), (x_b + 40, y_b + 242), (x_b + 15, y_b + 215), (x_b + 8, y_b + 175),
        (x_b + 35, y_b + 145), (x_b + 55, y_b + 100), (x_b + 72, y_b + 50)
    ]
    d.polygon(poly_v1, fill=255)
    poly_v1_inner = [
        (x_b + 80, y_b + 5), (x_b + 70, y_b + 55), (x_b + 50, y_b + 105), (x_b + 30, y_b + 150),
        (x_b + 10, y_b + 170), (x_b + 10, y_b + 0), (x_b + 80, y_b + 0)
    ]
    d.polygon(poly_v1_inner, fill=0)
    poly_v2 = [
        (x_b + 160, y_b + 65), (x_b + 185, y_b + 75), (x_b + 205, y_b + 115), (x_b + 202, y_b + 155),
        (x_b + 175, y_b + 205), (x_b + 150, y_b + 245), (x_b + 120, y_b + 282), (x_b + 95, y_b + 285),
        (x_b + 70, y_b + 248), (x_b + 110, y_b + 190), (x_b + 145, y_b + 135), (x_b + 160, y_b + 65)
    ]
    d.polygon(poly_v2, fill=255)
    poly_v2_inner = [
        (x_b + 160, y_b + 65), (x_b + 180, y_b + 90), (x_b + 190, y_b + 130), (x_b + 175, y_b + 175),
        (x_b + 145, y_b + 225), (x_b + 120, y_b + 240), (x_b + 120, y_b + 190), (x_b + 145, y_b + 135)
    ]
    d.polygon(poly_v2_inner, fill=0)
    return np.array(mask.filter(ImageFilter.GaussianBlur(radius=1.0))) / 255.0

def process_all_mudras():
    print("Coloring existing bangles in enhanced, luminous 22k temple gold across all 5 mudras...")
    
    clean_images = {
        'thattu': Image.open(os.path.join(MUDRAS_DIR, 'thattu_clean.png')).convert('RGB'),
        'chinna': Image.open(os.path.join(MUDRAS_DIR, 'chinna_clean.png')).convert('RGB'),
        'mothira': Image.open(os.path.join(MUDRAS_DIR, 'mothira_clean.png')).convert('RGB'),
        'nadu': Image.open(os.path.join(MUDRAS_DIR, 'nadu_clean.png')).convert('RGB'),
        'veechu': Image.open(os.path.join(MUDRAS_DIR, 'veechu_clean.png')).convert('RGB'),
    }
    
    masks = {
        'thattu': build_thattu_mask(),
        'mothira': build_mothira_mask(),
        'nadu': build_nadu_mask(),
        'veechu': build_veechu_mask(),
    }
    
    colored_mudras = {}
    
    # 1. Thattu, Mothira, Nadu, Veechu
    for name in ['thattu', 'mothira', 'nadu', 'veechu']:
        colored_mudras[name] = apply_enhanced_gold(
            clean_images[name], masks[name], is_nadu=(name == 'nadu')
        )
        
    # 2. Chinna: Enhance existing bronze/ochre bangles with pixel-perfect gold saturation
    chinna_im = clean_images['chinna']
    arr_c = np.array(chinna_im)
    rc = arr_c[:, :, 0].astype(float); gc = arr_c[:, :, 1].astype(float); bc = arr_c[:, :, 2].astype(float)
    lum_c = 0.299 * rc + 0.587 * gc + 0.114 * bc
    is_wrist_zone = np.zeros(arr_c.shape[:2], dtype=bool)
    is_wrist_zone[530:800, 240:460] = True
    is_gold_c = is_wrist_zone & (rc > 90) & (gc > 65) & (bc < 115) & (rc >= gc) & (gc > bc * 1.18) & (lum_c < 225)
    dl_c = np.clip(lum_c / 50.0, 0.0, 1.0)
    ln_c = np.clip(lum_c / 255.0, 0.0, 1.0)
    c_gr = np.clip((55 + 200 * (ln_c ** 0.72)) * dl_c + lum_c * (1 - dl_c), 0, 255)
    c_gg = np.clip((35 + 175 * (ln_c ** 0.82)) * dl_c + lum_c * (1 - dl_c), 0, 255)
    c_gb = np.clip((5 + 65 * (ln_c ** 1.50)) * dl_c + lum_c * (1 - dl_c), 0, 255)
    r_out_c = rc.copy(); g_out_c = gc.copy(); b_out_c = bc.copy()
    r_out_c[is_gold_c] = c_gr[is_gold_c]; g_out_c[is_gold_c] = c_gg[is_gold_c]; b_out_c[is_gold_c] = c_gb[is_gold_c]
    colored_mudras['chinna'] = Image.fromarray(np.stack([r_out_c, g_out_c, b_out_c], axis=2).astype(np.uint8))
            
    # Save production WebP files
    for name, im in colored_mudras.items():
        # 1. light_{name}.webp
        p_light = os.path.join(MUDRAS_DIR, f'light_{name}.webp')
        im.save(p_light, 'WEBP', quality=95)
        print(f"Saved {p_light}")
        
        # 2. {name}.webp
        p_base = os.path.join(MUDRAS_DIR, f'{name}.webp')
        im.save(p_base, 'WEBP', quality=95)
        print(f"Saved {p_base}")
        
        # 3. dark_{name}.webp (dark background variant)
        arr = np.array(im)
        r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Parchment background (lum > 225) -> Deep obsidian dark (#0E0A07)
        bg_factor = np.clip((lum - 215) / (235 - 215), 0.0, 1.0)
        
        dark_bg = np.array([14.0, 10.0, 7.0])
        fg_col = np.stack([r, g, b], axis=2)
        dark_res = fg_col * (1 - bg_factor[:, :, None]) + dark_bg * bg_factor[:, :, None]
        dark_im = Image.fromarray(np.clip(dark_res, 0, 255).astype(np.uint8))
        
        p_dark = os.path.join(MUDRAS_DIR, f'dark_{name}.webp')
        dark_im.save(p_dark, 'WEBP', quality=95)
        print(f"Saved {p_dark}")
        
    print("All mudra images successfully updated with enhanced 22k gold bangles!")

if __name__ == '__main__':
    process_all_mudras()
