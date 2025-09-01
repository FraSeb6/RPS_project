# models/CNN_2/datacleaning.py
import os, random, shutil, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
DEF_INPUT  = SCRIPT_DIR / "data"        # paper/rock/scissors
DEF_OUTPUT = SCRIPT_DIR / "data_clean"  # destinazione

def parse_args():
    p = argparse.ArgumentParser("Green removal + soft edges + muted backgrounds + letterbox")
    p.add_argument("--input_dir",  type=Path, default=DEF_INPUT)
    p.add_argument("--output_dir", type=Path, default=DEF_OUTPUT)
    p.add_argument("--reset", choices=["clean","backup","keep"], default="clean")
    p.add_argument("--width",  type=int, default=300)
    p.add_argument("--height", type=int, default=200)
    p.add_argument("--seed",   type=int, default=42)
    # HSV soglia verde (puoi regolarla se serve)
    p.add_argument("--hsv_lower", type=int, nargs=3, default=[35, 35, 35])
    p.add_argument("--hsv_upper", type=int, nargs=3, default=[90, 255, 255])
    # intensità bordi morbidi (px del blur)
    p.add_argument("--feather", type=int, default=5)
    # intensità di augmentation luminanza/contrasto
    p.add_argument("--alpha_min", type=float, default=0.85)
    p.add_argument("--alpha_max", type=float, default=1.15)
    p.add_argument("--beta_abs",  type=int,   default=20)
    return p.parse_args()

ARGS = parse_args()
INPUT_BASE  = ARGS.input_dir.resolve()
OUTPUT_BASE = ARGS.output_dir.resolve()
TARGET_SIZE = (ARGS.width, ARGS.height)  # (w,h) per OpenCV
SEED        = ARGS.seed
LOWER_GREEN = np.array(ARGS.hsv_lower, np.uint8)
UPPER_GREEN = np.array(ARGS.hsv_upper, np.uint8)

# palette “muted” in BGR (no colori ultra saturi)
PALETTE = [
    (210, 210, 210),  # grigio chiaro
    (200, 200, 230),  # grigio-azzurro
    (210, 230, 210),  # verdino spento
    (220, 205, 220),  # lilla spento
    (200, 210, 220),  # blu-grigio
]

def choose_bg_color():
    return np.array(random.choice(PALETTE), dtype=np.uint8)

def skin_mask_bgr(img_bgr):
    """Maschera della pelle in YCrCb (range classico)."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77],  dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask  = cv2.inRange(ycrcb, lower, upper)
    return mask

def green_mask(img_bgr):
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    # pulizia base
    mask  = cv2.medianBlur(mask, 5)
    kernel = np.ones((5,5), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, 1)
    # preserva pelle (evita di “mangiare” la mano)
    skin  = cv2.dilate(skin_mask_bgr(img_bgr), kernel, 1)
    mask  = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    return mask  # 255=background da sostituire

def composite_soft(img_bgr, bg_bgr, mask_bg, feather=5):
    """Composita con bordi morbidi usando un alpha derivato dalla maschera."""
    alpha = cv2.GaussianBlur(mask_bg, (0,0), feather).astype(np.float32)/255.0
    alpha = alpha[..., None]  # HxWx1
    out = (img_bgr * (1.0 - alpha) + bg_bgr * alpha).astype(np.uint8)
    return out

def letterbox(img_bgr, target_size, fill_color):
    """Mantiene aspect ratio: ridimensiona e centra su canvas di colore 'fill_color'."""
    tw, th = target_size
    h, w = img_bgr.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.full((th, tw, 3), fill_color, dtype=np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def jitter_light(img_bgr, a_min=0.85, a_max=1.15, b_abs=20):
    alpha = random.uniform(a_min, a_max)
    beta  = random.randint(-b_abs, b_abs)
    return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

def rotate_or_flip(img_bgr):
    op = random.choice(["none", "flip", "rot"])
    if op == "flip":
        return cv2.flip(img_bgr, 1)
    if op == "rot":
        angle = random.choice([-15, -8, 8, 15])
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return img_bgr

def process_one(img_bgr):
    mask = green_mask(img_bgr)
    bg_color = choose_bg_color()
    bg = np.full_like(img_bgr, bg_color)
    comp = composite_soft(img_bgr, bg, mask, feather=ARGS.feather)
    comp = rotate_or_flip(comp)
    comp = jitter_light(comp, ARGS.alpha_min, ARGS.alpha_max, ARGS.beta_abs)
    # letterbox a fine pipeline, usando lo stesso colore dello sfondo
    comp = letterbox(comp, TARGET_SIZE, fill_color=bg_color)
    return comp

def is_image(name): return name.lower().endswith((".jpg",".jpeg",".png"))

def main():
    if not INPUT_BASE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_BASE}")
    random.seed(SEED); np.random.seed(SEED)

    if OUTPUT_BASE.exists():
        if ARGS.reset == "clean":
            shutil.rmtree(OUTPUT_BASE)
        elif ARGS.reset == "backup":
            bkp = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.name}_old_{datetime.now():%Y%m%d_%H%M%S}")
            OUTPUT_BASE.rename(bkp)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    classes = [d for d in INPUT_BASE.iterdir() if d.is_dir()]
    total = 0
    for cls in sorted(classes):
        out_dir = OUTPUT_BASE / cls.name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in os.listdir(cls) if is_image(f)]
        print(f"- {cls.name}: {len(files)} files")
        for fname in files:
            src = cls / fname
            img = cv2.imread(str(src))
            if img is None:
                print("  ! cannot read:", src); continue
            out = process_one(img)
            ok = cv2.imwrite(str(out_dir / fname), out)
            if ok: total += 1
    print(f"\n✅ wrote {total} images to {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
