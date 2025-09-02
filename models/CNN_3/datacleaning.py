# models/CNN_3/datacleaning.py
# Clean & augment RPS dataset for CNN_3 (revised)
from pathlib import Path
from typing import Tuple, List
import argparse
import shutil
import cv2
import numpy as np

CLASSES = ["rock", "paper", "scissors"]

# HSV green range (conservativo)
LOWER_GREEN = np.array([35, 35, 40], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)

# ----------------------------
# Paths & IO
# ----------------------------
def detect_input_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data" / "raw" / "rps-cv-images",
        base_dir / "data" / "raw",
        base_dir / "data",
    ]
    for cand in candidates:
        if all((cand / c).exists() for c in CLASSES):
            return cand
    raise FileNotFoundError(
        f"Input folder not found. Expected one of:\n"
        f" - {candidates[0]}\n - {candidates[1]}\n - {candidates[2]}\n"
        "Each containing subfolders: {rock,paper,scissors}"
    )

def ensure_clean_dir(path: Path, fresh: bool = True) -> None:
    if fresh and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def is_image_file(name: str) -> bool:
    name = name.lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))

# ----------------------------
# Geometry helpers
# ----------------------------
def center_crop_to_aspect(bgr: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    cur_aspect = w / h
    if abs(cur_aspect - target_aspect) < 1e-3:
        return bgr
    if cur_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return bgr[:, x0 : x0 + new_w]
    else:
        new_h = int(round(w / target_aspect))
        y0 = (h - new_h) // 2
        return bgr[y0 : y0 + new_h, :]

def smart_resize(bgr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    tgt_w, tgt_h = target_size
    h, w = bgr.shape[:2]
    if tgt_w < w or tgt_h < h:
        interp = cv2.INTER_AREA      # downscale = antialias
    else:
        interp = cv2.INTER_CUBIC     # upscale = nitido ma non seghettato
    return cv2.resize(bgr, (tgt_w, tgt_h), interpolation=interp)

# ----------------------------
# Color / enhancement
# ----------------------------
def grayworld_white_balance(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    kb, kg, kr = mean_gray / mean_b, mean_gray / mean_g, mean_gray / mean_r
    b = np.clip(b * kb, 0, 255)
    g = np.clip(g * kg, 0, 255)
    r = np.clip(r * kr, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

def clahe_on_foreground(bgr: np.ndarray, fg_mask: np.ndarray,
                        clip: float = 1.5, tiles: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applica CLAHE solo sui pixel del soggetto (evita sbruciature su sfondo bianco)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    L2 = L.copy()
    L2[fg_mask > 0] = clahe.apply(L[fg_mask > 0])
    lab2 = cv2.merge([L2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def light_denoise(bgr: np.ndarray) -> np.ndarray:
    # leggerissimo per non sfocare i bordi
    return cv2.bilateralFilter(bgr, d=5, sigmaColor=30, sigmaSpace=30)

# ----------------------------
# Green-screen removal (robusto + matte morbida)
# ----------------------------
def green_mask_hsv_dom(bgr: np.ndarray) -> np.ndarray:
    """Combina HSV inRange con un criterio di 'green dominance' per catturare verdi problematici."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # green dominance: G nettamente > R e B
    b, g, r = cv2.split(bgr.astype(np.int16))
    dom = (g - np.maximum(b, r)) > 25
    m2 = np.zeros_like(m1)
    m2[dom] = 255

    mask = cv2.bitwise_or(m1, m2)

    # chiudi piccoli buchi senza mangiare il bordo
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def soft_alpha_from_mask(bg_mask: np.ndarray, feather_px: int = 6) -> np.ndarray:
    """
    Converte una mask binaria (255 = background) in alpha morbida con distance transform.
    feather_px controlla la transizione (raggio ~ in pixel).
    """
    bg = (bg_mask > 127).astype(np.uint8)
    fg = 1 - bg
    # distanza dal foregound e dal background
    d_to_fg = cv2.distanceTransform(bg, distanceType=cv2.DIST_L2, maskSize=3)
    d_to_bg = cv2.distanceTransform(fg, distanceType=cv2.DIST_L2, maskSize=3)
    # alpha = 1 su foreground, 0 sul background, transizione morbida
    alpha = d_to_bg / (d_to_bg + d_to_fg + 1e-6)
    if feather_px > 0:
        ksz = max(3, int(2 * feather_px + 1) | 1)
        alpha = cv2.GaussianBlur(alpha, (ksz, ksz), feather_px)
    alpha = np.clip(alpha, 0, 1)
    return alpha.astype(np.float32)

def remove_green_and_add_white_bg(bgr: np.ndarray,
                                  feather_px: int = 6,
                                  do_despill: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Ritorna (composited_bgr, fg_mask_uint8). Edge puliti, niente scalini.
    """
    mask_bg = green_mask_hsv_dom(bgr)  # 255 = background
    alpha = soft_alpha_from_mask(mask_bg, feather_px=feather_px)

    # Despill solo in una corona sottile attorno ai bordi
    if do_despill:
        ring = cv2.Canny((mask_bg == 255).astype(np.uint8)*255, 50, 150)
        ring = cv2.dilate(ring, np.ones((3, 3), np.uint8), iterations=1) > 0
    else:
        ring = None

    bgr2 = bgr.astype(np.float32).copy()
    if ring is not None:
        g = bgr2[:, :, 1]
        rb = np.minimum(bgr2[:, :, 0], bgr2[:, :, 2]) * 1.05
        g[ring] = np.minimum(g[ring], rb[ring])

    white = np.full_like(bgr2, 255.0, dtype=np.float32)
    out = bgr2 * alpha[..., None] + white * (1.0 - alpha[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)

    fg_mask = (alpha > 0.5).astype(np.uint8) * 255
    return out, fg_mask

# ----------------------------
# Auto-crop del soggetto
# ----------------------------
def crop_to_subject(bgr: np.ndarray, fg_mask: np.ndarray,
                    pad_ratio: float = 0.06) -> np.ndarray:
    """Croppa al bounding box del soggetto + padding percentuale."""
    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return bgr
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h, w = bgr.shape[:2]
    pad = int(round(max(h, w) * pad_ratio))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return bgr[y1:y2, x1:x2]

# ----------------------------
# Augmentation
# ----------------------------
def strong_augmentation(bgr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    img = bgr.copy()
    if rng.rand() < 0.5:
        img = cv2.flip(img, 1)

    angle = float(rng.uniform(-15, 15))  # meno aggressivo
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    alpha = float(rng.uniform(0.85, 1.15))  # contrast
    beta  = int(rng.uniform(-25, 25))       # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# ----------------------------
# Processing pipeline
# ----------------------------
def process_image(
    img_path: Path,
    target_size: Tuple[int, int],
    remove_green: bool = True,
    do_white_balance: bool = False,   # default OFF per evitare dominanti
    do_clahe: bool = True,
    do_denoise: bool = False,
) -> np.ndarray:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise ValueError(f"Unreadable image: {img_path}")

    fg_mask = None
    if remove_green:
        # feather dinamico in base alla dimensione originale
        short_side = min(bgr.shape[:2])
        feather = max(4, int(short_side * 0.01))
        bgr, fg_mask = remove_green_and_add_white_bg(bgr, feather_px=feather, do_despill=True)

    if do_white_balance:
        bgr = grayworld_white_balance(bgr)

    if do_clahe:
        if fg_mask is None:
            # se non c'è green removal, stimiamo il foreground come "non quasi bianco"
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            fg_mask = (gray < 245).astype(np.uint8) * 255
        bgr = clahe_on_foreground(bgr, fg_mask)

    if do_denoise:
        bgr = light_denoise(bgr)

    # crop al soggetto per far riempire la frame, poi aspect e resize
    if fg_mask is not None:
        bgr = crop_to_subject(bgr, fg_mask, pad_ratio=0.06)

    tgt_w, tgt_h = target_size
    bgr = center_crop_to_aspect(bgr, target_aspect=tgt_w / tgt_h)
    bgr = smart_resize(bgr, (tgt_w, tgt_h))
    return bgr

# ----------------------------
# CLI
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CNN_3 Data Cleaning & Augmentation (revised)")
    p.add_argument("--width", type=int, default=300, help="Target width")
    p.add_argument("--height", type=int, default=200, help="Target height")
    p.add_argument("--augment", type=int, default=0, help="Augmentations per image")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-green", action="store_true", help="Disable green-screen removal")
    p.add_argument("--fresh", action="store_true", help="Delete existing output folder first")
    p.add_argument("--wb-on", action="store_true", help="Enable gray-world white balance")
    p.add_argument("--clahe-off", action="store_true", help="Disable CLAHE-on-foreground")
    p.add_argument("--denoise", action="store_true", help="Enable light denoising")
    return p

def main():
    print("==> CNN_3 Data Cleaning (revised)")

    base_dir = Path(__file__).resolve().parent
    try:
        input_dir = detect_input_dir(base_dir)
    except FileNotFoundError as e:
        print("❌", e)
        return

    output_dir = base_dir / "data_clean"
    args = build_argparser().parse_args()

    target_size = (args.width, args.height)
    rng = np.random.RandomState(args.seed)

    ensure_clean_dir(output_dir, fresh=args.fresh)

    # Prepare class folders
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    total_in, total_ok, total_fail, total_aug = 0, 0, 0, 0

    for cls in CLASSES:
        in_cls = input_dir / cls
        out_cls = output_dir / cls

        if not in_cls.exists():
            print(f"⚠️  Missing class folder: {in_cls}")
            continue

        files = sorted([p for p in in_cls.iterdir() if p.is_file() and is_image_file(p.name)])
        print(f"[{cls}] {len(files)} images")

        for p in files:
            total_in += 1
            try:
                img = process_image(
                    p,
                    target_size=target_size,
                    remove_green=not args.no_green,
                    do_white_balance=args.wb_on,
                    do_clahe=not args.clahe_off,
                    do_denoise=args.denoise,
                )
                # save base
                out_path = out_cls / p.name
                cv2.imwrite(str(out_path), img)
                total_ok += 1

                # augmentations (in-place size)
                for i in range(args.augment):
                    aug = strong_augmentation(img, rng)
                    aug_name = f"{p.stem}_aug{i+1}{p.suffix}"
                    cv2.imwrite(str(out_cls / aug_name), aug)
                    total_aug += 1

            except Exception as ex:
                total_fail += 1
                print(f"   ✖ Failed: {p.name} -> {ex}")

    print("\n==> Done")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Images    : {total_in}  | saved: {total_ok}  | failed: {total_fail}  | aug: {total_aug}")

if __name__ == "__main__":
    main()
