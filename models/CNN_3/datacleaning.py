# models/CNN_3/datacleaning.py
# Clean & augment RPS dataset for CNN_3
# - Input is auto-detected inside this folder (see detect_input_dir)
# - Output is written to: CNN_3/data_clean/{rock,paper,scissors}
# - Green-screen removal uses soft feather + despill to avoid jagged edges

from pathlib import Path
from typing import Tuple, List
import argparse
import shutil
import cv2
import numpy as np

# ----------------------------
# Defaults & constants
# ----------------------------
CLASSES = ["rock", "paper", "scissors"]
# HSV green range
LOWER_GREEN = np.array([35, 40, 40], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)

# ----------------------------
# Utility: paths & IO
# ----------------------------
def detect_input_dir(base_dir: Path) -> Path:
    """
    Try common layouts and return the first directory that contains
    the class subfolders (rock/paper/scissors).
    """
    candidates = [
        base_dir / "data" / "raw" / "rps-cv-images",
        base_dir / "data" / "raw",
        base_dir / "data",
    ]
    for cand in candidates:
        if all((cand / c).exists() for c in CLASSES):
            return cand
    raise FileNotFoundError(
        "Input folder not found. Expected one of:\n"
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
        # too wide -> crop width
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return bgr[:, x0 : x0 + new_w]
    else:
        # too tall -> crop height
        new_h = int(round(w / target_aspect))
        y0 = (h - new_h) // 2
        return bgr[y0 : y0 + new_h, :]

def smart_resize(bgr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    tgt_w, tgt_h = target_size
    h, w = bgr.shape[:2]
    interp = cv2.INTER_AREA if (tgt_w < w or tgt_h < h) else cv2.INTER_LANCZOS4
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
    out = cv2.merge([b, g, r]).astype(np.uint8)
    return out

def contrast_clahe(bgr: np.ndarray, clip: float = 2.0, tiles: Tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def light_denoise(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 15)

# ----------------------------
# Green-screen removal (soft)
# ----------------------------
def remove_green_and_add_white_bg_soft(
    bgr: np.ndarray,
    feather_px: int = 3,
    erode_px: int = 1,
    close_px: int = 3,
    do_despill: bool = True,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)  # 255 = background to remove

    # mask refinement
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        mask = cv2.erode(mask, k, iterations=1)

    # feather (soft edge)
    if feather_px > 0:
        ksz = max(3, int(6 * feather_px + 1))
        if ksz % 2 == 0:
            ksz += 1
        soft = cv2.GaussianBlur(mask, (ksz, ksz), feather_px)
    else:
        soft = mask

    # alpha: 1 on subject, 0 on background
    alpha = (255 - soft).astype(np.float32) / 255.0

    # despill: reduce green near the edge ring
    bgr2 = bgr.copy()
    if do_despill:
        ring = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1) - mask
        g = bgr2[:, :, 1].astype(np.float32)
        rb = np.minimum(bgr2[:, :, 0], bgr2[:, :, 2]).astype(np.float32)
        g = np.where(ring > 0, np.minimum(g, rb * 1.05), g)
        bgr2[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)

    white = np.full_like(bgr2, 255, dtype=np.uint8)
    out = bgr2.astype(np.float32) * alpha[..., None] + white.astype(np.float32) * (1.0 - alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)

# ----------------------------
# Augmentation
# ----------------------------
def strong_augmentation(bgr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    img = bgr.copy()
    if rng.rand() < 0.5:
        img = cv2.flip(img, 1)

    angle = float(rng.uniform(-25, 25))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LANCZOS4,           # anti-aliasing
        borderMode=cv2.BORDER_REFLECT_101
    )

    alpha = float(rng.uniform(0.7, 1.4))   # contrast
    beta  = int(rng.uniform(-40, 40))      # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# ----------------------------
# Processing pipeline
# ----------------------------
def process_image(
    img_path: Path,
    target_size: Tuple[int, int],
    remove_green: bool = True,
    do_white_balance: bool = True,
    do_clahe: bool = True,
    do_denoise: bool = False,
) -> np.ndarray:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise ValueError(f"Unreadable image: {img_path}")

    if remove_green:
        # ~1% of the shortest side as feather looks natural
        feather = max(2, int(min(target_size) * 0.01))
        bgr = remove_green_and_add_white_bg_soft(
            bgr, feather_px=feather, erode_px=1, close_px=3, do_despill=True
        )

    if do_white_balance:
        bgr = grayworld_white_balance(bgr)
    if do_clahe:
        bgr = contrast_clahe(bgr)
    if do_denoise:
        bgr = light_denoise(bgr)

    tgt_w, tgt_h = target_size
    bgr = center_crop_to_aspect(bgr, target_aspect=tgt_w / tgt_h)
    bgr = smart_resize(bgr, (tgt_w, tgt_h))
    return bgr

# ----------------------------
# Main
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CNN_3 Data Cleaning & Augmentation")
    p.add_argument("--width", type=int, default=300, help="Target width (default: 300)")
    p.add_argument("--height", type=int, default=200, help="Target height (default: 200)")
    p.add_argument("--augment", type=int, default=0, help="Augmentations per image (default: 0)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--no-green", action="store_true", help="Disable green-screen removal")
    p.add_argument("--fresh", action="store_true", help="Delete existing output folder first")
    p.add_argument("--wb-off", action="store_true", help="Disable gray-world white balance")
    p.add_argument("--clahe-off", action="store_true", help="Disable CLAHE")
    p.add_argument("--denoise", action="store_true", help="Enable light denoising")
    return p

def main():
    print("==> CNN_3 Data Cleaning")

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
                    do_white_balance=not args.wb_off,
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
