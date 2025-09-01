# models/CNN_1/datacleaning.py
import os, cv2, numpy as np, random, shutil, argparse
import cv2  # type: ignore
from pathlib import Path
from datetime import datetime

# ---------- Config / CLI ----------
SCRIPT_DIR = Path(__file__).resolve().parent
DEF_INPUT  = SCRIPT_DIR / "data"             # must contain class subfolders: paper/rock/scissors
DEF_OUTPUT = SCRIPT_DIR / "data_augmented"   # default destination

def parse_args():
    p = argparse.ArgumentParser("Data cleaning & offline augmentation")
    p.add_argument("--input_dir",  type=Path, default=DEF_INPUT,
                   help="Source folder with one subfolder per class (default: CNN_1/data)")
    p.add_argument("--output_dir", type=Path, default=DEF_OUTPUT,
                   help="Destination folder (default: CNN_1/data_augmented)")
    p.add_argument("--reset", choices=["clean","backup","keep"], default="clean",
                   help="If output exists: clean=delete, backup=rename, keep=leave as is (default: clean)")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite the input in place (ignores --reset and --output_dir)")
    p.add_argument("--width",  type=int, default=300, help="Resize width (default: 300)")
    p.add_argument("--height", type=int, default=200, help="Resize height (default: 200)")
    p.add_argument("--seed",   type=int, default=42,  help="Random seed for reproducibility")
    return p.parse_args()

ARGS = parse_args()
INPUT_BASE  = ARGS.input_dir.resolve()
OUTPUT_BASE = (INPUT_BASE if ARGS.inplace else ARGS.output_dir.resolve())
TARGET_SIZE = (ARGS.width, ARGS.height)  # OpenCV uses (width, height)
SEED        = ARGS.seed

# ---------- Augmentation ----------
def random_rotate(img, max_angle=25, min_angle=5):
    angle = random.choice([random.uniform(-max_angle, -min_angle),
                           random.uniform(min_angle, max_angle)])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def random_brightness_contrast(img, alpha_range=(0.6, 1.6), beta_range=(-60, 60)):
    alpha = random.uniform(*alpha_range)  # contrast
    beta  = random.randint(*beta_range)   # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def transform(img, p_flip=0.5, p_rotate=0.5):
    if random.random() < p_flip:
        img = cv2.flip(img, 1)
    if random.random() < p_rotate:
        img = random_rotate(img)
    return random_brightness_contrast(img)

def is_image_file(name: str) -> bool:
    n = name.lower()
    return n.endswith((".jpg", ".jpeg", ".png"))

# ---------- Main ----------
def main():
    if not INPUT_BASE.exists():
        raise FileNotFoundError(f"Source folder not found: {INPUT_BASE}")

    random.seed(SEED); np.random.seed(SEED)

    # prepare destination
    if not ARGS.inplace:
        if OUTPUT_BASE.exists():
            if ARGS.reset == "clean":
                shutil.rmtree(OUTPUT_BASE)
            elif ARGS.reset == "backup":
                bkp = OUTPUT_BASE.with_name(
                    f"{OUTPUT_BASE.name}_old_{datetime.now():%Y%m%d_%H%M%S}"
                )
                OUTPUT_BASE.rename(bkp)
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # classes = all subfolders in the input
    class_dirs = [d for d in INPUT_BASE.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found in {INPUT_BASE}")

    mode = "INPLACE (overwriting input)" if ARGS.inplace else f"→ {OUTPUT_BASE}"
    print(f"➡️  Offline augmentation\n   Input : {INPUT_BASE}\n   Output: {mode}\n   Size  : {TARGET_SIZE}\n   Seed  : {SEED}")

    total = 0
    for cls in sorted(class_dirs):
        dst = (cls if ARGS.inplace else OUTPUT_BASE / cls.name)
        dst.mkdir(parents=True, exist_ok=True)

        files = [f for f in os.listdir(cls) if is_image_file(f)]
        print(f"  - {cls.name}: {len(files)} files")
        for fname in files:
            src = cls / fname
            img = cv2.imread(str(src))
            if img is None:
                print(f"    ! Cannot read: {src}")
                continue
            img = cv2.resize(img, TARGET_SIZE)
            img = transform(img)
            ok = cv2.imwrite(str(dst / fname), img)
            if not ok:
                print(f"    ! Save error: {dst / fname}")
                continue
            total += 1

    print(f"\n✅ Done: {total} images processed in {OUTPUT_BASE if not ARGS.inplace else INPUT_BASE}")

if __name__ == "__main__":
    main()
