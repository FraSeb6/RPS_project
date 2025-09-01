# models/CNN_1/CNN_1.py
import warnings, urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import os, csv, json, random
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- Reproducibility --------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- CLI / Paths --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEF_DATA_DIR    = SCRIPT_DIR / "data_augmented"   # created by datacleaning.py (default)
DEF_OUTPUTS_DIR = SCRIPT_DIR / "outputs"

def parse_args():
    p = argparse.ArgumentParser("Train CNN_1 on Rock/Paper/Scissors")
    p.add_argument("--data_dir",    type=Path, default=DEF_DATA_DIR,
                   help="Directory containing class folders (default: CNN_1/data_augmented)")
    p.add_argument("--outputs_dir", type=Path, default=DEF_OUTPUTS_DIR,
                   help="Where to save models/plots/wrong predictions (default: CNN_1/outputs)")
    p.add_argument("--img_height", type=int, default=200)
    p.add_argument("--img_width",  type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=20)
    return p.parse_args()

ARGS = parse_args()
DATA_DIR = ARGS.data_dir.resolve()
OUT_DIR  = ARGS.outputs_dir.resolve()
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

# Run folders
RUN_ID   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR  = OUT_DIR / RUN_ID
MODEL_DIR = RUN_DIR / "models"
PLOTS_DIR = RUN_DIR / "plots"
WRONG_DIR = RUN_DIR / "wrong_predictions"
WRONG_IMG_DIR = WRONG_DIR / "images"
for d in (MODEL_DIR, PLOTS_DIR, WRONG_IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Data --------------------
img_height, img_width = ARGS.img_height, ARGS.img_width
batch_size, epochs    = ARGS.batch_size, ARGS.epochs
AUTOTUNE = tf.data.AUTOTUNE

train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    validation_split=0.2,
    subset='training',
    seed=SEED,
    shuffle=True
)
val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    validation_split=0.2,
    subset='validation',
    seed=SEED
)

# class names from the dataset (alphabetical by folder)
CLASS_NAMES = train_data.class_names

# Normalization + efficient pipeline
norm = lambda x, y: (x / 255.0, y)
train_data = train_data.map(norm, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_data   = val_data.map(norm,   num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# -------------------- Model (simple) --------------------
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# -------------------- Train --------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop]
)

# -------------------- Plot & Save --------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN_1 - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "accuracy.png", dpi=150)
plt.close()

# save model + history
model_path = MODEL_DIR / "predictor_CNN_1.keras"
model.save(model_path)
with open(RUN_DIR / "history.json", "w") as f:
    json.dump(history.history, f, indent=2)

# -------------------- Wrong predictions --------------------
rows, counter = [], 1
for image, label in val_data.unbatch():
    img_array = image.numpy()
    true_idx  = int(np.argmax(label.numpy()))
    pred      = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
    pred_idx  = int(np.argmax(pred))

    if pred_idx != true_idx:
        filename = f"wrong_{counter:03d}.jpg"
        path     = WRONG_IMG_DIR / filename
        img_uint8 = (img_array * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(path)
        rows.append([filename, CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx]])
        counter += 1

csv_path = WRONG_DIR / "wrong_preds.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "predicted_label"])
    writer.writerows(rows)

print(f"Saved model to: {model_path}")
print(f"Saved {len(rows)} validation errors in:\n - images: {WRONG_IMG_DIR}\n - CSV: {csv_path}")
print(f"Run dir: {RUN_DIR}")
