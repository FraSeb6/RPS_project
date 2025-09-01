# models/CNN_2/CNN_2.py
import os, csv, json, random, argparse, warnings
from pathlib import Path
from datetime import datetime

try:
    import urllib3
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# -------------------- Seeds --------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# -------------------- CLI / Paths --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEF_DATA_DIR    = SCRIPT_DIR / "data_clean"
DEF_OUTPUTS_DIR = SCRIPT_DIR / "outputs"

def parse_args():
    p = argparse.ArgumentParser("Train CNN_2 (clean BG, conservative)")
    p.add_argument("--data_dir",    type=Path, default=DEF_DATA_DIR)
    p.add_argument("--outputs_dir", type=Path, default=DEF_OUTPUTS_DIR)
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

RUN_ID   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR  = OUT_DIR / RUN_ID
MODEL_DIR = RUN_DIR / "models"
PLOTS_DIR = RUN_DIR / "plots"
WRONG_DIR = RUN_DIR / "wrong_predictions"
WRONG_IMG_DIR = WRONG_DIR / "images"
for d in (MODEL_DIR, PLOTS_DIR, WRONG_IMG_DIR): d.mkdir(parents=True, exist_ok=True)

# -------------------- Data --------------------
img_height, img_width = ARGS.img_height, ARGS.img_width
batch_size, epochs    = ARGS.batch_size, ARGS.epochs
AUTOTUNE = tf.data.AUTOTUNE

train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, labels='inferred', label_mode='int',
    batch_size=batch_size, image_size=(img_height, img_width),
    validation_split=0.2, subset='training', seed=SEED, shuffle=True
)
val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, labels='inferred', label_mode='int',
    batch_size=batch_size, image_size=(img_height, img_width),
    validation_split=0.2, subset='validation', seed=SEED
)
CLASS_NAMES = train_data.class_names

# Augmentation molto leggero (evitiamo rotazioni/contrasti per ora)
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
])

def norm(x, y): return (x / 255.0, y)

# cache -> augment (stocastica) -> normalize -> prefetch
train_data = (train_data
              .cache()
              .map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
              .map(norm, num_parallel_calls=AUTOTUNE)
              .prefetch(AUTOTUNE))
val_data   = (val_data
              .cache()
              .map(norm, num_parallel_calls=AUTOTUNE)
              .prefetch(AUTOTUNE))

# -------------------- Model (no BN, L2 lieve) --------------------
l2 = regularizers.l2(1e-4)
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),

    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2),
    layers.MaxPooling2D(2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['sparse_categorical_accuracy']
)

early_stop = EarlyStopping(monitor='val_sparse_categorical_accuracy',
                           mode='max', patience=6, restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
                               mode='max', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
ckpt_path  = MODEL_DIR / "best_model.keras"
checkpoint = ModelCheckpoint(ckpt_path, monitor='val_sparse_categorical_accuracy',
                             mode='max', save_best_only=True, verbose=1)

# -------------------- Train --------------------
history = model.fit(
    train_data, validation_data=val_data,
    epochs=epochs, callbacks=[early_stop, reduce_lr, checkpoint]
)

# -------------------- Plot & Save --------------------
plt.plot(history.history['sparse_categorical_accuracy'],     label='Train Acc')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Acc')
plt.title("CNN_2 - Accuracy (conservative)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(PLOTS_DIR / "accuracy.png", dpi=150); plt.close()

model_path = MODEL_DIR / "predictor_CNN_2.keras"
model.save(model_path)
with open(RUN_DIR / "history.json", "w") as f:
    json.dump(history.history, f, indent=2)

# -------------------- Wrong predictions --------------------
rows, counter = [], 1
for image, label in val_data.unbatch():
    img_array = image.numpy()
    true_idx  = int(label.numpy())
    probs     = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
    pred_idx  = int(np.argmax(probs))
    if pred_idx != true_idx:
        filename = f"wrong_{counter:03d}.jpg"
        path     = WRONG_IMG_DIR / filename
        img_uint8 = (img_array * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(path)
        rows.append([filename, CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx], float(probs[pred_idx])])
        counter += 1

csv_path = WRONG_DIR / "wrong_preds.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "predicted_label", "confidence"])
    writer.writerows(rows)

print(f"Saved best model to: {ckpt_path}")
print(f"Saved final model to: {model_path}")
print(f"Wrong predictions: {len(rows)}\n - images: {WRONG_IMG_DIR}\n - CSV: {csv_path}")
print(f"Run dir: {RUN_DIR}")
