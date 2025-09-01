# CNN3.py
# Clean, robust training script for CNN_3 with correct batch sweep, stable pipelines, and improved model.

import os
import math
import csv
import shutil
import datetime
from collections import Counter

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

# --- Try to use AdamW; fall back to Adam if unavailable ---
try:
    from tensorflow.keras.optimizers import AdamW
    OPTIMIZER_CLS = AdamW
    OPT_KW = dict(weight_decay=1e-4)
except Exception:
    from tensorflow.keras.optimizers import Adam as AdamW  # fallback alias
    OPTIMIZER_CLS = AdamW
    OPT_KW = {}

# -------------------- Config --------------------
SEED = 42
IMG_HEIGHT, IMG_WIDTH = 200, 300
NUM_CLASSES = 3
VAL_SPLIT = 0.2
EPOCHS = 20
BATCH_OPTIONS = [16, 32, 64]
INIT_LR = 3e-4

# Project-relative paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data_clean")   # must contain subfolders: paper/rock/scissors
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = os.path.join(THIS_DIR, "outputs", TIMESTAMP)
MODEL_DIR = os.path.join(OUT_DIR, "models")
FIG_DIR = os.path.join(OUT_DIR, "figures")
WRONG_DIR = os.path.join(OUT_DIR, "wrong_predictions")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(WRONG_DIR, exist_ok=True)

# -------------------- Utils --------------------
def set_seeds(seed=SEED):
    tf.keras.utils.set_random_seed(seed)

def count_images_per_class(base_dir):
    """Count images per class folder (non-recursive)."""
    counts = {}
    total = 0
    for cls in sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]):
        cls_dir = os.path.join(base_dir, cls)
        n = sum(
            1 for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )
        counts[cls] = n
        total += n
    return counts, total

def compute_class_weights(class_names, counts_dict):
    """Return class_weight mapping {index: weight} based on inverse frequency."""
    total = sum(counts_dict.get(c, 0) for c in class_names)
    if total == 0:
        return None
    weights = {}
    for i, c in enumerate(class_names):
        cnt = counts_dict.get(c, 0)
        weights[i] = (total / (len(class_names) * max(cnt, 1)))
    return weights

def plot_and_save_history(history, out_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.title("CNN_3 - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------- Data --------------------
AUTOTUNE = tf.data.AUTOTUNE

# Moderated, realistic augmentation (keeps hands interpretable)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),            # ~±15°
    layers.RandomTranslation(0.05, 0.05),   # small shifts
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

def make_train_val_datasets(base_dir, img_size, batch, seed=SEED, val_split=VAL_SPLIT, augment=True, cache=True):
    """Create train/val datasets with stable preprocessing and class_names kept."""
    raw_train = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch,
        validation_split=val_split,
        subset='training',
        seed=seed,
        shuffle=True
    )
    class_names = list(raw_train.class_names)  # capture before mapping

    raw_val = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch,
        validation_split=val_split,
        subset='validation',
        seed=seed,
        shuffle=False
    )

    if augment:
        train = raw_train.map(lambda x, y: (data_augmentation(x, training=True) / 255.0, y),
                              num_parallel_calls=AUTOTUNE)
    else:
        train = raw_train.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)

    val = raw_val.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)

    if cache:
        train = train.cache()
        val = val.cache()

    train = train.prefetch(AUTOTUNE)
    val = val.prefetch(AUTOTUNE)
    return train, val, class_names

# -------------------- Model --------------------
def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    # Conv blocks with BatchNorm and ReLU
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Global pooling for robustness
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="cnn3_model")
    return model

def compile_model(model, lr=INIT_LR):
    opt = OPTIMIZER_CLS(learning_rate=lr, **OPT_KW)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model

# -------------------- Batch sweep --------------------
def batch_sweep(base_dir, img_size, batch_options, steps=40, seed=SEED):
    """Choose the best batch size using a repeated small training loop and full validation."""
    print("\n[Batch sweep] Starting...")
    counts, total = count_images_per_class(base_dir)
    est_train = int(total * (1.0 - VAL_SPLIT))

    best_batch, best_val_acc = None, -1.0
    for b in batch_options:
        print(f"\n[Batch sweep] Testing batch size: {b}")
        train_ds, val_ds, class_names = make_train_val_datasets(base_dir, img_size, b, seed=seed, augment=True)
        # Repeat the train_ds so we never run out; cap steps per epoch
        train_rep = train_ds.repeat()
        # Build/compile a fresh lightweight model for fairness
        model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=len(class_names))
        compile_model(model, lr=3e-4)

        ckpt_tmp = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"_sweep_tmp_bs{b}.keras"),
            monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=0
        )
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True, verbose=0)

        hist = model.fit(
            train_rep,
            steps_per_epoch=steps,           # fixed small number of steps
            validation_data=val_ds,          # full validation each epoch
            epochs=3,
            callbacks=[ckpt_tmp, es],
            verbose=1
        )
        val_acc = max(hist.history.get('val_accuracy', [0.0]))
        print(f"→ Batch {b}: best val_acc = {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_batch = b

        # Cleanup tmp model file if created
        try:
            os.remove(os.path.join(MODEL_DIR, f"_sweep_tmp_bs{b}.keras"))
        except Exception:
            pass

    print(f"\n🏆 Selected batch size: {best_batch}")
    return best_batch

# -------------------- Wrong predictions --------------------
def collect_wrong_predictions(model, val_ds, class_names, out_dir):
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    rows = []
    counter = 1

    # val_ds is already normalized [0,1]
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true = np.argmax(batch_y.numpy(), axis=1)
        y_pred = np.argmax(preds, axis=1)

        mism = np.where(y_true != y_pred)[0]
        for idx in mism:
            img = (batch_x[idx].numpy() * 255.0).astype(np.uint8)
            fname = f"wrong_{counter:04d}.jpg"
            Image.fromarray(img).save(os.path.join(images_dir, fname))
            rows.append([fname, class_names[y_true[idx]], class_names[y_pred[idx]]])
            counter += 1

    with open(os.path.join(out_dir, "wrong_preds.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "predicted_label"])
        writer.writerows(rows)

# -------------------- Main --------------------
def main():
    print("==> CNN_3 Training")
    print(f"Using data from: {DATA_DIR}")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Input folder not found: {DATA_DIR}\n"
            f"Expected structure: data_clean/{{paper,rock,scissors}}"
        )

    set_seeds(SEED)

    # Quick dataset stats & class weights
    counts, total = count_images_per_class(DATA_DIR)
    print(f"Found images per class: {counts} (total={total})")

    # Batch sweep (robust)
    best_batch = batch_sweep(DATA_DIR, (IMG_HEIGHT, IMG_WIDTH), BATCH_OPTIONS, steps=40, seed=SEED)

    # Final datasets with best batch
    train_ds, val_ds, class_names = make_train_val_datasets(
        DATA_DIR, (IMG_HEIGHT, IMG_WIDTH), best_batch, seed=SEED, augment=True
    )
    print(f"Detected classes: {class_names}")

    # Compute class weights aligned to class_names
    class_weights = compute_class_weights(class_names, counts)
    if class_weights:
        print(f"Using class weights: {class_weights}")

    # Build & compile final model
    model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=len(class_names))
    compile_model(model, lr=INIT_LR)

    # Callbacks
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.keras"),
        monitor='val_accuracy', mode='max',
        save_best_only=True, verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True, verbose=1
    )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt_best, es, rlrop],
        class_weight=class_weights
    )

    # Save final model and plot
    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model.save(final_model_path)
    plot_path = os.path.join(FIG_DIR, "training_accuracy.png")
    plot_and_save_history(history, plot_path)

    print(f"\n✅ Models saved to: {MODEL_DIR}")
    print("Collecting wrong predictions on validation set...")
    collect_wrong_predictions(model, val_ds, class_names, WRONG_DIR)
    print(f"📂 Wrong predictions saved to: {WRONG_DIR}")
    print(f"📊 Plot saved to: {plot_path}")
    print("✅ Training completed.")

if __name__ == "__main__":
    main()
