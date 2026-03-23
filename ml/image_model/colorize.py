"""
Train a 3-level U-Net colorizer: grayscale → RGB.

Improvements over v1:
- Deeper U-Net: 3 encoder/decoder levels (64→128→256→512→256→128→64)
- Combined loss: MAE + 0.5*(1 - SSIM) for perceptual quality
- SSIM reported as metric alongside MAE
- tf.data pipeline with on-GPU augmentation (flip, brightness, contrast)
- ModelCheckpoint saves best weights during training
- EarlyStopping patience increased to 10

Input:  grayscale image  (H, W, 1)
Output: RGB image        (H, W, 3)

Usage:
    python ml/image_model/colorize.py
    python ml/image_model/colorize.py --dataset oxford_iiit_pet --epochs 40
    docker compose --profile train up train-colorizer
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ml.runtime.gpu import describe_tensorflow_runtime, setup_tensorflow_runtime
from ml.runtime.progress import KerasProgressCallback, write_progress_snapshot

os.makedirs("ml/image_model/models", exist_ok=True)

DATA_ROOT = Path("ml/datasets")
KERAS_CACHE_DIR = DATA_ROOT / "keras"
TFDS_DIR = DATA_ROOT / "tfds"
KERAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TFDS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(KERAS_CACHE_DIR.resolve()))

MODEL_PATH = Path("ml/image_model/models/colorizer_oxford_iiit_pet.h5")
SAMPLES_PATH = "ml/image_model/models/colorization_samples.png"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
IMG_SIZE = (128, 128)
LR = 1e-3


# ── Loss & metrics ────────────────────────────────────────────────────────────

def _ssim_metric(y_true, y_pred):
    """Structural Similarity Index — perceptual quality measure (higher = better)."""
    import tensorflow as tf
    return tf.reduce_mean(tf.image.ssim(
        tf.cast(y_true, tf.float32),
        tf.cast(y_pred, tf.float32),
        max_val=1.0,
    ))


def _combined_loss(y_true, y_pred):
    """MAE + 0.5*(1 - SSIM): pixel accuracy + structural similarity."""
    import tensorflow as tf
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true_f - y_pred_f))
    ssim = tf.reduce_mean(tf.image.ssim(y_true_f, y_pred_f, max_val=1.0))
    return mae + (1.0 - ssim) * 0.5


# ── Model ─────────────────────────────────────────────────────────────────────

def build_colorizer():
    """3-level U-Net: 64→128→256→512(bottleneck)→256→128→64 → RGB."""
    import tensorflow as tf

    inp = tf.keras.Input(shape=(*IMG_SIZE, 1))

    # Encoder
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    skip1 = x
    x = tf.keras.layers.MaxPooling2D()(x)  # 64×64

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip2 = x
    x = tf.keras.layers.MaxPooling2D()(x)  # 32×32

    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip3 = x
    x = tf.keras.layers.MaxPooling2D()(x)  # 16×16

    # Bottleneck
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D()(x)  # 32×32
    x = tf.keras.layers.Concatenate()([x, skip3])
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.UpSampling2D()(x)  # 64×64
    x = tf.keras.layers.Concatenate()([x, skip2])
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.UpSampling2D()(x)  # 128×128
    x = tf.keras.layers.Concatenate()([x, skip1])
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    out = tf.keras.layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=_combined_loss,
        metrics=["mae", _ssim_metric],
    )
    return model


# ── Data ──────────────────────────────────────────────────────────────────────

def default_model_path(dataset_name: str) -> Path:
    return Path(f"ml/image_model/models/colorizer_{dataset_name}.h5")


def load_data(dataset_name: str, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray_images, color_images) normalised to [0, 1]."""
    import tensorflow as tf
    raw = None

    if dataset_name == "oxford_iiit_pet":
        import tensorflow_datasets as tfds

        print("Loading Oxford-IIIT Pet dataset…")
        ds = tfds.load(
            "oxford_iiit_pet",
            split="train+test",
            as_supervised=False,
            data_dir=str(TFDS_DIR.resolve()),
        )
        raw = [r["image"].numpy() for r in ds.take(limit)]
    elif dataset_name == "stl10":
        import tensorflow_datasets as tfds

        print("Trying STL-10 dataset…")
        ds = tfds.load(
            "stl10",
            split="unlabelled",
            as_supervised=False,
            data_dir=str(TFDS_DIR.resolve()),
        )
        raw = [r["image"].numpy() for r in ds.take(limit)]

    if raw is None:
        print("STL-10 not available — falling back to CIFAR-10 (resized).")
        (x, _), _ = tf.keras.datasets.cifar10.load_data()
        raw = list(x[:limit])

    def resize_img(arr: np.ndarray) -> np.ndarray:
        return np.array(
            Image.fromarray(arr.astype(np.uint8)).resize(IMG_SIZE),
            dtype=np.float32,
        ) / 255.0

    color = np.stack([resize_img(img) for img in raw])
    gray = np.mean(color, axis=-1, keepdims=True)
    return gray, color


def make_dataset(x_gray: np.ndarray, x_color: np.ndarray, batch_size: int, augment: bool = False):
    """Build a tf.data pipeline; optionally apply consistent augmentation."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x_gray, x_color))

    if augment:
        def augment_fn(gray, color):
            # Consistent horizontal flip for both input and target
            combined = tf.concat([gray, color], axis=-1)  # (H, W, 4)
            combined = tf.image.random_flip_left_right(combined)
            gray = combined[..., :1]
            color = combined[..., 1:]
            # Brightness and contrast only on the color image
            color = tf.image.random_brightness(color, 0.15)
            color = tf.image.random_contrast(color, 0.85, 1.15)
            color = tf.clip_by_value(color, 0.0, 1.0)
            # Recompute grayscale from the augmented color so they stay in sync
            gray = tf.reduce_mean(color, axis=-1, keepdims=True)
            return gray, color

        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_samples(model, x_gray: np.ndarray, x_color: np.ndarray, n: int = 8) -> None:
    preds = model.predict(x_gray[:n], verbose=0).astype(np.float32)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        axes[i, 0].imshow(x_gray[i, :, :, 0], cmap="gray")
        axes[i, 0].set_title("Grayscale")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(np.clip(preds[i], 0, 1))
        axes[i, 1].set_title("Predicted")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(x_color[i])
        axes[i, 2].set_title("Original")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(SAMPLES_PATH, dpi=100)
    plt.close()
    print(f"Saved samples → {SAMPLES_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import tensorflow as tf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["stl10", "oxford_iiit_pet", "cifar10"],
        default="stl10",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--mixed-precision", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()

    runtime_info = setup_tensorflow_runtime(
        enable_mixed_precision=args.mixed_precision != "off"
    )
    if args.mixed_precision == "off":
        runtime_info["mixed_precision"] = False
    print(describe_tensorflow_runtime(runtime_info))
    write_progress_snapshot(
        {
            "status": "initializing",
            "stage": f"colorize:{args.dataset}",
            "epoch": 0,
            "total_epochs": args.epochs,
            "elapsed_seconds": 0.0,
            "epoch_seconds": None,
            "eta_seconds": None,
            "metrics": {},
            "runtime": describe_tensorflow_runtime(runtime_info),
        }
    )

    model_path = args.model_path or default_model_path(args.dataset)
    checkpoint_path = str(model_path).replace(".h5", "_best.h5")

    print("Loading data…")
    x_gray, x_color = load_data(args.dataset, args.limit)

    split = int(len(x_gray) * 0.85)
    x_g_tr, x_g_val = x_gray[:split], x_gray[split:]
    x_c_tr, x_c_val = x_color[:split], x_color[split:]
    print(f"Train: {len(x_g_tr)} | Val: {len(x_g_val)}")

    train_ds = make_dataset(x_g_tr, x_c_tr, args.batch_size, augment=True)
    val_ds = make_dataset(x_g_val, x_c_val, args.batch_size, augment=False)

    model = build_colorizer()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_mae"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, monitor="val_mae"),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor="val_mae", verbose=1,
        ),
        KerasProgressCallback(total_epochs=args.epochs, stage=f"colorize:{args.dataset}"),
    ]

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    model.save(model_path)
    print(f"Colorizer saved → {model_path}")
    save_samples(model, x_g_val[:8], x_c_val[:8])


if __name__ == "__main__":
    main()
