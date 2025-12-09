"""
TensorFlow Multi-Task SewerAnalytics MobileNet Training Script
--------------------------------------------------------------

✔ No PyTorch
✔ Pure TensorFlow/Keras
✔ MobileNetV2 or MobileNetV3 backbone (frozen)
✔ Multi-head:
      - Classification (softmax)
      - Regression scalars (sigmoid)
✔ Joint loss = CE + MSE
✔ Automatic TF Lite export
"""

# ============================================================
# Imports
# ============================================================
import os
import json
import math
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile

import tensorflow as tf
from tensorflow.keras import layers, models
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# ============================================================
# Config
# ============================================================
BASE_DIR = r"D:\datasets\SAV"

TRAIN_CSV = os.path.join(BASE_DIR, "SAV_lvl2_train.csv")
VAL_CSV   = os.path.join(BASE_DIR, "SAV_lvl2_val.csv")
TEST_CSV  = os.path.join(BASE_DIR, "SAV_lvl2_test.csv")

EXPERIMENT_NAME = "TF_SAV_MobileNet_MTL"
CHECKPOINT_DIR = f"checkpoints_{EXPERIMENT_NAME}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 1
LR = 3e-4
TARGET_SIZE = 224
BACKBONE_NAME = "mobilenet_v2"   # or "mobilenet_v3"

SEED = 1337


# ============================================================
# Setup
# ============================================================
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# CSV Decode
# ============================================================
def decode_sa_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "image_path" not in df.columns:
        raise AssertionError("CSV must contain 'image_path'")
    df["filepath"] = df["image_path"].astype(str).str.strip()

    cls_cols = [c for c in df.columns if c.startswith("cls_")]
    scalar_cols = [c for c in df.columns if c.startswith("scalar_")]

    df[cls_cols] = df[cls_cols].fillna(0).astype(int)
    df = df[df[cls_cols].sum(axis=1) > 0].reset_index()

    df["code"] = df[cls_cols].idxmax(axis=1).str.replace("cls_", "")
    df[scalar_cols] = df[scalar_cols].astype(float)

    out = df[["code", "filepath"]].copy()
    for s in scalar_cols:
        out[s] = df[s]

    return out


def sanitize_df(df):
    df = df.copy()
    df = df.dropna(subset=["filepath", "code"])
    df["filepath"] = df["filepath"].astype(str)
    df["code"] = df["code"].astype(str)
    return df


# ============================================================
# TF Dataset Loader
# ============================================================
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (TARGET_SIZE, TARGET_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_tf_dataset(df, class_to_idx, scalar_cols, shuffle=False):
    paths = df["filepath"].tolist()
    class_labels = [class_to_idx[c] for c in df["code"]]
    reg_targets = df[scalar_cols].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, class_labels, reg_targets))

    def mapper(path, y_cls, y_reg):
        img = load_image(path)
        return img, {"cls": y_cls, "reg": y_reg}

    if shuffle:
        ds = ds.shuffle(4096)
    return ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)\
             .batch(BATCH_SIZE)\
             .prefetch(tf.data.AUTOTUNE)


# ============================================================
# Build MobileNet Model
# ============================================================
def build_model(num_classes, num_scalars):
    if BACKBONE_NAME == "mobilenet_v2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )
    else:
        base = tf.keras.applications.MobileNetV3Large(
            input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )

    base.trainable = False  # freeze backbone

    inputs = layers.Input((TARGET_SIZE, TARGET_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Shared MLP
    h = layers.Dense(1024, activation=None)(x)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Dropout(0.2)(h)

    h = layers.Dense(1024, activation=None)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)

    # Heads
    cls_out = layers.Dense(num_classes, activation="softmax", name="cls")(h)
    reg_out = layers.Dense(num_scalars, activation="sigmoid", name="reg")(h)

    model = models.Model(inputs, {"cls": cls_out, "reg": reg_out})
    return model


# ============================================================
# Main
# ============================================================
def main():

    # Load CSVs
    df_tr = sanitize_df(decode_sa_csv(TRAIN_CSV))
    df_val = sanitize_df(decode_sa_csv(VAL_CSV))
    df_te = sanitize_df(decode_sa_csv(TEST_CSV))

    # Class mapping
    classes = sorted(df_tr["code"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    scalar_cols = [c for c in df_tr.columns if c.startswith("scalar_")]

    # Save class list
    with open(os.path.join(CHECKPOINT_DIR, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    # TF datasets
    train_ds = make_tf_dataset(df_tr, class_to_idx, scalar_cols, shuffle=True)
    val_ds   = make_tf_dataset(df_val, class_to_idx, scalar_cols)
    test_ds  = make_tf_dataset(df_te, class_to_idx, scalar_cols)

    # Model
    model = build_model(num_classes=len(classes), num_scalars=len(scalar_cols))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss={
            "cls": "sparse_categorical_crossentropy",
            "reg": "mse"
        },
        metrics={
            "cls": ["accuracy"]
        }
    )

    # # Train
    # model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=EPOCHS,
    # )

    # Save best model
    # model.save(os.path.join(CHECKPOINT_DIR, "model.keras"))

    # ============================================================
    # Export TFLite Model
    # ============================================================
    trained_model = tf.keras.models.load_model("checkpoints_TF_SAV_MobileNet_MTL\model.keras") #os.path.join(CHECKPOINT_DIR, "model.keras"))
    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model) #os.path.join(CHECKPOINT_DIR, "model.keras"))
    tflite_model = converter.convert()

    with open(os.path.join(CHECKPOINT_DIR, "model.tflite"), "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved to:", os.path.join(CHECKPOINT_DIR, "model.tflite"))


if __name__ == "__main__":
    main()
