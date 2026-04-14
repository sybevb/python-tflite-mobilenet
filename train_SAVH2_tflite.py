#!/usr/bin/env python3
"""
Train a small TensorFlow SAVH2 model and export a TensorFlow 2 TFLite model.

The label loading follows the SAVH2 label_tool CSV layout used by the PyTorch
DINO trainer:

    D:\\datasets\\SAV_extracted_hash\\
        labeled_train.csv
        labeled_val.csv
        labeled_test.csv
        ground_truth_val.csv
        label_tool_state\\labels.json
        train\\*.jpg
        val\\*.jpg
        test\\*.jpg

The exported TFLite model has one flat output:
    [class probabilities..., scalar values..., ordinal probabilities...]

Classes are multi-label sigmoid outputs. Scalars are sigmoid values in [0, 1].
Each ordinal target is exported as one softmax probability block.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


DATA_ROOT = r"D:\datasets\SAV_extracted_hash"
DEFAULT_OUT_DIR = os.path.join("SAV_detection", "models_savh2")
SEED = 1337


def safe_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def read_label_tool_labels(labels_json: str) -> Tuple[List[str], List[str], List[dict]]:
    if not os.path.exists(labels_json):
        print(f"[warn] Missing labels.json: {labels_json}")
        return [], [], []

    with open(labels_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = [
        str(item.get("name", "")).strip()
        for item in payload.get("classes", [])
        if item.get("name")
    ]
    scalars = [
        str(item.get("name", "")).strip()
        for item in payload.get("scalars", [])
        if item.get("name")
    ]

    ordinals = []
    for item in payload.get("ordinals", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            bins = int(item.get("bins", 0))
        except (TypeError, ValueError):
            print(f"[warn] Skipping ordinal {name!r}: invalid bin count")
            continue
        if bins <= 0:
            print(f"[warn] Skipping ordinal {name!r}: non-positive bin count")
            continue
        ordinals.append({"name": name, "bins": bins})

    return classes, scalars, ordinals


def load_index_csv(index_csv: str) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    if "image_path" not in df.columns:
        raise ValueError(f"Missing image_path column in {index_csv}")
    df = df.copy()
    df["filename"] = df["image_path"].astype(str).map(lambda p: Path(p).name)
    return df.drop_duplicates("filename")[["filename", "image_path"]]


def coerce_label(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def coerce_scalar(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)


def coerce_ordinal(series: pd.Series, bins: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).round().astype(int)
    return values.clip(0, bins - 1)


def load_label_tool_csv(
    labeled_csv: str,
    index_csv: Optional[str],
    class_names: Sequence[str],
    scalar_names: Sequence[str],
    ordinal_defs: Sequence[dict],
    *,
    split_dir: Optional[str],
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[int]]:
    df_labels = pd.read_csv(labeled_csv)
    if "filename" not in df_labels.columns:
        if "image_path" not in df_labels.columns:
            raise ValueError(f"Missing filename or image_path column in {labeled_csv}")
        df_labels = df_labels.copy()
        df_labels["filename"] = df_labels["image_path"].astype(str).map(
            lambda p: Path(p).name
        )

    df = df_labels.copy()
    if split_dir:
        df["image_path"] = df["filename"].map(lambda n: os.path.join(split_dir, str(n)))
    elif index_csv:
        df = df.merge(load_index_csv(index_csv), on="filename", how="left")
    else:
        raise ValueError("Either split_dir or index_csv is required.")

    safe_classes = [safe_name(name) for name in class_names]
    safe_scalars = [safe_name(name) for name in scalar_names]
    safe_ordinals = [safe_name(item["name"]) for item in ordinal_defs]
    ordinal_bins_by_col = {
        safe_name(item["name"]): int(item["bins"]) for item in ordinal_defs
    }

    for name in safe_classes:
        prefixed = f"class_{name}"
        if prefixed in df.columns and name not in df.columns:
            df[name] = df[prefixed]
    for name in safe_scalars:
        prefixed = f"scalar_{name}"
        if prefixed in df.columns and name not in df.columns:
            df[name] = df[prefixed]
    for name in safe_ordinals:
        prefixed = f"ordinal_{name}"
        if prefixed in df.columns and name not in df.columns:
            df[name] = df[prefixed]

    label_cols = [col for col in safe_classes if col in df.columns]
    scalar_cols = [col for col in safe_scalars if col in df.columns]
    ordinal_cols = [col for col in safe_ordinals if col in df.columns]
    ordinal_bins = [ordinal_bins_by_col[col] for col in ordinal_cols]

    if not label_cols:
        drop_cols = {"filename", "image_path"}
        drop_cols.update(c for c in df.columns if c.startswith("cluster_id_lvl"))
        drop_cols.update(c for c in df.columns if c.startswith("scalar_"))
        drop_cols.update(c for c in df.columns if c.startswith("ordinal_"))
        label_cols = [c for c in df.columns if c not in drop_cols]
        print(f"[warn] Falling back to label columns from CSV: {label_cols}")

    if not scalar_cols:
        drop_cols = {"filename", "image_path"}
        drop_cols.update(label_cols)
        drop_cols.update(ordinal_cols)
        drop_cols.update(c for c in df.columns if c.startswith("cluster_id_lvl"))
        scalar_cols = [
            c for c in df.columns if c.startswith("scalar_") and c not in drop_cols
        ]

    df = df.copy()
    for col in label_cols:
        df[col] = coerce_label(df[col])
    for col in scalar_cols:
        df[col] = coerce_scalar(df[col])
    for col, bins in zip(ordinal_cols, ordinal_bins):
        df[col] = coerce_ordinal(df[col], bins)

    df["filepath"] = df["image_path"].astype(str)
    before = len(df)
    df = df[df["filepath"].map(os.path.exists)].reset_index(drop=True)
    if len(df) != before:
        print(f"[warn] Dropped {before - len(df)} rows with missing image files.")
    return df, label_cols, scalar_cols, ordinal_cols, ordinal_bins


def align_columns(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    scalar_cols: Sequence[str],
    ordinal_cols: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in label_cols:
        if col not in df.columns:
            df[col] = 0
    for col in scalar_cols:
        if col not in df.columns:
            df[col] = 0.0
    for col in ordinal_cols:
        if col not in df.columns:
            df[col] = 0
    return df


def load_image(path: tf.Tensor, target_size: int, augment: bool) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [target_size, target_size], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.08)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def make_dataset(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    scalar_cols: Sequence[str],
    ordinal_cols: Sequence[str],
    *,
    target_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
) -> tf.data.Dataset:
    paths = df["filepath"].astype(str).to_numpy()
    labels = df[list(label_cols)].astype(np.float32).to_numpy()
    tensors = [paths, labels]
    output_names = ["classes"]

    if scalar_cols:
        scalars = df[list(scalar_cols)].astype(np.float32).to_numpy()
        tensors.append(scalars)
        output_names.append("scalars")

    for col in ordinal_cols:
        tensors.append(df[col].astype(np.int32).to_numpy())
        output_names.append(f"ordinal_{col}")

    ds = tf.data.Dataset.from_tensor_slices(tuple(tensors))
    if shuffle:
        ds = ds.shuffle(min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True)

    def mapper(*items):
        path = items[0]
        img = load_image(path, target_size, augment)
        targets = {name: value for name, value in zip(output_names, items[1:])}
        return img, targets

    return (
        ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def weighted_binary_crossentropy(pos_weights: np.ndarray):
    weights = tf.constant(pos_weights.astype(np.float32))

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pos = -y_true * tf.math.log(y_pred) * weights
        neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(pos + neg)

    return loss


def build_train_model(
    *,
    input_size: int,
    num_classes: int,
    num_scalars: int,
    ordinal_cols: Sequence[str],
    ordinal_bins: Sequence[int],
    alpha: float,
    dropout: float,
    backbone_weights: Optional[str],
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input((input_size, input_size, 3), name="image")
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="to_mobilenet_range")(inputs)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights=backbone_weights,
        alpha=alpha,
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = tf.keras.layers.Dense(256, use_bias=False, name="pred_dense_0")(x)
    x = tf.keras.layers.BatchNormalization(name="pred_bn_0")(x)
    x = tf.keras.layers.ReLU(name="pred_relu_0")(x)
    x = tf.keras.layers.Dropout(dropout, name="pred_dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="pred_dense_1")(x)

    outputs = [
        tf.keras.layers.Dense(num_classes, activation="sigmoid", name="classes")(x)
    ]
    if num_scalars > 0:
        outputs.append(
            tf.keras.layers.Dense(num_scalars, activation="sigmoid", name="scalars")(x)
        )
    for col, bins in zip(ordinal_cols, ordinal_bins):
        outputs.append(
            tf.keras.layers.Dense(int(bins), activation="softmax", name=f"ordinal_{col}")(x)
        )
    return tf.keras.Model(inputs, outputs, name="savh2_mobilenetv2_035")


def build_export_model(train_model: tf.keras.Model) -> tf.keras.Model:
    outputs = train_model(train_model.input, training=False)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if len(outputs) == 1:
        flat = tf.keras.layers.Activation("linear", name="savh2_output")(outputs[0])
    else:
        flat = tf.keras.layers.Concatenate(name="savh2_output")(outputs)
    return tf.keras.Model(train_model.input, flat, name="savh2_tflite_export")


def make_metadata(
    *,
    label_cols: Sequence[str],
    scalar_cols: Sequence[str],
    ordinal_cols: Sequence[str],
    ordinal_bins: Sequence[int],
    input_size: int,
    threshold: float,
) -> Dict[str, object]:
    offset = 0
    offsets: Dict[str, object] = {}
    offsets["classes"] = [offset, offset + len(label_cols)]
    offset += len(label_cols)
    offsets["scalars"] = [offset, offset + len(scalar_cols)]
    offset += len(scalar_cols)
    ordinal_offsets = {}
    for col, bins in zip(ordinal_cols, ordinal_bins):
        ordinal_offsets[col] = [offset, offset + int(bins)]
        offset += int(bins)
    offsets["ordinals"] = ordinal_offsets

    return {
        "format": "savh2_tflite_v1",
        "classes": list(label_cols),
        "scalars": list(scalar_cols),
        "ordinals": list(ordinal_cols),
        "ordinal_bins": {
            col: int(bins) for col, bins in zip(ordinal_cols, ordinal_bins)
        },
        "input_size": int(input_size),
        "threshold": float(threshold),
        "flat_output_size": int(offset),
        "offsets": offsets,
        "output_order": ["classes", "scalars"]
        + [f"ordinal_{col}" for col in ordinal_cols],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and export a SAVH2 TFLite model.")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--mobilenet-alpha", type=float, default=0.35)
    p.add_argument("--backbone-weights", choices=("imagenet", "none"), default="imagenet")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fine-tune-epochs", type=int, default=0)
    p.add_argument("--fine-tune-lr", type=float, default=3e-5)
    p.add_argument("--fine-tune-last-layers", type=int, default=30)
    p.add_argument("--max-pos-weight", type=float, default=20.0)
    p.add_argument("--no-pos-weight", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--use-index-csv", action="store_true")
    p.add_argument("--quantize-dynamic-range", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_json = os.path.join(args.data_root, "label_tool_state", "labels.json")
    class_names, scalar_names, ordinal_defs = read_label_tool_labels(labels_json)

    def csv_path(name: str) -> str:
        return os.path.join(args.data_root, name)

    use_split_dirs = not args.use_index_csv
    train_split = os.path.join(args.data_root, "train") if use_split_dirs else None
    val_split = os.path.join(args.data_root, "val") if use_split_dirs else None
    test_split = os.path.join(args.data_root, "test") if use_split_dirs else None

    df_train, label_cols, scalar_cols, ordinal_cols, ordinal_bins = load_label_tool_csv(
        csv_path("labeled_train.csv"),
        None if use_split_dirs else csv_path("dataset_index_train.csv"),
        class_names,
        scalar_names,
        ordinal_defs,
        split_dir=train_split,
    )
    if not label_cols:
        raise ValueError("No class label columns were found.")
    if df_train.empty:
        raise ValueError("Training dataframe is empty after loading image paths.")

    val_csv = csv_path("labeled_val.csv")
    if os.path.exists(val_csv):
        df_val, _, _, _, _ = load_label_tool_csv(
            val_csv,
            None if use_split_dirs else csv_path("dataset_index_val.csv"),
            class_names,
            scalar_names,
            ordinal_defs,
            split_dir=val_split,
        )
        df_val = align_columns(df_val, label_cols, scalar_cols, ordinal_cols)
    else:
        df_val = pd.DataFrame()
        print("[warn] No labeled_val.csv found; training without validation.")

    print(f"Train samples: {len(df_train):,}")
    print(f"Val samples:   {len(df_val):,}")
    print(f"Classes: {len(label_cols)} -> {label_cols}")
    print(f"Scalars: {len(scalar_cols)} -> {scalar_cols}")
    print(f"Ordinals: {len(ordinal_cols)} -> {ordinal_cols}")

    train_ds = make_dataset(
        df_train,
        label_cols,
        scalar_cols,
        ordinal_cols,
        target_size=args.input_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=not args.no_augment,
    )
    val_ds = None
    if not df_val.empty:
        val_ds = make_dataset(
            df_val,
            label_cols,
            scalar_cols,
            ordinal_cols,
            target_size=args.input_size,
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
        )

    model = build_train_model(
        input_size=args.input_size,
        num_classes=len(label_cols),
        num_scalars=len(scalar_cols),
        ordinal_cols=ordinal_cols,
        ordinal_bins=ordinal_bins,
        alpha=args.mobilenet_alpha,
        dropout=args.dropout,
        backbone_weights=None if args.backbone_weights == "none" else args.backbone_weights,
    )

    y_train = df_train[label_cols].astype(np.float32).to_numpy()
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pos_weights = np.clip(neg / np.maximum(pos, 1.0), 1.0, args.max_pos_weight)
    class_loss = (
        "binary_crossentropy"
        if args.no_pos_weight
        else weighted_binary_crossentropy(pos_weights)
    )

    losses: Dict[str, object] = {"classes": class_loss}
    metrics: Dict[str, list] = {"classes": [tf.keras.metrics.BinaryAccuracy(name="bin_acc")]}
    if scalar_cols:
        losses["scalars"] = "mse"
        metrics["scalars"] = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    for col in ordinal_cols:
        losses[f"ordinal_{col}"] = "sparse_categorical_crossentropy"
        metrics[f"ordinal_{col}"] = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=losses,
        metrics=metrics,
    )

    monitor = "val_classes_bin_acc" if val_ds is not None else "classes_bin_acc"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "epoch_{epoch:03d}.keras"),
            save_best_only=False,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_curve.csv")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        base = next((layer for layer in model.layers if "mobilenet" in layer.name), None)
        if base is not None:
            base.trainable = True
            for layer in base.layers[:-args.fine_tune_last_layers]:
                layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(args.fine_tune_lr),
                loss=losses,
                metrics=metrics,
            )
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                callbacks=callbacks,
            )

    keras_path = out_dir / "model.keras"
    model.save(keras_path, include_optimizer=False)

    metadata = make_metadata(
        label_cols=label_cols,
        scalar_cols=scalar_cols,
        ordinal_cols=ordinal_cols,
        ordinal_bins=ordinal_bins,
        input_size=args.input_size,
        threshold=args.threshold,
    )
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    export_model = build_export_model(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    if args.quantize_dynamic_range:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = out_dir / "model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    if not args.skip_test:
        test_csv = csv_path("labeled_test.csv")
        if os.path.exists(test_csv):
            df_test, _, _, _, _ = load_label_tool_csv(
                test_csv,
                None if use_split_dirs else csv_path("dataset_index_test.csv"),
                class_names,
                scalar_names,
                ordinal_defs,
                split_dir=test_split,
            )
            df_test = align_columns(df_test, label_cols, scalar_cols, ordinal_cols)
            if not df_test.empty:
                test_ds = make_dataset(
                    df_test,
                    label_cols,
                    scalar_cols,
                    ordinal_cols,
                    target_size=args.input_size,
                    batch_size=args.batch_size,
                    shuffle=False,
                    augment=False,
                )
                print("Test metrics:")
                model.evaluate(test_ds, verbose=2, return_dict=True)

    print(f"Saved Keras model:  {keras_path}")
    print(f"Saved TFLite model: {tflite_path}")
    print(f"Saved metadata:     {out_dir / 'classes.json'}")


if __name__ == "__main__":
    main()
