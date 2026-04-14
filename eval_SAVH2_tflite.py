#!/usr/bin/env python3
"""Evaluate an exported SAVH2 TensorFlow Lite model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from train_SAVH2_tflite import (
    DATA_ROOT,
    align_columns,
    load_label_tool_csv,
    read_label_tool_labels,
)

DEFAULT_CHECKPOINT = os.path.join("SAV_detection", "models_savh2")
ImageFile.LOAD_TRUNCATED_IMAGES = True
tflite = None


def load_tflite_runtime() -> None:
    global tflite
    try:
        import tflite_runtime.interpreter as _tflite
        tflite = _tflite
        return
    except Exception:
        pass

    try:
        import tensorflow as tf

        class _TFLite:
            Interpreter = tf.lite.Interpreter

        tflite = _TFLite
    except Exception:
        print("No TFLite runtime available.")
        sys.exit(1)


def load_metadata(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {
            "format": "legacy_classes",
            "classes": payload,
            "scalars": [],
            "ordinals": [],
            "ordinal_bins": {},
            "threshold": 0.5,
            "offsets": {
                "classes": [0, len(payload)],
                "scalars": [len(payload), len(payload)],
                "ordinals": {},
            },
        }
    return payload


def get_input_info(input_detail) -> Tuple[int, int, str]:
    shape = [int(v) for v in input_detail["shape"]]
    if len(shape) != 4:
        raise ValueError(f"Expected 4D model input, got {shape}")
    if shape[3] == 3:
        return shape[1], shape[2], "nhwc"
    if shape[1] == 3:
        return shape[2], shape[3], "nchw"
    raise ValueError(f"Could not infer channel layout from input shape {shape}")


def prepare_image(path: str, input_detail) -> np.ndarray:
    h, w, layout = get_input_info(input_detail)
    dtype = input_detail["dtype"]
    with Image.open(path) as img:
        img = img.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img)
    if np.issubdtype(dtype, np.floating):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(dtype)
    if layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    return arr[None, ...].astype(dtype, copy=False)


def dequantize_if_needed(arr: np.ndarray, detail) -> np.ndarray:
    quant = detail.get("quantization", None)
    if quant and quant[0] not in (0, 0.0):
        scale, zero_point = quant
        return (arr.astype(np.float32) - float(zero_point)) * float(scale)
    return arr.astype(np.float32, copy=False)


def parse_flat_output(flat: np.ndarray, metadata: Dict[str, object]):
    classes = metadata.get("classes", [])
    scalars = metadata.get("scalars", [])
    ordinals = metadata.get("ordinals", [])
    offsets = metadata.get("offsets", {})
    cls_start, cls_end = offsets.get("classes", [0, len(classes)])
    scalar_start, scalar_end = offsets.get("scalars", [cls_end, cls_end + len(scalars)])
    ordinal_offsets = offsets.get("ordinals", {})
    cls_probs = flat[int(cls_start):int(cls_end)]
    scalar_values = flat[int(scalar_start):int(scalar_end)]
    ordinal_values = {}
    for name in ordinals:
        start, end = ordinal_offsets.get(name, [0, 0])
        probs = flat[int(start):int(end)]
        if probs.size:
            ordinal_values[name] = probs
    return cls_probs, scalar_values, ordinal_values


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def macro_fbeta(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, beta: float) -> float:
    if tp.size == 0:
        return 0.0
    beta2 = beta * beta
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    fbeta = (1.0 + beta2) * precision * recall / np.maximum(
        beta2 * precision + recall, 1e-12
    )
    return float(np.mean(fbeta))


def per_class_metrics(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, names: Sequence[str]
) -> List[dict]:
    rows = []
    for i, name in enumerate(names):
        precision = safe_div(float(tp[i]), float(tp[i] + fp[i]))
        recall = safe_div(float(tp[i]), float(tp[i] + fn[i]))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(tp[i] + fn[i]),
            }
        )
    return rows


def load_split_dataframe(
    *,
    split_name: str,
    data_root: str,
    use_split_dirs: bool,
    class_names: Sequence[str],
    scalar_names: Sequence[str],
    ordinal_defs: Sequence[dict],
    label_cols: Sequence[str],
    scalar_cols: Sequence[str],
    ordinal_cols: Sequence[str],
) -> pd.DataFrame:
    if split_name == "ground_truth":
        csv_name = "ground_truth_val.csv"
        index_name = "dataset_index_val.csv"
        split_dir = os.path.join(data_root, "val")
    else:
        csv_name = f"labeled_{split_name}.csv"
        index_name = f"dataset_index_{split_name}.csv"
        split_dir = os.path.join(data_root, split_name)

    df, _, _, _, _ = load_label_tool_csv(
        os.path.join(data_root, csv_name),
        None if use_split_dirs else os.path.join(data_root, index_name),
        class_names,
        scalar_names,
        ordinal_defs,
        split_dir=split_dir if use_split_dirs else None,
    )
    return align_columns(df, label_cols, scalar_cols, ordinal_cols)


def evaluate_split(
    *,
    split_name: str,
    df: pd.DataFrame,
    interpreter,
    input_detail,
    output_details,
    metadata: Dict[str, object],
    label_cols: Sequence[str],
    scalar_cols: Sequence[str],
    ordinal_cols: Sequence[str],
    threshold: float,
    out_dir: Path,
) -> dict:
    true_class_rows = []
    true_scalar_rows = []
    true_ordinal_rows = []
    pred_class_rows = []
    pred_scalar_rows = []
    pred_ordinal_rows = []
    prediction_rows = []
    inference_times = []
    bad_images = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval [{split_name}]"):
        try:
            inp = prepare_image(str(row["filepath"]), input_detail)
        except Exception as exc:
            bad_images += 1
            print(f"[warn] Skipping unreadable image {row['filepath']}: {exc}")
            continue

        interpreter.set_tensor(input_detail["index"], inp)
        t0 = time.time()
        interpreter.invoke()
        inference_times.append((time.time() - t0) * 1000.0)

        raw = interpreter.get_tensor(output_details[0]["index"])
        flat = dequantize_if_needed(raw, output_details[0]).reshape(-1)
        cls_probs, scalar_values, ordinal_values = parse_flat_output(flat, metadata)

        true_cls = row[list(label_cols)].astype(np.int32).to_numpy()
        pred_cls = (cls_probs >= threshold).astype(np.int32)
        true_class_rows.append(true_cls)
        pred_class_rows.append(pred_cls)

        if scalar_cols:
            true_scalars = row[list(scalar_cols)].astype(np.float32).to_numpy()
            true_scalar_rows.append(true_scalars)
            pred_scalar_rows.append(scalar_values)

        ord_pred = []
        if ordinal_cols:
            true_ord = row[list(ordinal_cols)].astype(np.int32).to_numpy()
            true_ordinal_rows.append(true_ord)
            for name in ordinal_cols:
                probs = ordinal_values.get(name)
                ord_pred.append(int(np.argmax(probs)) if probs is not None and probs.size else 0)
            pred_ordinal_rows.append(np.asarray(ord_pred, dtype=np.int32))

        pred_row = {
            "filename": row.get("filename", Path(str(row["filepath"])).name),
            "filepath": row["filepath"],
        }
        for name, prob, pred, true in zip(label_cols, cls_probs, pred_cls, true_cls):
            pred_row[f"prob_{name}"] = float(prob)
            pred_row[f"pred_{name}"] = int(pred)
            pred_row[f"true_{name}"] = int(true)
        for name, value in zip(scalar_cols, scalar_values):
            pred_row[f"pred_scalar_{name}"] = float(value)
            pred_row[f"true_scalar_{name}"] = float(row[name])
        for name, pred in zip(ordinal_cols, ord_pred):
            pred_row[f"pred_ordinal_{name}"] = int(pred)
            pred_row[f"true_ordinal_{name}"] = int(row[name])
        prediction_rows.append(pred_row)

    if not pred_class_rows:
        raise ValueError(f"No valid samples evaluated for split {split_name}.")

    y_true_cls = np.asarray(true_class_rows, dtype=np.int32)
    y_pred_cls = np.asarray(pred_class_rows, dtype=np.int32)
    tp = np.logical_and(y_true_cls == 1, y_pred_cls == 1).sum(axis=0).astype(np.float64)
    fp = np.logical_and(y_true_cls == 0, y_pred_cls == 1).sum(axis=0).astype(np.float64)
    fn = np.logical_and(y_true_cls == 1, y_pred_cls == 0).sum(axis=0).astype(np.float64)
    f1 = macro_fbeta(tp, fp, fn, beta=1.0)
    f2 = macro_fbeta(tp, fp, fn, beta=2.0)
    acc_like = float(np.mean(np.all(y_true_cls == y_pred_cls, axis=1)))

    if scalar_cols and pred_scalar_rows:
        true_scalars = np.asarray(true_scalar_rows, dtype=np.float32)
        pred_scalars = np.asarray(pred_scalar_rows, dtype=np.float32)
        scalar_diff = pred_scalars - true_scalars
        scalar_mae = float(np.mean(np.abs(scalar_diff)))
        scalar_rmse = float(math.sqrt(float(np.mean(scalar_diff * scalar_diff))))
    else:
        scalar_mae = 0.0
        scalar_rmse = 0.0

    ordinal_rows = []
    if ordinal_cols and pred_ordinal_rows:
        true_ord = np.asarray(true_ordinal_rows, dtype=np.int32)
        pred_ord = np.asarray(pred_ordinal_rows, dtype=np.int32)
        for i, name in enumerate(ordinal_cols):
            ordinal_rows.append(
                {
                    "ordinal": name,
                    "acc": float(np.mean(pred_ord[:, i] == true_ord[:, i])),
                    "mae": float(np.mean(np.abs(pred_ord[:, i] - true_ord[:, i]))),
                }
            )
        ord_acc = float(np.mean([row["acc"] for row in ordinal_rows]))
        ord_mae = float(np.mean([row["mae"] for row in ordinal_rows]))
    else:
        ord_acc = 0.0
        ord_mae = 0.0

    per_class = per_class_metrics(tp, fp, fn, label_cols)
    summary = {
        "split": split_name,
        "samples": int(len(pred_class_rows)),
        "bad_images": int(bad_images),
        "threshold": float(threshold),
        "f1_macro": f1,
        "f2_macro": f2,
        "acc_like": acc_like,
        "scalar_mae": scalar_mae,
        "scalar_rmse": scalar_rmse,
        "ordinal_acc": ord_acc,
        "ordinal_mae": ord_mae,
        "mean_inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
    }

    pd.DataFrame(per_class).to_csv(out_dir / f"eval_{split_name}_per_class.csv", index=False)
    pd.DataFrame(ordinal_rows).to_csv(out_dir / f"eval_{split_name}_per_ordinal.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / f"eval_{split_name}_predictions.csv", index=False)
    with open(out_dir / f"eval_{split_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"{split_name} | samples {len(pred_class_rows):,} | "
        f"F2_macro {f2:.4f} F1_macro {f1:.4f} | "
        f"acc_like {acc_like:.4f} | "
        f"MAE {scalar_mae:.4f} RMSE {scalar_rmse:.4f} | "
        f"OrdAcc {ord_acc:.4f} OrdMAE {ord_mae:.4f} | "
        f"{summary['mean_inference_ms']:.2f} ms/img"
    )
    return {"summary": summary, "per_class": per_class, "per_ordinal": ordinal_rows}


def compare_label_sets(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    label_cols: Sequence[str],
    ordinal_cols: Sequence[str],
    out_dir: Path,
) -> None:
    merged = df_true[["filename"] + list(label_cols) + list(ordinal_cols)].merge(
        df_pred[["filename"] + list(label_cols) + list(ordinal_cols)],
        on="filename",
        how="inner",
        suffixes=("_true", "_pred"),
    )
    if merged.empty:
        print("[warn] No overlapping filenames for val vs ground_truth comparison.")
        return

    y_true = merged[[f"{c}_true" for c in label_cols]].astype(np.int32).to_numpy()
    y_pred = merged[[f"{c}_pred" for c in label_cols]].astype(np.int32).to_numpy()
    tp = np.logical_and(y_true == 1, y_pred == 1).sum(axis=0).astype(np.float64)
    fp = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0).astype(np.float64)
    fn = np.logical_and(y_true == 1, y_pred == 0).sum(axis=0).astype(np.float64)
    pd.DataFrame(per_class_metrics(tp, fp, fn, label_cols)).to_csv(
        out_dir / "eval_val_vs_ground_truth_per_class.csv", index=False
    )

    ordinal_rows = []
    for name in ordinal_cols:
        true = merged[f"{name}_true"].astype(int).to_numpy()
        pred = merged[f"{name}_pred"].astype(int).to_numpy()
        ordinal_rows.append(
            {
                "ordinal": name,
                "acc": float(np.mean(true == pred)),
                "mae": float(np.mean(np.abs(true - pred))),
            }
        )
    pd.DataFrame(ordinal_rows).to_csv(
        out_dir / "eval_val_vs_ground_truth_per_ordinal.csv", index=False
    )
    print(
        f"val vs ground_truth | samples {len(merged):,} | "
        f"F2_macro {macro_fbeta(tp, fp, fn, beta=2.0):.4f} "
        f"F1_macro {macro_fbeta(tp, fp, fn, beta=1.0):.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a SAVH2 TFLite model.")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model", "-m", default="model.tflite")
    p.add_argument("--classes", "-l", default="classes.json")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "ground_truth", "test"],
        choices=["train", "val", "test", "ground_truth"],
    )
    p.add_argument("--threshold", "-t", type=float, default=None)
    p.add_argument("--use-index-csv", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_tflite_runtime()

    out_dir = Path(args.checkpoint)
    metadata = load_metadata(str(out_dir / args.classes))
    model_path = out_dir / args.model

    label_cols = list(metadata.get("classes", []))
    scalar_cols = list(metadata.get("scalars", []))
    ordinal_cols = list(metadata.get("ordinals", []))
    threshold = args.threshold
    if threshold is None:
        threshold = float(metadata.get("threshold", 0.5))

    class_names, scalar_names, ordinal_defs = read_label_tool_labels(
        os.path.join(args.data_root, "label_tool_state", "labels.json")
    )

    print("Loading TFLite model:", model_path)
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    h, w, layout = get_input_info(input_detail)
    print(f"Input: {w}x{h}, layout={layout}, dtype={input_detail['dtype']}")
    print(f"Classes: {len(label_cols)}, Scalars: {len(scalar_cols)}, Ordinals: {len(ordinal_cols)}")
    print(f"Threshold: {threshold:.3f}")

    use_split_dirs = not args.use_index_csv
    results = {}
    loaded_dfs = {}
    for split_name in args.splits:
        try:
            df = load_split_dataframe(
                split_name=split_name,
                data_root=args.data_root,
                use_split_dirs=use_split_dirs,
                class_names=class_names,
                scalar_names=scalar_names,
                ordinal_defs=ordinal_defs,
                label_cols=label_cols,
                scalar_cols=scalar_cols,
                ordinal_cols=ordinal_cols,
            )
        except FileNotFoundError:
            print(f"[warn] Missing CSV for split {split_name}; skipping.")
            continue

        loaded_dfs[split_name] = df
        results[split_name] = evaluate_split(
            split_name=split_name,
            df=df,
            interpreter=interpreter,
            input_detail=input_detail,
            output_details=output_details,
            metadata=metadata,
            label_cols=label_cols,
            scalar_cols=scalar_cols,
            ordinal_cols=ordinal_cols,
            threshold=threshold,
            out_dir=out_dir,
        )

    if "val" in loaded_dfs and "ground_truth" in loaded_dfs:
        compare_label_sets(
            loaded_dfs["val"],
            loaded_dfs["ground_truth"],
            label_cols,
            ordinal_cols,
            out_dir,
        )

    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
