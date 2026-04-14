#!/usr/bin/env python3
"""
Run a SAVH2 TensorFlow Lite model on camera frames without streaming.

This is the no-stream runtime companion for train_SAVH2_tflite.py.
It logs one row per inference to SAV_detection/logs_savh2/.
"""

import argparse
import atexit
import csv
import datetime
import json
import os
import signal
import sys
import time


CUR_PATH = os.path.dirname(__file__)
DEFAULT_CHECKPOINT = os.path.join(CUR_PATH, "models_savh2")
cv2 = None
np = None
tflite = None


def load_runtime_modules():
    global cv2, np, tflite
    try:
        import numpy as _np
    except Exception:
        print("No NumPy runtime available. Install numpy.")
        sys.exit(1)
    np = _np

    try:
        import cv2 as _cv2
    except Exception:
        print("No OpenCV runtime available. Install opencv-python.")
        sys.exit(1)
    cv2 = _cv2

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


def load_metadata(path):
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
            "offsets": {"classes": [0, len(payload)], "scalars": [len(payload), len(payload)], "ordinals": {}},
        }
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="Non-streaming SAVH2 TFLite inference logger")
    p.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model", "-m", default="model.tflite")
    p.add_argument("--classes", "-l", default="classes.json")
    p.add_argument("--device", "-d", default="/dev/video0")
    p.add_argument("--width", "-x", default=1280, type=int)
    p.add_argument("--height", "-y", default=720, type=int)
    p.add_argument("--framerate", "-f", default=30, type=int)
    p.add_argument("--threshold", "-t", default=None, type=float)
    p.add_argument("--top-k", default=5, type=int)
    p.add_argument("--livestream", action="store_true", help="Show an annotated live preview window")
    p.add_argument("--display-scale", default=1.0, type=float, help="Scale factor for --livestream preview")
    p.add_argument("--record-video", action="store_true", help="Record annotated video to disk")
    p.add_argument("--record-path", default=None, help="Output video path; defaults to videos_savh2/video_####.mp4")
    p.add_argument("--record-fps", default=None, type=float, help="Recording FPS; defaults to --framerate")
    p.add_argument("--record-codec", default="mp4v", help="OpenCV fourcc codec, for example mp4v, XVID, MJPG")
    p.add_argument("--record-raw", action="store_true", help="Record raw camera frames without annotation overlay")
    p.add_argument("--record-processed-only", action="store_true", help="Write only processed frames; output may play faster than real time")
    p.add_argument("--print-every", default=30, type=int, help="Print console output every N frames; 0 disables")
    p.add_argument("--stats-every", default=60, type=int, help="Print speed stats every N frames; 0 disables")
    p.add_argument("--flush-every", default=30, type=int, help="Flush CSV log every N rows; 0 flushes only on exit")
    p.add_argument("--sync-log-on-flush", action="store_true", help="Call fsync whenever the CSV log is flushed")
    p.add_argument("--no-log", action="store_true", help="Disable CSV logging")
    p.add_argument("--loop-sleep", default=0.0, type=float, help="Optional sleep per loop in seconds")
    return p.parse_args()


def next_log_path(logs_dir):
    os.makedirs(logs_dir, exist_ok=True)
    max_idx = 0
    for name in os.listdir(logs_dir):
        if not (name.startswith("log_") and name.endswith(".csv")):
            continue
        try:
            max_idx = max(max_idx, int(name[4:8]))
        except ValueError:
            pass
    return os.path.join(logs_dir, f"log_{max_idx + 1:04d}.csv")


def next_video_path(videos_dir):
    os.makedirs(videos_dir, exist_ok=True)
    max_idx = 0
    for name in os.listdir(videos_dir):
        if not (name.startswith("video_") and name.endswith((".mp4", ".avi", ".mkv"))):
            continue
        try:
            max_idx = max(max_idx, int(name[6:10]))
        except ValueError:
            pass
    return os.path.join(videos_dir, f"video_{max_idx + 1:04d}.mp4")


def open_video_writer(path, frame_shape, fps, codec):
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path} codec={codec} fps={fps}")
    return writer


def get_input_size_and_channels(input_shape):
    shape = [int(v) for v in input_shape]
    if len(shape) != 4:
        raise ValueError(f"Expected 4D model input, got shape {shape}")
    if shape[3] == 3:
        return shape[1], shape[2], "nhwc"
    if shape[1] == 3:
        return shape[2], shape[3], "nchw"
    raise ValueError(f"Could not infer channel order from input shape {shape}")


def prepare_input(frame, input_details):
    input_shape = input_details["shape"]
    input_dtype = input_details["dtype"]
    h, w, layout = get_input_size_and_channels(input_shape)
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if np.issubdtype(input_dtype, np.floating):
        arr = img.astype(np.float32) / 255.0
    else:
        arr = img.astype(input_dtype)

    if layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    return arr[None, ...].astype(input_dtype, copy=False)


def dequantize_if_needed(arr, detail):
    quant = detail.get("quantization", None)
    if quant and quant[0] not in (0, 0.0):
        scale, zero_point = quant
        return (arr.astype(np.float32) - float(zero_point)) * float(scale)
    return arr.astype(np.float32, copy=False)


def parse_flat_output(flat, metadata):
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


def maybe_flush_log(log_file, *, sync=False):
    if log_file is None:
        return
    log_file.flush()
    if sync:
        try:
            os.fsync(log_file.fileno())
        except Exception:
            pass


def draw_livestream_overlay(frame, *, dt_ms, top_class, top_prob, active, scalar_values, scalars, ordinal_values):
    overlay = frame.copy()
    y = 28
    lines = [
        f"Inference: {dt_ms:.1f} ms",
        f"Top: {top_class} ({top_prob:.3f})",
    ]
    if active:
        lines.append("Active: " + ", ".join(f"{name}:{prob:.2f}" for name, prob in active[:4]))
    if len(scalar_values) > 0:
        scalar_txt = ", ".join(
            f"{scalars[i] if i < len(scalars) else f'S{i}'}:{float(v):.2f}"
            for i, v in enumerate(scalar_values[:4])
        )
        lines.append("Scalars: " + scalar_txt)
    if ordinal_values:
        ord_parts = []
        for name, probs in list(ordinal_values.items())[:4]:
            pred_bin = int(np.argmax(probs))
            ord_parts.append(f"{name}:{pred_bin}")
        lines.append("Ordinals: " + ", ".join(ord_parts))

    cv2.rectangle(overlay, (8, 8), (frame.shape[1] - 8, 18 + 28 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return frame


def main():
    args = parse_args()
    load_runtime_modules()

    metadata_path = os.path.join(args.checkpoint, args.classes)
    if not os.path.exists(metadata_path):
        print("Missing classes/metadata JSON:", metadata_path)
        sys.exit(1)

    metadata = load_metadata(metadata_path)
    classes = metadata.get("classes", [])
    scalars = metadata.get("scalars", [])
    ordinals = metadata.get("ordinals", [])
    threshold = args.threshold
    if threshold is None:
        threshold = float(metadata.get("threshold", 0.5))

    model_path = os.path.join(args.checkpoint, args.model)
    print("Loading TFLite model:", model_path)
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_h, input_w, layout = get_input_size_and_channels(input_details["shape"])
    print(f"Input: {input_w}x{input_h}, layout={layout}, dtype={input_details['dtype']}")
    print(f"Classes: {len(classes)}, Scalars: {len(scalars)}, Ordinals: {len(ordinals)}")
    print(f"Multi-label threshold: {threshold:.3f}")

    log_path = None
    log_file = None
    log_writer = None
    if not args.no_log:
        logs_dir = os.path.join(CUR_PATH, "logs_savh2")
        log_path = next_log_path(logs_dir)
        log_file = open(log_path, "w", newline="", encoding="utf-8", buffering=1024 * 1024)
        log_writer = csv.writer(log_file)
        header = ["timestamp", "top_class", "top_prob", "inference_ms"]
        header.extend(f"class_{name}" for name in classes)
        header.extend(f"scalar_{name}" for name in scalars)
        header.extend(f"ordinal_{name}" for name in ordinals)
        log_writer.writerow(header)
        maybe_flush_log(log_file, sync=args.sync_log_on_flush)
        print(f"[LOG] Writing inference logs to {log_path}")
    else:
        print("[LOG] CSV logging disabled")

    print("Opening camera:", args.device)
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.framerate)
    if not cap.isOpened():
        print("ERROR: Could not open camera", args.device)
        if log_file is not None:
            log_file.close()
        sys.exit(1)

    stats = {
        "start_time": time.time(),
        "last_stats_time": time.time(),
        "frames": 0,
        "infer_sum_ms": 0.0,
        "infer_min_ms": float("inf"),
        "infer_max_ms": 0.0,
    }
    closed = {"done": False}
    video_path = None
    video_writer = None
    video_start_time = None
    video_frames_written = 0
    if args.record_video:
        video_path = args.record_path or next_video_path(os.path.join(CUR_PATH, "videos_savh2"))
        video_dir = os.path.dirname(os.path.abspath(video_path))
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        print(f"[VIDEO] Recording enabled: {video_path}")

    def print_speed_stats(prefix="[STATS]"):
        frames = int(stats["frames"])
        if frames <= 0:
            return
        now = time.time()
        elapsed = max(now - float(stats["start_time"]), 1e-9)
        mean_infer = float(stats["infer_sum_ms"]) / frames
        infer_fps = 1000.0 / mean_infer if mean_infer > 0 else 0.0
        loop_fps = frames / elapsed
        min_infer = float(stats["infer_min_ms"])
        max_infer = float(stats["infer_max_ms"])
        print(
            f"{prefix} frames={frames} loop_fps={loop_fps:.2f} "
            f"infer_mean={mean_infer:.2f}ms infer_fps={infer_fps:.2f} "
            f"infer_min={min_infer:.2f}ms infer_max={max_infer:.2f}ms"
        )

    def close_resources():
        if closed["done"]:
            return
        closed["done"] = True
        print_speed_stats(prefix="[SUMMARY]")
        try:
            nonlocal video_writer
            if video_writer is not None:
                video_writer.release()
                print(f"[VIDEO] Closed video file {video_path}")
        except Exception:
            pass
        try:
            if log_file is not None:
                maybe_flush_log(log_file, sync=True)
                log_file.close()
                print(f"[LOG] Closed log file {log_path}")
        except Exception:
            pass
        try:
            cap.release()
            print("[CAMERA] Released camera")
        except Exception:
            pass
        if args.livestream:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    atexit.register(close_resources)

    def signal_handler(signum, frame):
        print(f"[MAIN] Signal {signum} received, shutting down")
        close_resources()
        sys.exit(0)

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, signal_handler)
            except Exception:
                pass

    print("Camera opened. Press Ctrl+C to stop.")
    if args.livestream:
        print("Livestream preview enabled. Press q in the preview window to stop.")
    print()

    frame_idx = 0
    log_rows_since_flush = 0
    while True:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            time.sleep(0.05)
            continue

        inp = prepare_input(frame, input_details)
        interpreter.set_tensor(input_details["index"], inp)
        t0 = time.time()
        interpreter.invoke()
        dt_ms = (time.time() - t0) * 1000.0
        stats["frames"] += 1
        stats["infer_sum_ms"] += dt_ms
        stats["infer_min_ms"] = min(float(stats["infer_min_ms"]), dt_ms)
        stats["infer_max_ms"] = max(float(stats["infer_max_ms"]), dt_ms)

        out_detail = output_details[0]
        raw = interpreter.get_tensor(out_detail["index"])
        flat = dequantize_if_needed(raw, out_detail).reshape(-1)
        cls_probs, scalar_values, ordinal_values = parse_flat_output(flat, metadata)

        if len(cls_probs) > 0:
            top_idx = int(np.argmax(cls_probs))
            top_class = classes[top_idx] if top_idx < len(classes) else str(top_idx)
            top_prob = float(cls_probs[top_idx])
            active = [
                (classes[i], float(p))
                for i, p in enumerate(cls_probs)
                if i < len(classes) and float(p) >= threshold
            ]
            top_items = sorted(
                [(classes[i], float(p)) for i, p in enumerate(cls_probs[: len(classes)])],
                key=lambda item: item[1],
                reverse=True,
            )[: max(1, args.top_k)]
        else:
            top_class = "N/A"
            top_prob = 0.0
            active = []
            top_items = []

        if args.print_every > 0 and frame_idx % args.print_every == 0:
            print("-----------------------------------------------------------")
            print(f"Frame: {frame_idx} | Inference: {dt_ms:.2f} ms")
            print(f"Top class: {top_class} | Prob: {top_prob:.3f}")
            if active:
                print("Active labels:")
                for name, prob in active:
                    print(f"  {name}: {prob:.3f}")
            else:
                print("Active labels: none")

            if top_items:
                print("Top labels:")
                for name, prob in top_items:
                    print(f"  {name}: {prob:.3f}")

            if len(scalar_values) > 0:
                print("Scalars:")
                for i, value in enumerate(scalar_values):
                    name = scalars[i] if i < len(scalars) else f"S{i}"
                    print(f"  {name}: {float(value):.4f}")

            if ordinal_values:
                print("Ordinals:")
                for name, probs in ordinal_values.items():
                    pred_bin = int(np.argmax(probs))
                    pred_prob = float(probs[pred_bin])
                    print(f"  {name}: bin {pred_bin} | Prob: {pred_prob:.3f}")

        if log_writer is not None:
            row = [
                datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                top_class,
                f"{top_prob:.6f}",
                f"{dt_ms:.3f}",
            ]
            row.extend(f"{float(p):.6f}" for p in cls_probs[: len(classes)])
            row.extend(f"{float(v):.6f}" for v in scalar_values[: len(scalars)])
            for name in ordinals:
                probs = ordinal_values.get(name)
                row.append("" if probs is None or probs.size == 0 else str(int(np.argmax(probs))))
            log_writer.writerow(row)
            log_rows_since_flush += 1
            if args.flush_every > 0 and log_rows_since_flush >= args.flush_every:
                maybe_flush_log(log_file, sync=args.sync_log_on_flush)
                log_rows_since_flush = 0

        if args.stats_every > 0 and frame_idx % args.stats_every == 0:
            print_speed_stats()

        annotated_frame = None
        if args.livestream or args.record_video:
            annotated_frame = frame.copy() if args.record_raw else draw_livestream_overlay(
                frame.copy(),
                dt_ms=dt_ms,
                top_class=top_class,
                top_prob=top_prob,
                active=active,
                scalar_values=scalar_values,
                scalars=scalars,
                ordinal_values=ordinal_values,
            )

        if args.record_video:
            if video_writer is None:
                record_fps = args.record_fps if args.record_fps is not None else args.framerate
                video_writer = open_video_writer(video_path, annotated_frame.shape, record_fps, args.record_codec)
                video_start_time = time.time()
                video_frames_written = 0

            if args.record_processed_only:
                video_writer.write(annotated_frame)
                video_frames_written += 1
            else:
                record_fps = args.record_fps if args.record_fps is not None else args.framerate
                elapsed = max(time.time() - video_start_time, 0.0)
                target_frames = max(1, int(elapsed * float(record_fps)) + 1)
                frames_to_write = max(1, target_frames - video_frames_written)
                for _ in range(frames_to_write):
                    video_writer.write(annotated_frame)
                video_frames_written += frames_to_write

        if args.livestream:
            preview = annotated_frame
            if args.display_scale > 0 and args.display_scale != 1.0:
                preview = cv2.resize(preview, None, fx=args.display_scale, fy=args.display_scale)
            cv2.imshow("SAVH2 livestream", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[MAIN] q pressed, shutting down")
                break

        if args.loop_sleep > 0:
            time.sleep(args.loop_sleep)


if __name__ == "__main__":
    main()
