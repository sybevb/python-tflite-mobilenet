#!/usr/bin/env python3
"""
SAV_detection_no_stream.py

Runs your multi-task MobileNet TFLite model on live camera frames
and prints class + regression output to the console.

Usage:
python3 SAV_detection_no_stream.py --checkpoint models --model model.tflite --classes classes.json --device /dev/video0
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import cv2

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        from tensorflow.lite import Interpreter as _Interpreter
        class tflite:
            Interpreter = _Interpreter
    except Exception:
        print("No TFLite runtime available.")
        sys.exit(1)

CUR_PATH = os.path.dirname(__file__)
DEFAULT_CHECKPOINT = os.path.join(CUR_PATH, "checkpoints_TF_SAV_MobmodelileNet_MTL")


def load_classes(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Non-streaming SAV inference logger")
    p.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model", "-m", default="model.tflite")
    p.add_argument("--classes", "-l", default="classes.json")
    p.add_argument("--reglabels", "-r", default=None,
                   help='Comma-separated labels for regressors')
    p.add_argument("--device", "-d", default="/dev/video0")
    p.add_argument("--width", "-x", default=1280, type=int)
    p.add_argument("--height", "-y", default=720, type=int)
    p.add_argument("--framerate", "-f", default=30, type=int)
    return p.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # Load classes
    # ---------------------------------------------------------
    classes_path = os.path.join(args.checkpoint, args.classes)
    if not os.path.exists(classes_path):
        print("Missing classes.json:", classes_path)
        sys.exit(1)

    classes = load_classes(classes_path)
    num_classes = len(classes)

    # ---------------------------------------------------------
    # Load TFLite model
    # ---------------------------------------------------------
    model_path = os.path.join(args.checkpoint, args.model)
    print("Loading TFLite model:", model_path)

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]        # [1, H, W, 3]
    input_dtype = input_details[0]["dtype"]
    input_size = int(input_shape[1])

    print(f"Model input size: {input_size}, dtype={input_dtype}")

    # ---------------------------------------------------------
    # Determine regressor count
    # ---------------------------------------------------------
    num_reg = None
    for od in output_details:
        shape = od["shape"]
        if len(shape) == 2 and shape[1] != num_classes:
            num_reg = shape[1]

    if num_reg is None:
        # fallback: any 2D non-class output
        for od in output_details:
            shape = od["shape"]
            if len(shape) == 2 and shape[1] != num_classes:
                num_reg = shape[1]

    if num_reg is None:
        num_reg = 0

    # ---------------------------------------------------------
    # Regressor label logic (same as streaming script)
    # ---------------------------------------------------------
    if args.reglabels:
        reg_labels = [s.strip() for s in args.reglabels.split(",")]
    else:
        if num_reg == num_classes and num_reg > 0:
            reg_labels = list(classes)
        elif num_reg == 3:
            reg_labels = ["Dirty Lens", "Spray", "Under water"]
        else:
            reg_labels = [f"R{i}" for i in range(num_reg)]

    print(f"Classes: {num_classes}, Regressors: {num_reg}")
    print("Regressor labels:", reg_labels)

    # ---------------------------------------------------------
    # Open camera (NO GStreamer pipeline, raw V4L2)
    # ---------------------------------------------------------
    print("Opening camera:", args.device)

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.framerate)

    if not cap.isOpened():
        print("ERROR: Could not open camera", args.device)
        sys.exit(1)

    print("Camera opened. Press Ctrl+C to stop.\n")

    # ---------------------------------------------------------
    # Main loop — inference + logging
    # ---------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            continue

        # Resize → RGB
        img = cv2.resize(frame, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prepare tensor
        if np.issubdtype(input_dtype, np.floating):
            inp = (img.astype(np.float32) / 255.0)[None, ...]
        else:
            inp = img.astype(input_dtype)[None, ...]

        interpreter.set_tensor(input_details[0]["index"], inp)

        t0 = time.time()
        interpreter.invoke()
        dt = (time.time() - t0) * 1000.0

        # ---------------------------------------------------------
        # Parse outputs
        # ---------------------------------------------------------
        cls_probs = None
        regs = None

        for od in output_details:
            arr = interpreter.get_tensor(od["index"])
            if arr.ndim == 2 and arr.shape[1] == num_classes:
                cls_probs = arr[0]
            else:
                # Any 2D or 1D non-class output = regressors
                if arr.ndim == 2:
                    regs = arr[0]
                elif arr.ndim == 1:
                    regs = arr

        # Fallback: largest output = classes
        if cls_probs is None:
            largest = None
            for od in output_details:
                arr = interpreter.get_tensor(od["index"])
                if arr.ndim == 2:
                    if largest is None or arr.shape[1] > largest.shape[1]:
                        largest = arr
            if largest is not None:
                cls_probs = largest[0]

        # Classification result
        if cls_probs is not None:
            idx = int(np.argmax(cls_probs))
            pred_class = classes[idx]
            pred_prob = float(cls_probs[idx])
        else:
            pred_class = "N/A"
            pred_prob = 0.0

        # ---------------------------------------------------------
        # PRINT TO CONSOLE
        # ---------------------------------------------------------
        print("-----------------------------------------------------------")
        print(f"Inference: {dt:.2f} ms")
        print(f"Class: {pred_class}  |  Prob: {pred_prob:.3f}")

        if regs is not None and len(regs) > 0:
            print("Regressors:")
            for i, v in enumerate(regs):
                label = reg_labels[i] if i < len(reg_labels) else f"R{i}"
                print(f"  {label}: {float(v):.4f}")

        # Small sleep to avoid spamming console too fast
        time.sleep(0.01)


if __name__ == "__main__":
    main()
