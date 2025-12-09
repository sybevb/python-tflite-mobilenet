#!/usr/bin/env python
"""
SAV_detection.py

Stream annotated outputs from the multi-task MobileNet model trained by
`train_SAV_mobileNet.py`.

Usage example:
python3 SAV_detection.py --checkpoint models --model model.tflite --classes classes.json --device /dev/video0

This script follows the RTSP appsrc pattern used in `object-detection.py`.
It loads `model.tflite` and `classes.json`, runs inference on camera frames,
draws the predicted class and regression scalars on each frame, and streams
the annotated frames over RTSP at rtsp://<host>:8554/stream
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
    # Fallback to tensorflow if tflite_runtime is not available
    try:
        from tensorflow.lite import Interpreter as _Interpreter
        class tflite:
            Interpreter = _Interpreter
    except Exception:
        print("Neither tflite_runtime nor tensorflow.lite available. Install one to run.")
        sys.exit(1)

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

CUR_PATH = os.path.dirname(__file__)

DEFAULT_CHECKPOINT = os.path.join(CUR_PATH, "checkpoints_TF_SAV_MobmodelileNet_MTL")


def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        return json.load(f)


class InferenceDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, model_path, classes, device=0, width=1280, height=720, framerate=30, reg_labels=None):
        super(InferenceDataFactory, self).__init__()
        self.model_path = model_path
        self.classes = classes
        self.device = int(device) if isinstance(device, (str, int)) and str(device).isdigit() else device
        self.width = int(width)
        self.height = int(height)
        self.framerate = int(framerate)

        # Frame timing for GStreamer buffer
        self.duration = (1.0 / self.framerate) * Gst.SECOND

        # Video capture (use CAP_DSHOW on Windows for better compatibility)
        self.cap = cv2.VideoCapture(
            f'v4l2src device=/dev/video0 extra-controls="controls,horizontal_flip=0,vertical_flip=0" '
            f'! video/x-raw,width={self.width},height={self.height},framerate={self.framerate}/1 '
            f'! imxvideoconvert_g2d ! video/x-raw,format=BGRA ! appsink',
            cv2.CAP_GSTREAMER
        )


        # GStreamer launch string - use BGRA to match numpy's channels ordering
        self.launch_string = (
            f"appsrc name=source is-live=true format=GST_FORMAT_TIME "
            f"! video/x-raw,format=BGRA,width={self.width},height={self.height},framerate={self.framerate}/1 "
            f"! vpuenc_h264 bitrate=0 "
            f"! rtph264pay config-interval=1 name=pay0 pt=96"
        )


        # Load TFLite interpreter
        model_file = os.path.join(self.model_path, args.model)
        print("Loading TFLite model:", model_file)
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Determine input size and type
        id0 = self.input_details[0]
        self.input_shape = id0['shape']
        self.input_type = id0['dtype']
        # Expecting shape [1, H, W, C]
        self.input_size = int(self.input_shape[1])

        # Determine number of outputs and map them to class/reg
        self.num_classes = len(self.classes)
        self.num_regressors = None
        for od in self.output_details:
            shape = od['shape']
            if len(shape) == 2 and shape[1] != 1:
                # Could be classes or regressors; we infer by comparing to known num_classes
                if shape[1] == self.num_classes:
                    pass
                else:
                    # assume regressors
                    self.num_regressors = shape[1]

        if self.num_regressors is None:
            # fallback: pick any non-class second-output size
            for od in self.output_details:
                shape = od['shape']
                if len(shape) == 2 and shape[1] != self.num_classes:
                    self.num_regressors = shape[1]

        if self.num_regressors is None:
            self.num_regressors = 0

        print(f"Input size: {self.input_size}, dtype: {self.input_type}")
        print(f"Num classes: {self.num_classes}, Num regressors: {self.num_regressors}")

        # Parse user-provided regressor labels (from CLI) if given
        self.reg_labels = None
        if reg_labels is not None:
            if isinstance(reg_labels, str):
                self.reg_labels = [s.strip() for s in reg_labels.split(',') if s.strip()]
            elif isinstance(reg_labels, (list, tuple)):
                self.reg_labels = list(reg_labels)

        # Fallback heuristics for regressor labels:
        # - If number of regressors equals number of classes, assume same labels
        # - If exactly 3 regressors and no labels provided, use the requested defaults
        # - Otherwise label as R0, R1, ...
        if self.reg_labels is None:
            if self.num_regressors == self.num_classes and self.num_regressors > 0:
                self.reg_labels = list(self.classes)
            elif self.num_regressors == 3:
                self.reg_labels = ["Dirty Lens", "Spray", "Under water"]
            else:
                self.reg_labels = [f"R{i}" for i in range(self.num_regressors)]

    def on_need_data(self, src, length):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize frame for display pipeline (we will resize for model separately)
        frame_disp = cv2.resize(frame, (self.width, self.height))

        # Prepare model input
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if np.issubdtype(self.input_type, np.floating):
            inp = (img.astype(np.float32) / 255.0)[None, ...]
        else:
            # quantized model expected uint8
            inp = img.astype(self.input_type)[None, ...]

        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        t0 = time.time()
        self.interpreter.invoke()
        t1 = time.time()

        # Read outputs and determine which is class vs reg by shape
        cls_probs = None
        regs = None
        for od in self.output_details:
            arr = self.interpreter.get_tensor(od['index'])
            if arr.ndim == 2 and arr.shape[1] == self.num_classes:
                cls_probs = arr[0]
            else:
                # treat any 2-d output that is not classes as regressors
                if arr.ndim == 2:
                    regs = arr[0]
                elif arr.ndim == 1:
                    # sometimes regressors come as 1-d
                    regs = arr

        # Postprocess: class and regressors
        if cls_probs is None:
            # If class probs not found, try to find largest output
            candidate = None
            for od in self.output_details:
                arr = self.interpreter.get_tensor(od['index'])
                if arr.ndim == 2 and (candidate is None or arr.shape[1] > candidate.shape[1]):
                    candidate = arr
            if candidate is not None:
                cls_probs = candidate[0]

        pred_label = 'N/A'
        pred_prob = 0.0
        if cls_probs is not None:
            idx = int(np.argmax(cls_probs))
            pred_label = self.classes[idx]
            pred_prob = float(cls_probs[idx])

        # Draw results on frame_disp
        x = 10
        y = 30
        cv2.rectangle(frame_disp, (5, 5), (350, 140), (0, 0, 0), -1)
        cv2.putText(frame_disp, f"Class: {pred_label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_disp, f"Prob: {pred_prob:.3f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if regs is not None and len(regs) > 0:
            for i, val in enumerate(regs[:10]):
                # show up to 10 regressors with human-readable labels when available
                if i < len(self.reg_labels):
                    label = self.reg_labels[i]
                else:
                    label = f"R{i}"
                cv2.putText(frame_disp, f"{label}: {float(val):.3f}", (x, y + 60 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Put inference time
        cv2.putText(frame_disp, f"Inf: {(t1 - t0) * 1000:.1f} ms", (self.width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Convert to BGRA for GStreamer (frame_disp is BGR) and push
        frame_out = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2BGRA)

        data = GLib.Bytes.new_take(frame_out.tobytes())
        buf = Gst.Buffer.new_wrapped_bytes(data)
        buf.duration = self.duration
        timestamp = getattr(self, 'number_frames', 0) * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames = getattr(self, 'number_frames', 0) + 1

        retval = src.emit('push-buffer', buf)
        if retval != Gst.FlowReturn.OK:
            print('push-buffer returned', retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        self.rtsp_media = rtsp_media
        rtsp_media.set_reusable(True)
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class RtspServer(GstRtspServer.RTSPServer):
    def __init__(self, factory):
        super(RtspServer, self).__init__()
        self.factory = factory
        self.factory.set_shared(True)
        self.get_mount_points().add_factory('/stream', self.factory)
        self.attach(None)


def parse_args():
    p = argparse.ArgumentParser(description='SAV streaming inference (TFLite)')
    p.add_argument('--checkpoint', '-c', default=DEFAULT_CHECKPOINT, help='Checkpoint folder containing model and classes.json')
    p.add_argument('--model', '-m', default='model.tflite', help='TFLite model filename inside checkpoint folder')
    p.add_argument('--classes', '-l', default='classes.json', help='Classes JSON filename inside checkpoint folder')
    p.add_argument('--reglabels', '-r', default=None, help='Comma-separated regressor labels (e.g. "Dirty Lens,Under water,Spray")')
    p.add_argument('--device', '-d', default=0, help='Camera device (index or path)')
    p.add_argument('--width', '-x', default=1280, help='Stream width')
    p.add_argument('--height', '-y', default=720, help='Stream height')
    p.add_argument('--framerate', '-f', default=30, help='Stream framerate')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    checkpoint_dir = args.checkpoint
    classes_path = os.path.join(checkpoint_dir, args.classes)
    if not os.path.exists(classes_path):
        print('Could not find classes.json at', classes_path)
        sys.exit(1)

    classes = load_classes(classes_path)

    Gst.init(None)
    factory = InferenceDataFactory(model_path=checkpoint_dir, classes=classes, device=args.device,
                                   width=args.width, height=args.height, framerate=args.framerate,
                                   reg_labels=args.reglabels)

    server = RtspServer(factory)
    print('RTSP stream ready at rtsp://<this-host>:8554/stream')
    loop = GLib.MainLoop()
    loop.run()
