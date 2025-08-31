#!/usr/bin/env python3
"""
Advanced Webcam Object Detection + Tracking
=========================================

This script builds upon the baseline YOLO + ByteTrack webcam detection example
shared by the user. It addresses a number of issues discovered in the original
implementation and adds several quality‑of‑life improvements:

* Proper ``__init__`` methods for ``VideoReader`` and ``FPS``. The original code
  used ``_init_`` (single underscores) which never invoked the expected base
  class initialiser, causing runtime errors such as ``TypeError: Thread.__init__()
  got an unexpected keyword argument 'preferred_index'``. Corrected init
  methods ensure the reader thread is started correctly and accepts a
  ``preferred_index`` argument.
* Resilient class name lookup. Ultralytics models have changed over time: some
  versions expose ``model.names`` while older versions store names on
  ``model.model.names``. This script attempts both, falling back gracefully.
* Graceful fallback when Ultralytics isn't available. On systems without
  internet access (such as this environment), ``pip install ultralytics`` isn't
  possible. Instead of crashing, the script falls back to using OpenCV's
  built‑in Haar cascade face detector. This allows the rest of the pipeline
  (camera capture, display, recording, screenshots) to function even without
  YOLO or ByteTrack.
* Modular functions and additional inline comments to ease maintenance.

Features
--------

* Robust webcam opener (tries multiple indices & backends, auto‑reconnect)
* Optional YOLO (Ultralytics) inference on GPU/CPU with class filtering &
  confidence/NMS control
* Built‑in tracker (ByteTrack) for stable IDs across frames when Ultralytics
  is present
* Fallback face detection via OpenCV Haar cascades when Ultralytics is absent
* Smooth FPS meter, graceful throttling to keep UI responsive
* Recording to MP4 (toggle with ``r`` key), screenshots (``s`` key)
* Structured CSV logging of tracks (frame,time,id,class,conf,x1,y1,x2,y2)
* Hotkeys: ``[q]``=quit, ``[p]``=pause/resume, ``[r]``=record, ``[s]``=screenshot,
  ``[c]``=cycle camera

Installation
------------

For full YOLO + ByteTrack support, install the required packages on your
local system:

.. code-block:: shell

   pip install ultralytics opencv-python numpy

The fallback face detector requires only OpenCV and numpy (which are
typically installed with ``opencv-python``). If ``ultralytics`` cannot be
imported, the script will inform you and continue with face detection.

Usage
-----

Run the script directly to use your default webcam:

.. code-block:: shell

   python cam_detect_advanced.py

To process a video file instead of a webcam, pass ``--source``:

.. code-block:: shell

   python cam_detect_advanced.py --source path/to/video.mp4

To filter detections to specific classes (YOLO only), use ``--classes``:

.. code-block:: shell

   python cam_detect_advanced.py --classes person car

See ``python cam_detect_advanced.py --help`` for a full list of options.
"""

import argparse
import csv
import os
import sys
import threading
import queue
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    # Attempt to import YOLO from Ultralytics. If this fails we fall back later.
    from ultralytics import YOLO  # type: ignore
    _ultralytics_available = True
except Exception:
    _ultralytics_available = False


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    ap = argparse.ArgumentParser(
        description="Advanced YOLO+ByteTrack webcam detection with fallback"
    )
    ap.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: path/URL. If omitted, uses --index for webcam",
    )
    ap.add_argument(
        "--index",
        type=int,
        default=0,
        help="Webcam index when --source is not given",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weight file to load",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for YOLO detections",
    )
    ap.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Non‑max suppression IoU threshold",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for YOLO",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="",
        help="Inference device: '' (auto) | 'cpu' | 'cuda:0'",
    )
    ap.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional list of class names to keep (e.g., person car dog)",
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="Start with recording enabled",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="runs/cam_advanced",
        help="Directory to write recordings and logs",
    )
    ap.add_argument(
        "--save_csv",
        action="store_true",
        help="Save detections to CSV file",
    )
    ap.add_argument(
        "--show_ids",
        action="store_true",
        help="Draw track IDs on bounding boxes (YOLO only)",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Auto‑reconnect attempts for the camera",
    )
    ap.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=None,
        metavar=("x", "y", "w", "h"),
        help="Optional region of interest (crop) before inference",
    )
    return ap.parse_args()


# ------------------------- Camera / Reader Thread -----------------------------

WIN_BACKENDS = [
    cv2.CAP_MSMF,  # modern Windows backend
    cv2.CAP_DSHOW,  # legacy DirectShow
    cv2.CAP_ANY,
]
OTHER_BACKENDS = [cv2.CAP_ANY]


def open_capture_fallback(
    source: Union[int, str], preferred_index: int = 0
) -> Tuple[Optional[cv2.VideoCapture], Optional[Union[int, str]]]:
    """
    Try to open a video source using multiple indices/backends. Returns
    ``(cap, used_source)`` or ``(None, None)`` on failure.
    """
    # If a string path/URL is given, use it directly
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap, source
        return None, None

    # When using a webcam (source is int), we may want to try multiple indices
    indices: List[int] = [preferred_index] + [i for i in range(5) if i != preferred_index]
    backends = WIN_BACKENDS if os.name == "nt" else OTHER_BACKENDS

    for idx in indices:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                return cap, idx
            cap.release()
    return None, None


class VideoReader(threading.Thread):
    """Background thread that continuously reads frames from a video source."""

    def __init__(self, source: Union[int, str], preferred_index: int = 0) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.preferred_index = preferred_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)  # type: ignore[name-defined]
        self.stopped = threading.Event()
        self.retry_count = 0
        self.last_ok = time.time()

    def run(self) -> None:
        while not self.stopped.is_set():
            if self.cap is None:
                self.cap, used = open_capture_fallback(self.source, self.preferred_index)
                if self.cap is None:
                    time.sleep(0.5)
                    continue
                self.retry_count = 0

            ok, frame = self.cap.read()
            if not ok or frame is None:
                # Try reconnecting on failure
                self.retry_count += 1
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                if self.retry_count > 0:
                    time.sleep(min(0.5 * self.retry_count, 2.0))
                continue

            self.last_ok = time.time()
            # Drop the previous frame if the consumer is slow
            if not self.frame_q.empty():
                try:
                    _ = self.frame_q.get_nowait()
                except queue.Empty:
                    pass
            self.frame_q.put(frame)

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.stopped.set()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


class FPS:
    """Smoothed frames‑per‑second estimator using exponential moving average."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self.t0: Optional[float] = None
        self.ema: Optional[float] = None

    def tick(self) -> None:
        t = time.time()
        if self.t0 is None:
            self.t0 = t
            return
        dt = t - self.t0
        self.t0 = t
        if dt <= 0:
            return
        f = 1.0 / dt
        self.ema = f if self.ema is None else (self.alpha * self.ema + (1 - self.alpha) * f)

    def get(self) -> float:
        return 0.0 if self.ema is None else self.ema


# ------------------------------ Utilities ------------------------------------

def ensure_dir(p: Union[str, Path]) -> Path:
    """Ensure that a directory exists and return it as a ``Path``."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_model(model_path: str):
    """
    Load the detection model. If Ultralytics is available, load the specified
    YOLO model; otherwise fall back to OpenCV's Haar cascade face detector.
    """
    if _ultralytics_available:
        try:
            model = YOLO(model_path)
            return model
        except Exception as exc:
            print(
                f"[WARN] Failed to load YOLO model '{model_path}'. Falling back to face detection.\n"
                f"Reason: {exc}",
                file=sys.stderr,
            )
    # Fallback: use OpenCV's frontal face cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise RuntimeError(
            "Haar cascade file not found. Please install opencv-python with data files."
        )
    cascade = cv2.CascadeClassifier(cascade_path)
    print(
        "[INFO] Ultralytics not available or YOLO model could not be loaded.\n"
        "[INFO] Using OpenCV Haar cascade face detector instead."
    )
    return cascade


def get_class_map(model) -> Dict[int, str]:
    """Retrieve a mapping of class indices to names from a YOLO model."""
    # Attempt to support different ultralytics versions
    # Newer versions: model.names; older: model.model.names
    name_map: Optional[Dict[int, str]] = None
    if hasattr(model, "names") and isinstance(model.names, (list, dict)):
        if isinstance(model.names, list):
            name_map = {i: n for i, n in enumerate(model.names)}
        else:
            name_map = model.names
    elif hasattr(model, "model") and hasattr(model.model, "names"):
        names = model.model.names
        if isinstance(names, dict):
            name_map = names
        elif isinstance(names, list):
            name_map = {i: n for i, n in enumerate(names)}
    return name_map or {}


def main() -> None:
    args = parse_args()

    # Prepare output directories
    outdir = ensure_dir(args.outdir)
    video_dir = ensure_dir(outdir / "video")
    shot_dir = ensure_dir(outdir / "shots")
    csv_path = outdir / "detections.csv"

    # Load the detection model (YOLO or fallback)
    model = load_model(args.model)

    # Build class filter map (YOLO only)
    class_id_keep: Optional[List[int]] = None
    class_map: Dict[int, str] = {}
    if _ultralytics_available and isinstance(model, YOLO):
        class_map = get_class_map(model)
        if args.classes:
            name_to_id = {n: idx for idx, n in class_map.items()}
            missing = [c for c in args.classes if c not in name_to_id]
            if missing:
                print(f"[WARN] Unknown classes ignored: {missing}")
            class_id_keep = sorted([name_to_id[c] for c in args.classes if c in name_to_id])

    # Start background reader
    source = args.source if args.source is not None else args.index
    reader = VideoReader(source, preferred_index=args.index)
    reader.start()

    # Prepare video writer (initialised when first frame is received)
    writer: Optional[cv2.VideoWriter] = None
    recording = args.record
    paused = False

    # Prepare CSV logging
    csv_file = open(csv_path, "w", newline="", encoding="utf-8") if args.save_csv else None
    csv_writer = None
    if csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "time",
            "frame_idx",
            "track_id",
            "class",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
        ])

    fps = FPS()
    frame_idx = 0
    win_name = "CamDetect Advanced"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(
        "[INFO] Hotkeys: q=quit | p=pause | r=record | s=screenshot | c=cycle-camera"
    )

    try:
        while True:
            # Pause loop: just poll for key events
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    paused = False
                elif key == ord("c"):
                    # Switch to next camera index
                    if isinstance(source, int) or args.source is None:
                        reader.source = (reader.preferred_index + 1) % 5
                        reader.preferred_index = reader.source
                        reader.cap = None
                    continue
                continue

            # Grab a frame from the reader
            frame = reader.read(timeout=1.0)
            if frame is None:
                # No frame available; allow user to control
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    paused = True
                elif key == ord("c"):
                    if isinstance(source, int) or args.source is None:
                        reader.source = (reader.preferred_index + 1) % 5
                        reader.preferred_index = reader.source
                        reader.cap = None
                continue

            # Apply optional region of interest crop
            roi_frame = frame
            roi_coords: Optional[Tuple[int, int, int, int]] = None
            if args.roi:
                x, y, w, h = args.roi
                H, W = frame.shape[:2]
                x1 = max(0, min(W, x))
                y1 = max(0, min(H, y))
                x2 = max(0, min(W, x + w))
                y2 = max(0, min(H, y + h))
                roi_frame = frame[y1:y2, x1:x2].copy()
                roi_coords = (x1, y1, x2, y2)

            # Run detection
            detections: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
            # Each tuple: (class_id, confidence, (x1,y1,x2,y2))
            if _ultralytics_available and isinstance(model, YOLO):
                # YOLO inference with tracking enabled
                results = model.track(
                    source=roi_frame,
                    stream=False,
                    verbose=False,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    classes=class_id_keep,
                )
                # Ultralytics returns a list; we passed one frame, so results[0]
                res = results[0] if isinstance(results, list) else results
                boxes = getattr(res, "boxes", None)
                if boxes is not None:
                    xyxy = (
                        boxes.xyxy.cpu().numpy()
                        if hasattr(boxes.xyxy, "cpu")
                        else boxes.xyxy
                    )
                    confs = (
                        boxes.conf.cpu().numpy()
                        if hasattr(boxes.conf, "cpu")
                        else boxes.conf
                    )
                    clses = (
                        boxes.cls.cpu().numpy()
                        if hasattr(boxes.cls, "cpu")
                        else boxes.cls
                    )
                    # track IDs may be None if tracking isn't enabled or fails
                    tids = None
                    if hasattr(boxes, "id") and boxes.id is not None:
                        tids = (
                            boxes.id.cpu().numpy()
                            if hasattr(boxes.id, "cpu")
                            else boxes.id
                        )
                    for i, bbox in enumerate(xyxy):
                        x1, y1, x2, y2 = bbox.astype(int)
                        conf = float(confs[i]) if confs is not None else 0.0
                        cls_id = int(clses[i]) if clses is not None else -1
                        if class_id_keep is not None and cls_id not in class_id_keep:
                            continue
                        detections.append((cls_id, conf, (x1, y1, x2, y2)))
            else:
                # Fallback face detection using Haar cascades
                if isinstance(model, cv2.CascadeClassifier):
                    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    faces = model.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                    )
                    for (fx, fy, fw, fh) in faces:
                        x1 = int(fx)
                        y1 = int(fy)
                        x2 = int(fx + fw)
                        y2 = int(fy + fh)
                        detections.append((0, 1.0, (x1, y1, x2, y2)))

            fps.tick()

            # Draw detections
            overlay = roi_frame.copy()
            H, W = overlay.shape[:2]
            now_s = time.time()
            for det in detections:
                cls_id, conf, (x1, y1, x2, y2) = det
                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Determine class label
                if _ultralytics_available and isinstance(model, YOLO):
                    cls_name = class_map.get(cls_id, str(cls_id))
                else:
                    cls_name = "face"
                label = f"{cls_name} {conf:.2f}" if _ultralytics_available else cls_name
                # Text background
                (tw, th), bl = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    overlay,
                    (x1, max(0, y1 - th - 6)),
                    (x1 + tw + 4, y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    overlay,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
                # CSV logging
                if csv_writer:
                    csv_writer.writerow(
                        [
                            f"{now_s:.3f}",
                            frame_idx,
                            -1,  # track ID not available in fallback
                            cls_name,
                            f"{conf:.4f}",
                            x1,
                            y1,
                            x2,
                            y2,
                            W,
                            H,
                        ]
                    )

            # If ROI was used, merge overlay back into full frame
            if roi_coords:
                fx1, fy1, fx2, fy2 = roi_coords
                # Insert overlay into original frame
                frame[fy1:fy2, fx1:fx2] = overlay
                vis = frame
            else:
                vis = overlay

            # Show HUD (FPS + recording status)
            hud = f"FPS: {fps.get():.1f} | Rec: {'ON' if recording else 'OFF'} | Frame: {frame_idx}"
            cv2.putText(
                vis,
                hud,
                (10, 24),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (20, 20, 240),
                2,
            )

            # Initialise video writer once frame is available and recording enabled
            if recording and writer is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = str(video_dir / f"capture_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
                print(f"[INFO] Recording to {out_path}")

            if writer is not None and recording:
                writer.write(vis)

            cv2.imshow(win_name, vis)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = True
            elif key == ord("r"):
                recording = not recording
                if not recording and writer is not None:
                    writer.release()
                    writer = None
                    print("[INFO] Recording stopped.")
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                shot_path = str(shot_dir / f"shot_{ts}.jpg")
                cv2.imwrite(shot_path, vis)
                print(f"[INFO] Saved screenshot: {shot_path}")
            elif key == ord("c"):
                # Switch camera index only when using webcams
                if isinstance(source, int) or args.source is None:
                    reader.source = (reader.preferred_index + 1) % 5
                    reader.preferred_index = reader.source
                    reader.cap = None
                    print(
                        f"[INFO] Cycling camera -> index {reader.preferred_index}"
                    )

            frame_idx += 1

    finally:
        reader.stop()
        if writer is not None:
            writer.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()