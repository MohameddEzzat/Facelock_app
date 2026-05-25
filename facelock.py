"""
FACELOCK v1.0 — Face Anonymization Suite
Run: python facelock.py
Requires: pip install opencv-python numpy Pillow
"""

import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os

# ─────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────
BG       = "#04060A"
PANEL    = "#0A0D15"
PANEL2   = "#0D1120"
GOLD     = "#D4AF37"
GOLD2    = "#F5D060"
SILVER   = "#A8B2C0"
RED      = "#C0392B"
GREEN    = "#27AE60"
BORDER   = "#2A2D3A"
TEXT     = "#C8D0DC"
FONT_MONO = "Courier"


# ─────────────────────────────────────────────
#  FACE DETECTOR
# ─────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        # YuNet OpenCV DNN settings
        self.use_yunet = False
        self.yunet_tried_loading = False
        self.yunet_detector = None

        # YuNet internal detector threshold stays low enough to return candidates.
        # FACELOCK then applies its own stricter filtering below.
        self.yunet_internal_score_threshold = 0.30
        self.yunet_score_threshold = 0.55

        self.yunet_nms_threshold = 0.30
        self.yunet_top_k = 5000

        # Detection filter mode
        # sensitive = catches more tiny faces but may allow more false positives
        # balanced  = recommended default
        # strict    = fewer fake detections but may miss small/side faces
        self.filter_mode = "balanced"
        self.use_landmark_filter = True
        self.min_face_pixels = 14
        self.min_face_area_ratio = 0.00008
        self.min_aspect_ratio = 0.45
        self.max_aspect_ratio = 1.95

        # Multi scale detection
        # More scales help tiny faces but can also increase fake detections.
        self.scales = [1.0, 1.5, 2.0]

        # Prevent the app from becoming extremely slow on huge images
        self.max_detection_side = 2400

        # Model file must be in the same folder as Facelock.py
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "face_detection_yunet_2023mar.onnx"
        )

        # Haar fallback
        # This will only be used if YuNet cannot load
        self.use_haar_fallback = True

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            print("[FACELOCK] ERROR: Haar Cascade failed to load")
        else:
            print("[FACELOCK] Haar Cascade fallback loaded")

    def set_filter_mode(self, mode):
        """Change detector strictness without changing the rest of the app."""
        if mode not in ("sensitive", "balanced", "strict"):
            mode = "balanced"

        self.filter_mode = mode

        if mode == "sensitive":
            self.yunet_score_threshold = 0.38
            self.scales = [1.0, 2.0, 3.0]
            self.min_face_pixels = 8
            self.min_face_area_ratio = 0.00003
            self.min_aspect_ratio = 0.30
            self.max_aspect_ratio = 2.40
            self.use_landmark_filter = False
        elif mode == "strict":
            self.yunet_score_threshold = 0.70
            self.scales = [1.0, 1.5]
            self.min_face_pixels = 20
            self.min_face_area_ratio = 0.00015
            self.min_aspect_ratio = 0.55
            self.max_aspect_ratio = 1.70
            self.use_landmark_filter = True
        else:
            self.yunet_score_threshold = 0.55
            self.scales = [1.0, 1.5, 2.0]
            self.min_face_pixels = 14
            self.min_face_area_ratio = 0.00008
            self.min_aspect_ratio = 0.45
            self.max_aspect_ratio = 1.95
            self.use_landmark_filter = True

    def _score_threshold_for_scale(self, actual_scale):
        """Require stronger confidence when the image is enlarged for small-face search."""
        extra = 0.0
        if actual_scale >= 2.5:
            extra = 0.18
        elif actual_scale >= 2.0:
            extra = 0.12
        elif actual_scale >= 1.5:
            extra = 0.07
        return min(0.95, self.yunet_score_threshold + extra)

    def _valid_yunet_landmarks(self, det, x, y, bw, bh):
        """Reject face boxes whose YuNet landmarks do not look like a real face."""
        if not self.use_landmark_filter:
            return True

        if bw <= 0 or bh <= 0:
            return False

        points = []
        for i in range(5):
            px = float(det[4 + i * 2])
            py = float(det[5 + i * 2])
            points.append((px, py))

        margin_x = bw * 0.30
        margin_y = bh * 0.30
        inside_count = 0
        for px, py in points:
            if (x - margin_x) <= px <= (x + bw + margin_x) and (y - margin_y) <= py <= (y + bh + margin_y):
                inside_count += 1

        if inside_count < 4:
            return False

        left_eye, right_eye, nose, left_mouth, right_mouth = points
        eye_distance = float(np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))

        if eye_distance < bw * 0.12 or eye_distance > bw * 0.95:
            return False

        eye_mid_y = (left_eye[1] + right_eye[1]) / 2
        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2

        # For normal upright faces, the mouth should not appear clearly above the eyes.
        # This is intentionally loose so slightly tilted faces are not rejected.
        if mouth_mid_y < eye_mid_y - bh * 0.15:
            return False

        nose_x, nose_y = nose
        if nose_y < y - bh * 0.15 or nose_y > y + bh * 1.15:
            return False

        return True

    def _load_yunet(self, frame):
        if self.yunet_tried_loading:
            return

        self.yunet_tried_loading = True

        try:
            if not hasattr(cv2, "FaceDetectorYN_create"):
                print("[FACELOCK] ERROR: Your OpenCV version does not support FaceDetectorYN")
                print("[FACELOCK] Install with: python -m pip install --upgrade opencv-contrib-python")
                return

            if not os.path.exists(self.model_path):
                print("[FACELOCK] ERROR: YuNet model file not found")
                print("[FACELOCK] Missing file:", self.model_path)
                print("[FACELOCK] Download face_detection_yunet_2023mar.onnx and place it beside Facelock.py")
                return

            h, w = frame.shape[:2]

            self.yunet_detector = cv2.FaceDetectorYN_create(
                self.model_path,
                "",
                (w, h),
                self.yunet_internal_score_threshold,
                self.yunet_nms_threshold,
                self.yunet_top_k
            )

            self.use_yunet = True
            print("[FACELOCK] YuNet OpenCV DNN loaded successfully")

        except Exception:
            import traceback
            print("[FACELOCK] YuNet could not load")
            traceback.print_exc()
            print("[FACELOCK] Using Haar Cascade only")

    def detect(self, frame):
        self._load_yunet(frame)

        faces = []

        if self.use_yunet and self.yunet_detector is not None:
            for scale in self.scales:
                faces += self._detect_yunet_scaled(frame, scale)

            faces = self._remove_duplicate_boxes(faces)

            # If YuNet loads successfully do not use Haar
            # Haar is the main source of random fake faces
            return faces

        if self.use_haar_fallback:
            faces = self._detect_haar(frame)
            faces = self._remove_duplicate_boxes(faces)

        return faces

    def _detect_yunet_scaled(self, frame, scale):
        h, w = frame.shape[:2]

        # Decide actual scale
        target_w = int(w * scale)
        target_h = int(h * scale)

        if max(target_w, target_h) > self.max_detection_side:
            if scale == 1.0:
                actual_scale = 1.0
            else:
                return []
        else:
            actual_scale = scale

        if actual_scale == 1.0:
            work_frame = frame
        else:
            work_frame = cv2.resize(
                frame,
                None,
                fx=actual_scale,
                fy=actual_scale,
                interpolation=cv2.INTER_CUBIC
            )

        wh, ww = work_frame.shape[:2]

        try:
            self.yunet_detector.setInputSize((ww, wh))
        except Exception:
            self.yunet_detector = cv2.FaceDetectorYN_create(
                self.model_path,
                "",
                (ww, wh),
                self.yunet_internal_score_threshold,
                self.yunet_nms_threshold,
                self.yunet_top_k
            )

        retval, detections = self.yunet_detector.detect(work_frame)

        faces = []

        if detections is None:
            return faces

        for det in detections:
            x = float(det[0])
            y = float(det[1])
            bw = float(det[2])
            bh = float(det[3])
            score = float(det[14])

            if score < self._score_threshold_for_scale(actual_scale):
                continue

            if bw <= 0 or bh <= 0:
                continue

            if not self._valid_yunet_landmarks(det, x, y, bw, bh):
                continue

            # Convert box back to original frame size
            x1 = int(x / actual_scale)
            y1 = int(y / actual_scale)
            x2 = int((x + bw) / actual_scale)
            y2 = int((y + bh) / actual_scale)

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 0 or box_h <= 0:
                continue

            aspect_ratio = box_w / box_h

            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Ignore boxes that are almost the whole frame
            frame_area = w * h
            box_area = box_w * box_h
            area_ratio = box_area / frame_area

            if area_ratio > 0.80:
                continue

            if area_ratio < self.min_face_area_ratio:
                continue

            if box_w < self.min_face_pixels or box_h < self.min_face_pixels:
                continue

            # Add padding so the full face is covered
            pad_x = int(box_w * 0.25)
            pad_y = int(box_h * 0.35)

            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))

        return faces

    def _detect_haar(self, frame):
        h, w = frame.shape[:2]
        frame_area = w * h

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        min_face_size = int(min(w, h) * 0.045)
        min_face_size = max(35, min_face_size)

        haar_faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=7,
            minSize=(min_face_size, min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces = []

        for (x, y, fw, fh) in haar_faces:
            area_ratio = (fw * fh) / frame_area
            aspect_ratio = fw / fh

            if area_ratio < 0.001:
                continue

            if area_ratio > 0.70:
                continue

            if aspect_ratio < 0.45 or aspect_ratio > 1.90:
                continue

            faces.append((x, y, x + fw, y + fh))

        return faces

    def _remove_duplicate_boxes(self, boxes):
        if len(boxes) <= 1:
            return boxes

        boxes = sorted(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )

        kept = []

        for box in boxes:
            duplicate = False

            for kept_box in kept:
                if self._iou(box, kept_box) > 0.30:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(box)

        return kept

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area
# ─────────────────────────────────────────────
#  ANONYMIZER
# ─────────────────────────────────────────────
class Anonymizer:
    @staticmethod
    def apply(frame, faces, effect, intensity):
        for (x1, y1, x2, y2) in faces:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            if effect == "blur":
                k = int(np.interp(intensity, [1, 100], [3, 99])) | 1
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            elif effect == "pixel":
                block = max(3, int(np.interp(intensity, [1, 100], [3, 40])))
                h, w = roi.shape[:2]
                small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                                   interpolation=cv2.INTER_LINEAR)
                frame[y1:y2, x1:x2] = cv2.resize(small, (w, h),
                                                   interpolation=cv2.INTER_NEAREST)
            elif effect == "redact":
                frame[y1:y2, x1:x2] = 0
        return frame


# ─────────────────────────────────────────────
#  HUD DRAW (on OpenCV frame)
# ─────────────────────────────────────────────
GOLD_BGR  = (0, 175, 212)   # BGR for OpenCV
RED_BGR   = (40, 40, 192)
GREEN_BGR = (80, 200, 60)
DARK_BGR  = (20, 13, 10)
WHITE_BGR = (210, 218, 228)

def draw_hud(frame, faces, effect, intensity, mode, fps, frame_count, skipped_faces=None, show_boxes=True):
    if skipped_faces is None:
        skipped_faces = []

    h, w = frame.shape[:2]
    ts = time.strftime("%H:%M:%S")
    total_faces = len(faces) + len(skipped_faces)

    # Corner brackets on full frame
    cl, ct = 22, 2
    for (cx, cy) in [(0,0),(w,0),(0,h),(w,h)]:
        sx = 1 if cx == 0 else -1
        sy = 1 if cy == 0 else -1
        cv2.line(frame, (cx, cy), (cx + sx*cl, cy), GOLD_BGR, ct)
        cv2.line(frame, (cx, cy), (cx, cy + sy*cl), GOLD_BGR, ct)

    # Top-left panel
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (285, 112), DARK_BGR, -1)
    frame = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
    cv2.putText(frame, "FACELOCK", (8, 20), cv2.FONT_HERSHEY_DUPLEX, 0.58, GOLD_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"MODE     : {mode}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"EFFECT   : {effect.upper()}", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"INTENSITY: {intensity}", (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"ANON     : {len(faces)} / {total_faces}", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                GREEN_BGR if faces else WHITE_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"VISIBLE  : {len(skipped_faces)}", (8, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                GREEN_BGR if skipped_faces else WHITE_BGR, 1, cv2.LINE_AA)

    # Top-right panel
    ov2 = frame.copy()
    cv2.rectangle(ov2, (w - 190, 0), (w, 45), DARK_BGR, -1)
    frame = cv2.addWeighted(ov2, 0.6, frame, 0.4, 0)
    cv2.putText(frame, ts, (w - 168, 18), cv2.FONT_HERSHEY_DUPLEX, 0.52, GOLD_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:3d}  F:{frame_count:05d}", (w - 182, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, WHITE_BGR, 1, cv2.LINE_AA)

    def draw_box(box, color, corner_color, label):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        for (bx, by) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            sx = 1 if bx == x1 else -1
            sy = 1 if by == y1 else -1
            cv2.line(frame, (bx, by), (bx + sx*9, by), corner_color, 2)
            cv2.line(frame, (bx, by), (bx, by + sy*9), corner_color, 2)
        label_y = y1 - 5 if y1 > 12 else y2 + 14
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)

    if show_boxes:
        # Gold means this box will be anonymized
        for box in faces:
            draw_box(box, GOLD_BGR, RED_BGR, "[ ANONYMIZED ]")

        # Green means this box was clicked and removed from anonymization
        for box in skipped_faces:
            draw_box(box, GREEN_BGR, GREEN_BGR, "[ VISIBLE ]")

    return frame


# ─────────────────────────────────────────────
#  MAIN APPLICATION WINDOW
# ─────────────────────────────────────────────
class FacelockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FACELOCK — Face Anonymization Suite")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(960, 620)

        # State
        self.detector     = FaceDetector()
        self.effect       = tk.StringVar(value="blur")
        self.intensity    = tk.IntVar(value=30)
        self.show_boxes   = tk.BooleanVar(value=True)
        self.detect_every = tk.IntVar(value=3)
        self.filter_mode = tk.StringVar(value="balanced")
        self.playback_speed = tk.StringVar(value="1x")
        self.video_progress = tk.IntVar(value=0)
        self.source_mode  = tk.StringVar(value="none")   # none | webcam | image | video
        self.running      = False
        self.cap          = None
        self.current_image_path = None
        self.current_video_path = None
        self.frame_count  = 0
        self.fps          = 0
        self._fps_buffer  = []
        self._fps_last    = time.time()
        self._thread      = None
        self._stop_event  = threading.Event()
        self._writer      = None
        self._recording   = False

        # Performance cache
        # Video and webcam are faster because detection runs in the background after the first frame
        # and the displayed boxes are smoothed instead of jumping to every new detection result.
        self._cached_faces = []
        self._smoothed_faces = []
        self._last_detection_frame = -999
        self._last_detection_source = None
        self._last_detection_submit_frame = -999
        self._detector_busy = False
        self._detector_lock = threading.Lock()
        self._detector_call_lock = threading.Lock()
        self._detection_generation = 0
        self._box_smoothing_alpha = 0.45

        # Video controls
        self.video_total_frames = 0
        self.video_fps = 25.0
        self._video_slider_dragging = False
        self._video_seek_lock = threading.Lock()
        self._pending_seek_frame = None
        self._last_video_ui_update = 0

        # Tkinter draw throttling
        # Keeps the UI responsive by avoiding a long queue of canvas redraws
        self._frame_lock = threading.Lock()
        self._pending_frame = None
        self._draw_scheduled = False

        # Selectable face boxes
        # By default every detected face is anonymized
        # Clicking a detected box toggles that face between anonymized and visible
        self.current_faces = []
        self.disabled_faces = []
        self._faces_lock = threading.Lock()
        self._display_info = None
        self._selection_iou_threshold = 0.20

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── UI BUILDER ──────────────────────────
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(1, weight=1)

        self._build_header()
        self._build_viewport()
        self._build_sidebar()
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=PANEL, height=52)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        # Logo
        tk.Label(hdr, text="FACELOCK", font=("Courier", 22, "bold"),
                 bg=PANEL, fg=GOLD).pack(side="left", padx=(16, 0), pady=8)
        # tk.Label(hdr, text="LOCK", font=("Courier", 22, "bold"),
        #          bg=PANEL, fg=SILVER).pack(side="left", pady=8)       tk.Label(hdr, text="LOCK", font=("Courier", 22, "bold"),
        #          bg=PANEL, fg=SILVER).pack(side="left", pady=8)
        tk.Label(hdr, text="— FACE ANONYMIZATION SUITE",
                 font=(FONT_MONO, 9), bg=PANEL, fg=BORDER).pack(side="left", pady=8, padx=6)

        # Classified badge
        tk.Label(hdr, text="[ CLASSIFIED ]",
                 font=(FONT_MONO, 8), bg=PANEL, fg=RED).pack(side="right", padx=18, pady=8)

        # Live clock
        self.clock_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self.clock_var,
                 font=(FONT_MONO, 9), bg=PANEL, fg=GOLD).pack(side="right", padx=12, pady=8)
        self._tick_clock()

        # Separator line
        sep = tk.Frame(self.root, bg=GOLD, height=1)
        sep.grid(row=0, column=0, columnspan=2, sticky="sew")

    def _build_viewport(self):
        vp_frame = tk.Frame(self.root, bg=BG)
        vp_frame.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=12)
        vp_frame.rowconfigure(0, weight=1)
        vp_frame.columnconfigure(0, weight=1)

        # Canvas container with gold border
        self.canvas_border = tk.Frame(vp_frame, bg=GOLD, padx=1, pady=1)
        self.canvas_border.grid(row=0, column=0, sticky="nsew")
        self.canvas_border.rowconfigure(0, weight=1)
        self.canvas_border.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_border, bg="#000000",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Idle overlay text
        self._idle_id = self.canvas.create_text(
            400, 260,
            text="◈  SELECT A SOURCE TO BEGIN  ◈",
            font=(FONT_MONO, 13), fill="#2A2D3A", anchor="center"
        )
        self._idle_sub = self.canvas.create_text(
            400, 290,
            text="WEBCAM  ·  VIDEO  ·  IMAGE",
            font=(FONT_MONO, 9), fill="#1A1D28", anchor="center"
        )

    def _build_sidebar(self):
        # Scrollable right sidebar
        # This keeps the video canvas fixed while allowing all controls to be reached
        sidebar_shell = tk.Frame(self.root, bg=BG, width=300)
        sidebar_shell.grid(row=1, column=1, sticky="ns", padx=(6, 12), pady=12)
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar_shell.columnconfigure(0, weight=1)
        sidebar_shell.grid_propagate(False)

        sidebar_canvas = tk.Canvas(
            sidebar_shell,
            bg=BG,
            highlightthickness=0,
            bd=0,
            width=280
        )
        sidebar_scrollbar = tk.Scrollbar(
            sidebar_shell,
            orient="vertical",
            command=sidebar_canvas.yview,
            bg=PANEL2,
            troughcolor=BG,
            activebackground=GOLD
        )
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scrollbar.grid(row=0, column=1, sticky="ns")

        sb = tk.Frame(sidebar_canvas, bg=BG)
        sb.columnconfigure(0, weight=1)

        self.sidebar_canvas = sidebar_canvas
        self.sidebar_window = sidebar_canvas.create_window((0, 0), window=sb, anchor="nw")

        def _update_sidebar_scrollregion(event=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def _resize_sidebar_content(event):
            sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

        sb.bind("<Configure>", _update_sidebar_scrollregion)
        sidebar_canvas.bind("<Configure>", _resize_sidebar_content)

        # Mouse wheel support while the pointer is over the sidebar
        sidebar_canvas.bind("<Enter>", lambda event: self._enable_sidebar_scroll())
        sidebar_canvas.bind("<Leave>", lambda event: self._disable_sidebar_scroll())
        sb.bind("<Enter>", lambda event: self._enable_sidebar_scroll())
        sb.bind("<Leave>", lambda event: self._disable_sidebar_scroll())

        row = 0

        # ── SOURCE ─────────────────────────────
        row = self._section(sb, "SOURCE", row)
        src_frame = tk.Frame(sb, bg=BG)
        src_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        src_frame.columnconfigure((0,1,2), weight=1)
        row += 1

        self._src_btns = {}
        sources = [("📷", "WEBCAM", "webcam"), ("🎬", "VIDEO", "video"), ("🖼", "IMAGE", "image")]
        for col, (icon, label, key) in enumerate(sources):
            btn = tk.Button(src_frame, text=f"{icon}\n{label}",
                            font=(FONT_MONO, 8), bg=PANEL2, fg=SILVER,
                            activebackground=PANEL, activeforeground=GOLD,
                            relief="flat", bd=0, cursor="hand2", pady=8,
                            command=lambda k=key: self._select_source(k))
            btn.grid(row=0, column=col, padx=3, sticky="ew")
            self._src_btns[key] = btn

        # ── EFFECT ─────────────────────────────
        row = self._section(sb, "EFFECT", row)
        fx_frame = tk.Frame(sb, bg=BG)
        fx_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        fx_frame.columnconfigure((0,1,2), weight=1)
        row += 1

        self._fx_btns = {}
        effects = [("🌫", "BLUR", "blur"), ("⬛", "PIXEL", "pixel"), ("▮", "REDACT", "redact")]
        for col, (icon, label, key) in enumerate(effects):
            btn = tk.Button(fx_frame, text=f"{icon}\n{label}",
                            font=(FONT_MONO, 8), bg=PANEL2, fg=SILVER,
                            activebackground=PANEL, activeforeground=GOLD2,
                            relief="flat", bd=0, cursor="hand2", pady=8,
                            command=lambda k=key: self._select_effect(k))
            btn.grid(row=0, column=col, padx=3, sticky="ew")
            self._fx_btns[key] = btn
        self._select_effect("blur")  # default highlight

        # ── INTENSITY ──────────────────────────
        row = self._section(sb, "INTENSITY", row)
        int_frame = tk.Frame(sb, bg=BG)
        int_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        int_frame.columnconfigure(0, weight=1)
        row += 1

        self.intensity_label = tk.Label(int_frame, text="30",
                                        font=(FONT_MONO, 16, "bold"),
                                        bg=BG, fg=GOLD2)
        self.intensity_label.grid(row=0, column=0, sticky="w", padx=4, pady=(4, 2))

        slider = tk.Scale(int_frame, variable=self.intensity, from_=1, to=100,
                          orient="horizontal", bg=BG, fg=SILVER,
                          troughcolor=PANEL2, activebackground=GOLD,
                          highlightthickness=0, bd=0, sliderrelief="flat",
                          sliderlength=18, showvalue=False, cursor="hand2",
                          command=self._on_intensity_change)
        slider.grid(row=1, column=0, sticky="ew", padx=4)

        lbl_row = tk.Frame(int_frame, bg=BG)
        lbl_row.grid(row=2, column=0, sticky="ew", padx=4)
        tk.Label(lbl_row, text="LOW", font=(FONT_MONO, 7), bg=BG, fg=BORDER).pack(side="left")
        tk.Label(lbl_row, text="HIGH", font=(FONT_MONO, 7), bg=BG, fg=BORDER).pack(side="right")

        # ── PERFORMANCE ────────────────────────
        row = self._section(sb, "PERFORMANCE", row)
        perf_frame = tk.Frame(sb, bg=BG)
        perf_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        perf_frame.columnconfigure(0, weight=1)
        row += 1

        self.detect_every_label = tk.Label(
            perf_frame, text="ASYNC DETECT EVERY 3 FRAMES",
            font=(FONT_MONO, 8, "bold"), bg=BG, fg=GOLD2
        )
        self.detect_every_label.grid(row=0, column=0, sticky="w", padx=4, pady=(2, 0))

        detect_slider = tk.Scale(
            perf_frame, variable=self.detect_every, from_=1, to=8,
            orient="horizontal", bg=BG, fg=SILVER, troughcolor=PANEL2,
            activebackground=GOLD, highlightthickness=0, bd=0,
            sliderrelief="flat", sliderlength=18, showvalue=False,
            cursor="hand2", command=self._on_detect_every_change
        )
        detect_slider.grid(row=1, column=0, sticky="ew", padx=4)

        perf_hint = tk.Frame(perf_frame, bg=BG)
        perf_hint.grid(row=2, column=0, sticky="ew", padx=4)
        tk.Label(perf_hint, text="ACCURATE", font=(FONT_MONO, 7), bg=BG, fg=BORDER).pack(side="left")
        tk.Label(perf_hint, text="FASTER", font=(FONT_MONO, 7), bg=BG, fg=BORDER).pack(side="right")

        # ── DETECTION FILTER ───────────────────
        row = self._section(sb, "DETECTION FILTER", row)
        filter_frame = tk.Frame(sb, bg=BG)
        filter_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        filter_frame.columnconfigure((0, 1, 2), weight=1)
        row += 1

        self._filter_btns = {}
        filters = [("SENSITIVE", "sensitive"), ("BALANCED", "balanced"), ("STRICT", "strict")]
        for col, (label, key) in enumerate(filters):
            btn = tk.Button(
                filter_frame, text=label,
                font=(FONT_MONO, 7, "bold"), bg=PANEL2, fg=SILVER,
                activebackground=PANEL, activeforeground=GOLD2,
                relief="flat", bd=0, cursor="hand2", pady=7,
                command=lambda k=key: self._select_filter_mode(k)
            )
            btn.grid(row=0, column=col, padx=3, sticky="ew")
            self._filter_btns[key] = btn
        self._select_filter_mode("balanced")

        tk.Label(
            filter_frame,
            text="STRICT reduces fake boxes but can miss tiny/side faces",
            font=(FONT_MONO, 7), bg=BG, fg=BORDER, anchor="w"
        ).grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=(5, 0))

        # ── VIDEO CONTROLS ─────────────────────
        row = self._section(sb, "VIDEO CONTROLS", row)
        video_frame = tk.Frame(sb, bg=BG)
        video_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        row += 1

        speed_row = tk.Frame(video_frame, bg=BG)
        speed_row.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 4))
        speed_row.columnconfigure(1, weight=1)
        tk.Label(speed_row, text="SPEED", font=(FONT_MONO, 7), bg=BG, fg=SILVER).grid(row=0, column=0, sticky="w")

        self.speed_menu = tk.OptionMenu(
            speed_row, self.playback_speed,
            "0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "1.75x", "2x", "2.5x", "3x", "4x",
            command=self._on_video_speed_change
        )
        self.speed_menu.configure(
            bg=PANEL2, fg=GOLD2, activebackground=PANEL, activeforeground=GOLD2,
            relief="flat", bd=0, highlightthickness=0, font=(FONT_MONO, 8), cursor="hand2"
        )
        self.speed_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.video_slider = tk.Scale(
            video_frame, variable=self.video_progress, from_=0, to=1000,
            orient="horizontal", bg=BG, fg=SILVER, troughcolor=PANEL2,
            activebackground=GOLD, highlightthickness=0, bd=0,
            sliderrelief="flat", sliderlength=16, showvalue=False,
            cursor="hand2"
        )
        self.video_slider.grid(row=1, column=0, sticky="ew", padx=4)
        self.video_slider.bind("<ButtonPress-1>", self._on_video_seek_press)
        self.video_slider.bind("<ButtonRelease-1>", self._on_video_seek_release)

        self.video_time_var = tk.StringVar(value="00:00 / 00:00")
        tk.Label(video_frame, textvariable=self.video_time_var,
                 font=(FONT_MONO, 7), bg=BG, fg=BORDER).grid(row=2, column=0, sticky="e", padx=4)

        # ── BOX DISPLAY ────────────────────────
        row = self._section(sb, "BOX DISPLAY", row)
        box_frame = tk.Frame(sb, bg=BG)
        box_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        box_frame.columnconfigure(0, weight=1)
        row += 1

        self.btn_boxes = self._action_btn(box_frame, "▣  BOXES ON", GOLD2, self._toggle_boxes, 0)

        # ── ACTIONS ────────────────────────────
        row = self._section(sb, "ACTIONS", row)
        act_frame = tk.Frame(sb, bg=BG)
        act_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        act_frame.columnconfigure(0, weight=1)
        row += 1

        self.btn_start = self._action_btn(act_frame, "▶  START", GOLD, self._start_action, 0)
        self.btn_stop  = self._action_btn(act_frame, "■  STOP",  RED,  self._stop_action,  1)
        self.btn_save  = self._action_btn(act_frame, "⬇  SAVE FRAME", SILVER, self._save_frame, 2)
        self.btn_rec   = self._action_btn(act_frame, "⏺  RECORD VIDEO", "#E67E22", self._toggle_record, 3)

        self.btn_stop.configure(state="disabled", fg=BORDER)
        self.btn_save.configure(state="disabled", fg=BORDER)
        self.btn_rec.configure(state="disabled", fg=BORDER)

        # ── STATS ──────────────────────────────
        row = self._section(sb, "LIVE STATS", row)
        stats_frame = tk.Frame(sb, bg=PANEL2, pady=10, padx=12)
        stats_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        stats_frame.columnconfigure(1, weight=1)
        row += 1

        self.stat_vars = {}
        stat_rows = [("TARGETS", "0"), ("FPS", "—"), ("FRAMES", "0"), ("MODE", "IDLE")]
        for i, (lbl, val) in enumerate(stat_rows):
            tk.Label(stats_frame, text=lbl, font=(FONT_MONO, 7),
                     bg=PANEL2, fg=SILVER, anchor="w").grid(row=i, column=0, sticky="w", pady=1)
            v = tk.StringVar(value=val)
            self.stat_vars[lbl] = v
            tk.Label(stats_frame, textvariable=v, font=(FONT_MONO, 8, "bold"),
                     bg=PANEL2, fg=GOLD2, anchor="e").grid(row=i, column=1, sticky="e", pady=1)

        # ── HOTKEYS ────────────────────────────
        row = self._section(sb, "HOTKEYS", row)
        hk_frame = tk.Frame(sb, bg=PANEL2, pady=8, padx=12)
        hk_frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1

        hotkeys = [
            ("B", "Blur"),
            ("P", "Pixel"),
            ("R", "Redact"),
            ("X", "Toggle boxes"),
            ("A", "Anonymize all"),
            ("F", "Cycle filter"),
            ("+", "Intensity up"),
            ("-", "Intensity down"),
            ("S", "Save frame"),
            ("Q", "Quit")
        ]
        for i, (key, desc) in enumerate(hotkeys):
            tk.Label(hk_frame, text=f"[{key}]", font=(FONT_MONO, 8, "bold"),
                     bg=PANEL2, fg=GOLD, anchor="w", width=4).grid(row=i, column=0, sticky="w", pady=1)
            tk.Label(hk_frame, text=desc, font=(FONT_MONO, 7),
                     bg=PANEL2, fg=SILVER, anchor="w").grid(row=i, column=1, sticky="w", pady=1, padx=(6, 0))

        self._bind_sidebar_mousewheel_widgets(sb)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=26)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.status_var = tk.StringVar(value="◈  FACELOCK READY  —  SELECT A SOURCE TO BEGIN")
        tk.Label(bar, textvariable=self.status_var,
                 font=(FONT_MONO, 8), bg=PANEL, fg=SILVER,
                 anchor="w").pack(side="left", padx=14, pady=4)

        self.status_dot = tk.Label(bar, text="●", font=(FONT_MONO, 10),
                                   bg=PANEL, fg=BORDER)
        self.status_dot.pack(side="right", padx=14, pady=4)

        tk.Label(bar, text="PYTHON 3.12 · OPENCV · FACELOCK v1.0",
                 font=(FONT_MONO, 7), bg=PANEL, fg=BORDER).pack(side="right", padx=10, pady=4)

    # ─── UI HELPERS ──────────────────────────
    def _section(self, parent, label, row):
        f = tk.Frame(parent, bg=BG)
        f.grid(row=row, column=0, sticky="ew", pady=(10, 4))
        tk.Label(f, text=f"— {label} ", font=(FONT_MONO, 7), bg=BG, fg=GOLD).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, pady=6)
        return row + 1

    def _action_btn(self, parent, text, color, cmd, row):
        btn = tk.Button(parent, text=text, font=(FONT_MONO, 9, "bold"),
                        bg=PANEL2, fg=color, activebackground=PANEL,
                        activeforeground=color, relief="flat", bd=0,
                        cursor="hand2", anchor="w", padx=12, pady=7,
                        command=cmd)
        btn.grid(row=row, column=0, sticky="ew", pady=2)
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.configure(bg=PANEL))
        btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=PANEL2))
        return btn

    def _enable_sidebar_scroll(self):
        self.root.bind_all("<MouseWheel>", self._on_sidebar_mousewheel)
        self.root.bind_all("<Button-4>", self._on_sidebar_mousewheel)
        self.root.bind_all("<Button-5>", self._on_sidebar_mousewheel)

    def _disable_sidebar_scroll(self):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_sidebar_mousewheel(self, event):
        if not hasattr(self, "sidebar_canvas"):
            return

        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            direction = -1 if event.delta > 0 else 1

        self.sidebar_canvas.yview_scroll(direction, "units")

    def _bind_sidebar_mousewheel_widgets(self, widget):
        widget.bind("<MouseWheel>", self._on_sidebar_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_sidebar_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_sidebar_mousewheel, add="+")

        for child in widget.winfo_children():
            self._bind_sidebar_mousewheel_widgets(child)

    def _tick_clock(self):
        self.clock_var.set(time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _on_canvas_resize(self, event):
        self.canvas.coords(self._idle_id, event.width // 2, event.height // 2 - 15)
        self.canvas.coords(self._idle_sub, event.width // 2, event.height // 2 + 15)

    def _on_canvas_click(self, event):
        """
        Click a detected face box to remove it from anonymization
        Click the same box again to anonymize it again
        The box display can be hidden without changing anonymization
        """
        if self._display_info is None:
            return

        scale, offset_x, offset_y, original_w, original_h = self._display_info

        frame_x = int((event.x - offset_x) / scale)
        frame_y = int((event.y - offset_y) / scale)

        if frame_x < 0 or frame_y < 0 or frame_x >= original_w or frame_y >= original_h:
            return

        with self._faces_lock:
            candidates = [
                box for box in self.current_faces
                if self._point_inside_box(frame_x, frame_y, box)
            ]

            if not candidates:
                self._set_status("NO FACE BOX SELECTED", RED)
                return

            # If boxes overlap choose the smallest one under the cursor
            clicked_box = min(candidates, key=self._box_area)

            already_disabled = any(
                self._box_match_score(clicked_box, disabled_box) >= self._selection_iou_threshold
                for disabled_box in self.disabled_faces
            )

            if already_disabled:
                self.disabled_faces = [
                    disabled_box for disabled_box in self.disabled_faces
                    if self._box_match_score(clicked_box, disabled_box) < self._selection_iou_threshold
                ]
                self._set_status("FACE RE-ADDED TO ANONYMIZATION", GREEN)
            else:
                self.disabled_faces.append(clicked_box)
                self._set_status("FACE REMOVED FROM ANONYMIZATION", GOLD)

    def _point_inside_box(self, x, y, box):
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _box_area(self, box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _box_match_score(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h
        area_a = self._box_area(box_a)
        area_b = self._box_area(box_b)
        union_area = area_a + area_b - inter_area

        iou = inter_area / union_area if union_area else 0

        # Extra center check helps when the face box moves slightly between frames
        acx = (ax1 + ax2) / 2
        acy = (ay1 + ay2) / 2
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2

        center_inside = bx1 <= acx <= bx2 and by1 <= acy <= by2
        reverse_center_inside = ax1 <= bcx <= ax2 and ay1 <= bcy <= ay2

        if center_inside or reverse_center_inside:
            return max(iou, 0.25)

        return iou

    def _filter_faces_for_anonymization(self, faces):
        """
        Returns two lists
        active_faces will be anonymized
        skipped_faces were clicked by the user and will stay visible
        """
        active_faces = []
        skipped_faces = []
        updated_disabled_faces = []

        with self._faces_lock:
            self.current_faces = list(faces)

            for face in faces:
                matched_disabled = None
                best_score = 0

                for disabled_face in self.disabled_faces:
                    score = self._box_match_score(face, disabled_face)
                    if score > best_score:
                        best_score = score
                        matched_disabled = disabled_face

                if matched_disabled is not None and best_score >= self._selection_iou_threshold:
                    skipped_faces.append(face)
                    updated_disabled_faces.append(face)
                else:
                    active_faces.append(face)

            # Track the current position of skipped faces
            # Keep unmatched disabled boxes so image mode does not lose state during redraws
            for disabled_face in self.disabled_faces:
                still_tracked = any(
                    self._box_match_score(disabled_face, tracked_face) >= self._selection_iou_threshold
                    for tracked_face in updated_disabled_faces
                )
                if not still_tracked:
                    updated_disabled_faces.append(disabled_face)

            self.disabled_faces = updated_disabled_faces

        return active_faces, skipped_faces

    def _clear_face_selection(self):
        with self._faces_lock:
            self.current_faces = []
            self.disabled_faces = []
        self._display_info = None

    # ─── BUTTON HIGHLIGHTING ─────────────────
    def _select_effect(self, key):
        self.effect.set(key)
        for k, btn in self._fx_btns.items():
            if k == key:
                btn.configure(bg=PANEL, fg=GOLD2, relief="flat")
            else:
                btn.configure(bg=PANEL2, fg=SILVER, relief="flat")

    def _highlight_source(self, key):
        for k, btn in self._src_btns.items():
            btn.configure(bg=PANEL if k == key else PANEL2,
                          fg=GOLD if k == key else SILVER)

    def _on_intensity_change(self, val):
        self.intensity_label.configure(text=str(int(float(val))))

    def _on_detect_every_change(self, val):
        value = max(1, int(float(val)))
        self.detect_every_label.configure(text=f"ASYNC DETECT EVERY {value} FRAME" + ("" if value == 1 else "S"))
        self._reset_detection_cache()

    def _select_filter_mode(self, key):
        if key not in ("sensitive", "balanced", "strict"):
            key = "balanced"

        self.filter_mode.set(key)
        self.detector.set_filter_mode(key)

        if hasattr(self, "_filter_btns"):
            for mode, btn in self._filter_btns.items():
                if mode == key:
                    btn.configure(bg=PANEL, fg=GOLD2)
                else:
                    btn.configure(bg=PANEL2, fg=SILVER)

        self._reset_detection_cache()

        if hasattr(self, "status_var"):
            self._set_status(f"DETECTION FILTER SET TO {key.upper()}", GOLD)

    def _cycle_filter_mode(self):
        order = ["sensitive", "balanced", "strict"]
        current = self.filter_mode.get()
        try:
            index = order.index(current)
        except ValueError:
            index = 1
        self._select_filter_mode(order[(index + 1) % len(order)])

    def _on_video_speed_change(self, value):
        self._set_status(f"VIDEO SPEED SET TO {value}", GOLD)

    def _playback_speed_value(self):
        try:
            text = self.playback_speed.get().lower().replace("x", "").replace(",", ".")
            return max(0.25, float(text))
        except ValueError:
            return 1.0

    def _format_seconds(self, seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _refresh_video_time(self, frame_index=0):
        total_seconds = self.video_total_frames / self.video_fps if self.video_fps else 0
        current_seconds = frame_index / self.video_fps if self.video_fps else 0
        self.video_time_var.set(f"{self._format_seconds(current_seconds)} / {self._format_seconds(total_seconds)}")

    def _load_video_metadata(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.video_total_frames = 0
            self.video_fps = 25.0
            self.video_progress.set(0)
            self._refresh_video_time(0)
            return

        self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        self.video_progress.set(0)
        self._refresh_video_time(0)

    def _on_video_seek_press(self, event):
        self._video_slider_dragging = True

    def _on_video_seek_release(self, event):
        self._video_slider_dragging = False
        if self.source_mode.get() != "video" or self.video_total_frames <= 0:
            return

        ratio = self.video_progress.get() / 1000
        target_frame = int(ratio * max(0, self.video_total_frames - 1))

        with self._video_seek_lock:
            self._pending_seek_frame = target_frame

        # Seeking jumps to another part of the video
        # Clear selection to keep privacy-safe default behavior
        self._clear_face_selection()
        self._reset_detection_cache()
        self._set_status(f"SEEKING TO {self._format_seconds(target_frame / self.video_fps)}", GOLD)

    def _take_pending_seek_frame(self):
        with self._video_seek_lock:
            target = self._pending_seek_frame
            self._pending_seek_frame = None
        return target

    def _update_video_progress(self, frame_index, force=False):
        if self.video_total_frames <= 0:
            return

        now = time.time()
        if not force and now - self._last_video_ui_update < 0.10:
            return
        self._last_video_ui_update = now

        def _do():
            if not self._video_slider_dragging:
                ratio = frame_index / max(1, self.video_total_frames - 1)
                self.video_progress.set(int(max(0, min(1000, ratio * 1000))))
            self._refresh_video_time(frame_index)

        self.root.after(0, _do)

    def _reset_detection_cache(self):
        with self._detector_lock:
            self._cached_faces = []
            self._smoothed_faces = []
            self._last_detection_frame = -999
            self._last_detection_source = None
            self._last_detection_submit_frame = -999
            self._detector_busy = False
            self._detection_generation += 1

    def _detect_faces_safely(self, frame):
        """Run the detector with a lock because the YuNet detector object is reused internally."""
        try:
            with self._detector_call_lock:
                return self.detector.detect(frame)
        except Exception:
            import traceback
            print("[FACELOCK] Detection failed")
            traceback.print_exc()
            return []

    def _blend_box(self, current_box, target_box, alpha):
        return tuple(
            int(round(current_box[i] + (target_box[i] - current_box[i]) * alpha))
            for i in range(4)
        )

    def _smooth_faces_towards(self, current_faces, target_faces, alpha=None):
        """Move existing boxes toward the latest detection result to reduce visual jitter."""
        if alpha is None:
            alpha = self._box_smoothing_alpha

        target_faces = list(target_faces or [])
        current_faces = list(current_faces or [])

        if not target_faces:
            return []

        if not current_faces:
            return target_faces

        smoothed = []
        used_current_indices = set()

        for target_box in target_faces:
            best_index = None
            best_score = 0

            for index, current_box in enumerate(current_faces):
                if index in used_current_indices:
                    continue

                score = self._box_match_score(target_box, current_box)
                if score > best_score:
                    best_score = score
                    best_index = index

            if best_index is not None and best_score >= 0.08:
                used_current_indices.add(best_index)
                smoothed.append(self._blend_box(current_faces[best_index], target_box, alpha))
            else:
                smoothed.append(target_box)

        return smoothed

    def _finish_async_detection(self, detections, mode_label, frame_number, generation):
        with self._detector_lock:
            if generation != self._detection_generation:
                return

            self._cached_faces = list(detections or [])
            self._smoothed_faces = self._smooth_faces_towards(
                self._smoothed_faces,
                self._cached_faces,
                alpha=self._box_smoothing_alpha
            )
            self._last_detection_frame = frame_number
            self._last_detection_source = mode_label
            self._detector_busy = False

    def _async_detect_faces(self, frame, mode_label, frame_number, generation):
        detections = self._detect_faces_safely(frame)
        self._finish_async_detection(detections, mode_label, frame_number, generation)

    def _get_detected_faces(self, frame, mode_label):
        # Images should always be re-detected because the user may change settings while paused
        if mode_label == "IMAGE":
            return self._detect_faces_safely(frame)

        detect_every = max(1, int(self.detect_every.get()))

        with self._detector_lock:
            source_changed = self._last_detection_source != mode_label
            cache_empty = len(self._cached_faces) == 0 and len(self._smoothed_faces) == 0
            due_for_detection = (self.frame_count - self._last_detection_submit_frame) >= detect_every
            generation = self._detection_generation

        # First detection stays synchronous so the first displayed frame does not appear unprotected.
        if cache_empty and not self._detector_busy:
            detections = self._detect_faces_safely(frame.copy())
            with self._detector_lock:
                if generation == self._detection_generation:
                    self._cached_faces = list(detections or [])
                    self._smoothed_faces = list(detections or [])
                    self._last_detection_frame = self.frame_count
                    self._last_detection_submit_frame = self.frame_count
                    self._last_detection_source = mode_label
                    return list(self._smoothed_faces)
            return list(detections or [])

        # Later detections run asynchronously so video playback does not pause every detection cycle.
        if (source_changed or due_for_detection) and not self._detector_busy:
            with self._detector_lock:
                if not self._detector_busy:
                    self._detector_busy = True
                    self._last_detection_submit_frame = self.frame_count
                    self._last_detection_source = mode_label
                    generation = self._detection_generation
                    frame_number = self.frame_count
                    frame_for_detection = frame.copy()
                    threading.Thread(
                        target=self._async_detect_faces,
                        args=(frame_for_detection, mode_label, frame_number, generation),
                        daemon=True
                    ).start()

        # Each displayed frame moves boxes a small step toward the latest detection result.
        with self._detector_lock:
            self._smoothed_faces = self._smooth_faces_towards(
                self._smoothed_faces,
                self._cached_faces,
                alpha=self._box_smoothing_alpha
            )
            return list(self._smoothed_faces)

    def _toggle_boxes(self):
        self.show_boxes.set(not self.show_boxes.get())
        self._refresh_boxes_button()
        if self.show_boxes.get():
            self._set_status("FACE BOXES ARE VISIBLE", GREEN)
        else:
            self._set_status("FACE BOXES ARE HIDDEN", GOLD)

    def _refresh_boxes_button(self):
        if self.show_boxes.get():
            self.btn_boxes.configure(text="▣  BOXES ON", fg=GOLD2)
        else:
            self.btn_boxes.configure(text="□  BOXES OFF", fg=SILVER)

    # ─── SOURCE SELECTION ────────────────────
    def _select_source(self, mode):
        self._stop_action()
        self._clear_face_selection()
        self.source_mode.set(mode)
        self._highlight_source(mode)

        if mode == "image":
            self.video_progress.set(0)
            self._refresh_video_time(0)
            path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
            )
            if path:
                self.current_image_path = path
                self._set_status(f"IMAGE LOADED — {os.path.basename(path)}", GOLD)
                self.btn_start.configure(state="normal", fg=GOLD)
            else:
                self.source_mode.set("none")
                self._highlight_source("none")

        elif mode == "video":
            path = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
            )
            if path:
                self.current_video_path = path
                self._load_video_metadata(path)
                self._set_status(f"VIDEO LOADED — {os.path.basename(path)}", GOLD)
                self.btn_start.configure(state="normal", fg=GOLD)
                self.btn_rec.configure(state="normal", fg="#E67E22")
            else:
                self.source_mode.set("none")
                self._highlight_source("none")

        elif mode == "webcam":
            self.video_progress.set(0)
            self._refresh_video_time(0)
            self._set_status("WEBCAM SELECTED — PRESS START", GOLD)
            self.btn_start.configure(state="normal", fg=GOLD)
            self.btn_rec.configure(state="normal", fg="#E67E22")

    # ─── START / STOP ────────────────────────
    def _start_action(self):
        mode = self.source_mode.get()
        if mode == "none":
            self._set_status("SELECT A SOURCE FIRST", RED)
            return

        self._clear_face_selection()
        self._reset_detection_cache()
        self._stop_event.clear()
        self.running = True
        self.frame_count = 0
        self.btn_start.configure(state="disabled", fg=BORDER)
        self.btn_stop.configure(state="normal", fg=RED)
        self.btn_save.configure(state="normal", fg=SILVER)
        self.canvas.delete(self._idle_id)
        self.canvas.delete(self._idle_sub)
        self.status_dot.configure(fg=GREEN)

        if mode == "image":
            self._thread = threading.Thread(target=self._run_image, daemon=True)
        elif mode == "webcam":
            self._thread = threading.Thread(target=self._run_webcam, daemon=True)
        elif mode == "video":
            self._thread = threading.Thread(target=self._run_video, daemon=True)

        self._thread.start()

    def _stop_action(self):
        self._stop_event.set()
        self.running = False
        self._reset_detection_cache()

        # Wait for the background thread to finish so it can cleanly
        # release its own cap reference before we touch UI.
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

        if self._writer:
            self._writer.release()
            self._writer = None
            self._recording = False
            self.btn_rec.configure(text="⏺  RECORD VIDEO", fg="#E67E22")

        # cap is released by the thread itself; just clear the reference.
        self.cap = None

        self.btn_start.configure(state="normal", fg=GOLD)
        self.btn_stop.configure(state="disabled", fg=BORDER)
        self.btn_save.configure(state="disabled", fg=BORDER)
        self.status_dot.configure(fg=BORDER)
        self._update_stats(0, 0, 0, "IDLE")
        self._set_status("◈  STOPPED  —  SELECT A SOURCE TO BEGIN", SILVER)

        # Restore idle text
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 500
        self._idle_id = self.canvas.create_text(
            cw // 2, ch // 2 - 15,
            text="◈  SELECT A SOURCE TO BEGIN  ◈",
            font=(FONT_MONO, 13), fill="#2A2D3A", anchor="center"
        )
        self._idle_sub = self.canvas.create_text(
            cw // 2, ch // 2 + 15,
            text="WEBCAM  ·  VIDEO  ·  IMAGE",
            font=(FONT_MONO, 9), fill="#1A1D28", anchor="center"
        )

    # ─── RECORDING ───────────────────────────
    def _toggle_record(self):
        if not self.running:
            self._set_status("START THE FEED FIRST", RED)
            return
        if not self._recording:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4")],
                title="Save Recording As"
            )
            if save_path and self.cap:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                self._recording = True
                self.btn_rec.configure(text="⏹  STOP RECORD", fg=RED)
                self._set_status(f"RECORDING → {os.path.basename(save_path)}", RED)
        else:
            if self._writer:
                self._writer.release()
                self._writer = None
            self._recording = False
            self.btn_rec.configure(text="⏺  RECORD VIDEO", fg="#E67E22")
            self._set_status("RECORDING SAVED", GREEN)

    # ─── SAVE FRAME ──────────────────────────
    def _save_frame(self):
        if self._last_frame is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title="Save Frame As"
        )
        if path:
            cv2.imwrite(path, self._last_frame)
            self._set_status(f"FRAME SAVED → {os.path.basename(path)}", GREEN)

    # ─── PROCESSING LOOPS ────────────────────
    _last_frame = None

    def _calc_fps(self):
        now = time.time()
        dt = now - self._fps_last
        self._fps_last = now
        self._fps_buffer.append(1.0 / dt if dt > 0 else 0)
        if len(self._fps_buffer) > 15:
            self._fps_buffer.pop(0)
        return int(np.mean(self._fps_buffer))

    def _process_frame(self, frame, mode_label):
        detected_faces = self._get_detected_faces(frame, mode_label)
        faces_to_anonymize, skipped_faces = self._filter_faces_for_anonymization(detected_faces)

        # Default behavior
        # Every detected face is anonymized unless the user clicked its box
        frame = Anonymizer.apply(frame, faces_to_anonymize, self.effect.get(), self.intensity.get())

        fps = self._calc_fps()
        frame = draw_hud(frame, faces_to_anonymize, self.effect.get(), self.intensity.get(),
                         mode_label, fps, self.frame_count, skipped_faces,
                         show_boxes=self.show_boxes.get())
        self._last_frame = frame.copy()
        if self._recording and self._writer:
            self._writer.write(frame)
        self.frame_count += 1
        self._update_stats(f"{len(faces_to_anonymize)} / {len(detected_faces)}", fps, self.frame_count, mode_label)
        self._push_frame(frame)

    def _push_frame(self, frame):
        """Schedule frame display on the main Tkinter thread without queueing old frames."""
        with self._frame_lock:
            self._pending_frame = frame
            if self._draw_scheduled:
                return
            self._draw_scheduled = True

        self.root.after(0, self._draw_pending_frame)

    def _draw_pending_frame(self):
        with self._frame_lock:
            frame = self._pending_frame
            self._pending_frame = None
            self._draw_scheduled = False

        if frame is not None:
            self._draw_frame(frame)

    def _draw_frame(self, frame):
        """Convert OpenCV BGR frame → Tkinter PhotoImage and draw on canvas.
        Called only from the main thread via root.after()."""
        if not self.running:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        x, y = (cw - nw) // 2, (ch - nh) // 2
        self._display_info = (scale, x, y, w, h)
        self.canvas.delete("frame")
        self.canvas.create_image(x, y, anchor="nw", image=img, tags="frame")
        self.canvas._img_ref = img  # prevent GC

    def _run_image(self):
        self._set_status(f"PROCESSING IMAGE...", GOLD)
        frame = cv2.imread(self.current_image_path)
        if frame is None:
            self._set_status("ERROR: CANNOT READ IMAGE FILE", RED)
            return
        self._set_status(f"IMAGE MODE — {os.path.basename(self.current_image_path)}", GREEN)
        # Image mode: keep redrawing so live effect/intensity changes are reflected
        while not self._stop_event.is_set():
            self._process_frame(frame.copy(), "IMAGE")
            time.sleep(0.05)

    def _run_webcam(self):
        self._set_status("OPENING WEBCAM...", GOLD)
        cap = cv2.VideoCapture(0)
        self.cap = cap  # expose for external reference only
        if not cap.isOpened():
            self._set_status("ERROR: CANNOT ACCESS WEBCAM", RED)
            self.cap = None
            return
        # Lower live resolution improves detection speed and reduces webcam lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._set_status("WEBCAM ACTIVE — LIVE FEED RUNNING", GREEN)
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                self._process_frame(frame, "WEBCAM")
        finally:
            cap.release()
            self.cap = None

    def _run_video(self):
        self._set_status("OPENING VIDEO...", GOLD)
        cap = cv2.VideoCapture(self.current_video_path)
        self.cap = cap  # expose for external reference only
        if not cap.isOpened():
            self._set_status("ERROR: CANNOT OPEN VIDEO FILE", RED)
            self.cap = None
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or self.video_total_frames)
        self.video_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.video_fps or 25.0)
        self._update_video_progress(0, force=True)
        self._set_status(f"VIDEO MODE — {os.path.basename(self.current_video_path)}", GREEN)

        try:
            next_frame_time = time.time()

            while not self._stop_event.is_set():
                seek_frame = self._take_pending_seek_frame()
                if seek_frame is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
                    self.frame_count = seek_frame
                    self._reset_detection_cache()
                    self._update_video_progress(seek_frame, force=True)
                    next_frame_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    self._set_status("VIDEO COMPLETE", GOLD)
                    break

                current_frame = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                self.frame_count = current_frame
                self._process_frame(frame, "VIDEO")
                self._update_video_progress(current_frame)

                base_delay = 1.0 / self.video_fps if self.video_fps else 0
                speed = self._playback_speed_value()
                next_frame_time += base_delay / speed
                delay = next_frame_time - time.time()

                if delay > 0:
                    time.sleep(delay)
                else:
                    # If processing falls behind, reset the clock instead of building more stutter.
                    next_frame_time = time.time()
        finally:
            cap.release()
            self.cap = None

    # ─── STATS / STATUS UPDATES (thread-safe) ─
    def _set_status(self, msg, color=SILVER):
        self.root.after(0, lambda: self.status_var.set(f"  {msg}"))
        self.root.after(0, lambda: self.status_dot.configure(
            fg=GREEN if self.running else BORDER))

    def _update_stats(self, targets, fps, frames, mode):
        def _do():
            self.stat_vars["TARGETS"].set(str(targets))
            self.stat_vars["FPS"].set(str(fps) if fps else "—")
            self.stat_vars["FRAMES"].set(f"{frames:,}")
            self.stat_vars["MODE"].set(mode)
        self.root.after(0, _do)

    # ─── KEYBOARD SHORTCUTS ──────────────────
    def bind_keys(self):
        self.root.bind("<KeyPress>", self._on_key)

    def _on_key(self, event):
        k = event.char.lower()
        if k == 'a':
            with self._faces_lock:
                self.disabled_faces = []
            self._set_status("ALL DETECTED FACES WILL BE ANONYMIZED", GREEN)
        elif k == 'x':
            self._toggle_boxes()
        elif k == 'f':
            self._cycle_filter_mode()
        elif k == 'b':
            self._select_effect("blur")
        elif k == 'p':
            self._select_effect("pixel")
        elif k == 'r':
            self._select_effect("redact")
        elif k == '+' or k == '=':
            self.intensity.set(min(100, self.intensity.get() + 5))
            self.intensity_label.configure(text=str(self.intensity.get()))
        elif k == '-':
            self.intensity.set(max(1, self.intensity.get() - 5))
            self.intensity_label.configure(text=str(self.intensity.get()))
        elif k == 's':
            self._save_frame()
        elif k == 'q':
            self._on_close()

    # ─── CLOSE ───────────────────────────────
    def _on_close(self):
        self._stop_event.set()
        self.running = False
        if self._writer:
            self._writer.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()

    # Window icon color via title bar
    root.configure(bg=BG)

    # Center window on screen
    root.update_idletasks()
    w, h = 1100, 680
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    app = FacelockApp(root)
    app.bind_keys()
    root.mainloop()


if __name__ == "__main__":
    main()