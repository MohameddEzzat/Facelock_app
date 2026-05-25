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

        # Lower value detects more small faces
        # Higher value reduces fake detections
        self.yunet_score_threshold = 0.35

        self.yunet_nms_threshold = 0.30
        self.yunet_top_k = 5000

        # Multi scale detection
        # 1.0 detects normal faces
        # 2.0 and 3.0 help detect small faces
        self.scales = [1.0, 2.0, 3.0]

        # Prevent the app from becoming extremely slow on huge images
        self.max_detection_side = 2600

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
                self.yunet_score_threshold,
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
                self.yunet_score_threshold,
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

            if score < self.yunet_score_threshold:
                continue

            if bw <= 0 or bh <= 0:
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

            # Keep this loose so small tilted faces are not removed
            if aspect_ratio < 0.30 or aspect_ratio > 2.40:
                continue

            # Ignore boxes that are almost the whole frame
            frame_area = w * h
            box_area = box_w * box_h
            area_ratio = box_area / frame_area

            if area_ratio > 0.80:
                continue

            # Allow very small faces
            if box_w < 5 or box_h < 5:
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

def draw_hud(frame, faces, effect, intensity, mode, fps, frame_count):
    h, w = frame.shape[:2]
    ts = time.strftime("%H:%M:%S")

    # Corner brackets on full frame
    cl, ct = 22, 2
    for (cx, cy) in [(0,0),(w,0),(0,h),(w,h)]:
        sx = 1 if cx == 0 else -1
        sy = 1 if cy == 0 else -1
        cv2.line(frame, (cx, cy), (cx + sx*cl, cy), GOLD_BGR, ct)
        cv2.line(frame, (cx, cy), (cx, cy + sy*cl), GOLD_BGR, ct)

    # Top-left panel
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (260, 95), DARK_BGR, -1)
    frame = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
    cv2.putText(frame, "FACELOCK", (8, 20), cv2.FONT_HERSHEY_DUPLEX, 0.58, GOLD_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"MODE     : {mode}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"EFFECT   : {effect.upper()}", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"INTENSITY: {intensity}", (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"TARGETS  : {len(faces)}", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                GREEN_BGR if faces else WHITE_BGR, 1, cv2.LINE_AA)

    # Top-right panel
    ov2 = frame.copy()
    cv2.rectangle(ov2, (w - 190, 0), (w, 45), DARK_BGR, -1)
    frame = cv2.addWeighted(ov2, 0.6, frame, 0.4, 0)
    cv2.putText(frame, ts, (w - 168, 18), cv2.FONT_HERSHEY_DUPLEX, 0.52, GOLD_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:3d}  F:{frame_count:05d}", (w - 182, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, WHITE_BGR, 1, cv2.LINE_AA)

    # Face boxes with reticle corners
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GOLD_BGR, 1)
        for (bx, by) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            sx = 1 if bx == x1 else -1
            sy = 1 if by == y1 else -1
            cv2.line(frame, (bx, by), (bx + sx*9, by), RED_BGR, 2)
            cv2.line(frame, (bx, by), (bx, by + sy*9), RED_BGR, 2)
        cv2.putText(frame, "[ CLASSIFIED ]", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, GOLD_BGR, 1, cv2.LINE_AA)

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

        # Idle overlay text
        self._idle_id = self.canvas.create_text(
            400, 260,
            text="◈  SELECT A SOURCE TO BEGIN  ◈",
            font=(FONT_MONO, 13), fill="#2A2D3A", anchor="center"
        )
        self._idle_sub = self.canvas.create_text(
            400, 290,
            text="WEBCAM  ·  IMAGE  ·  VIDEO",
            font=(FONT_MONO, 9), fill="#1A1D28", anchor="center"
        )

    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=BG, width=280)
        sb.grid(row=1, column=1, sticky="ns", padx=(6, 12), pady=12)
        sb.columnconfigure(0, weight=1)
        sb.grid_propagate(False)

        row = 0

        # ── SOURCE ─────────────────────────────
        row = self._section(sb, "SOURCE", row)
        src_frame = tk.Frame(sb, bg=BG)
        src_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        src_frame.columnconfigure((0,1,2), weight=1)
        row += 1

        self._src_btns = {}
        sources = [("📷", "WEBCAM", "webcam"), ("🖼", "IMAGE", "image"), ("🎬", "VIDEO", "video")]
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
        row = self._section(sb, "HOTKEYS (window focused)", row)
        hk_frame = tk.Frame(sb, bg=PANEL2, pady=8, padx=12)
        hk_frame.grid(row=row, column=0, sticky="ew")
        row += 1

        hotkeys = [("B", "Blur"), ("P", "Pixel"), ("R", "Redact"),
                   ("+", "Intensity ↑"), ("-", "Intensity ↓"), ("S", "Save frame"), ("Q", "Quit")]
        for i, (key, desc) in enumerate(hotkeys):
            tk.Label(hk_frame, text=f"[{key}]", font=(FONT_MONO, 8, "bold"),
                     bg=PANEL2, fg=GOLD, anchor="w", width=4).grid(row=i, column=0, sticky="w", pady=1)
            tk.Label(hk_frame, text=desc, font=(FONT_MONO, 7),
                     bg=PANEL2, fg=SILVER, anchor="w").grid(row=i, column=1, sticky="w", pady=1, padx=(6,0))

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

    def _tick_clock(self):
        self.clock_var.set(time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _on_canvas_resize(self, event):
        self.canvas.coords(self._idle_id, event.width // 2, event.height // 2 - 15)
        self.canvas.coords(self._idle_sub, event.width // 2, event.height // 2 + 15)

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

    # ─── SOURCE SELECTION ────────────────────
    def _select_source(self, mode):
        self._stop_action()
        self.source_mode.set(mode)
        self._highlight_source(mode)

        if mode == "image":
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
                self._set_status(f"VIDEO LOADED — {os.path.basename(path)}", GOLD)
                self.btn_start.configure(state="normal", fg=GOLD)
                self.btn_rec.configure(state="normal", fg="#E67E22")
            else:
                self.source_mode.set("none")
                self._highlight_source("none")

        elif mode == "webcam":
            self._set_status("WEBCAM SELECTED — PRESS START", GOLD)
            self.btn_start.configure(state="normal", fg=GOLD)
            self.btn_rec.configure(state="normal", fg="#E67E22")

    # ─── START / STOP ────────────────────────
    def _start_action(self):
        mode = self.source_mode.get()
        if mode == "none":
            self._set_status("SELECT A SOURCE FIRST", RED)
            return

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
            text="WEBCAM  ·  IMAGE  ·  VIDEO",
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
        faces = self.detector.detect(frame)
        frame = Anonymizer.apply(frame, faces, self.effect.get(), self.intensity.get())
        fps = self._calc_fps()
        frame = draw_hud(frame, faces, self.effect.get(), self.intensity.get(),
                         mode_label, fps, self.frame_count)
        self._last_frame = frame.copy()
        if self._recording and self._writer:
            self._writer.write(frame)
        self.frame_count += 1
        self._update_stats(len(faces), fps, self.frame_count, mode_label)
        self._push_frame(frame)

    def _push_frame(self, frame):
        """Schedule frame display on the main (Tkinter) thread."""
        # All Tkinter canvas operations MUST run on the main thread.
        self.root.after(0, self._draw_frame, frame)

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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
        self._set_status(f"VIDEO MODE — {os.path.basename(self.current_video_path)}", GREEN)
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self._set_status("VIDEO COMPLETE", GOLD)
                    break
                self._process_frame(frame, "VIDEO")
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
        if k == 'b':
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