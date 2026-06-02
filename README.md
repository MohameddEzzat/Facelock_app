# FACELOCK

FACELOCK is a Python desktop application for automatic face anonymization in images, videos, and webcam footage.

The app detects faces using OpenCV YuNet and applies privacy effects such as blur, pixelation, or full redaction. It is designed for privacy-aware use cases, such as anonymizing people in social media content, surveillance footage, journalistic material, or recorded videos before sharing them publicly.

## Features

- Detect faces in images, videos, and webcam input.
- Apply blur, pixelation, or black redaction.
- Adjust anonymization intensity.
- Use YuNet face detection with Haar Cascade fallback.
- Choose detection strictness using Sensitive, Balanced, or Strict modes.
- Toggle face boxes on and off.
- Click detected faces to keep selected faces visible.
- Add manual privacy boxes when detection misses a face.
- Drag and resize manual boxes.
- Pause video and place manual boxes on specific frames.
- Set when a manual box should stop being active.
- Control video playback speed.
- Seek through videos using the progress slider.
- Save the current processed frame.
- Record processed video output.
- View live stats, such as FPS, frame count, and detected targets.

## Tech Stack

- Python
- Tkinter
- OpenCV
- OpenCV YuNet
- NumPy
- Pillow

## Project Structure

```text
Facelock_app/
├── facelock.py
├── face_detection_yunet_2023mar.onnx
└── requirments.txt
```

## Requirements

Install the required Python packages using the existing dependency file:

```bash
pip install -r requirments.txt
```

> Note:  
> The file is currently named `requirments.txt` in the repository.  
> You can rename it later to `requirements.txt` for the standard Python naming convention.

## How to Run

Clone the repository:

```bash
git clone https://github.com/MohameddEzzat/Facelock_app.git
```

Open the project folder:

```bash
cd Facelock_app
```

Install dependencies:

```bash
pip install -r requirments.txt
```

Run the app:

```bash
python facelock.py
```

## How to Use

1. Open the application.
2. Choose a source:
   - Webcam
   - Video
   - Image
3. Choose an anonymization effect:
   - Blur
   - Pixel
   - Redact
4. Adjust the intensity slider.
5. Choose a detection filter mode:
   - Sensitive, for catching more small faces.
   - Balanced, for normal use.
   - Strict, for reducing false detections.
6. Press Start.
7. Use manual boxes if a face is missed.
8. Save the processed frame or record the processed video.

## Detection Modes

### Sensitive

Catches more small or hard-to-detect faces, but may allow more false positives.

### Balanced

Recommended default mode for normal use.

### Strict

Reduces fake detections, but may miss tiny or side faces.

## Privacy Effects

### Blur

Applies Gaussian blur to detected face regions.

### Pixel

Applies a pixelated mosaic effect to detected face regions.

### Redact

Covers detected face regions with a black box.

## Manual Boxes

Manual boxes are used when the detector misses a face.

You can add a ready box, then drag or resize it over the face. For videos, you can pause or seek to a specific frame, then set when the selected manual box should be removed.

## Use Cases

- Anonymizing faces in social media videos.
- Protecting identities in public footage.
- Preparing privacy-safe screenshots.
- Blurring faces in recorded events.
- Redacting people from journalistic or documentary material.
- Processing webcam or video footage before sharing.

## Notes

- The YuNet model file must stay in the same folder as `facelock.py`.
- If YuNet cannot load, the app falls back to Haar Cascade detection.
- For best results, use well-lit images or videos.
- Strict mode can help reduce false detections.
- Sensitive mode can help with small or distant faces.

## Future Improvements

- Convert the app into a web application.
- Add batch processing for multiple files.
- Add automatic export presets.
- Improve video processing speed.
- Add test coverage.
- Add a cleaner project structure.
- Add screenshots or demo GIFs to the README.

## Contributors

1. [Mohamed Ezzat](https://github.com/MohameddEzzat)
2. 
3. 
