# FaceLocking

A **Face Locking and Action Detection system** built on top of a research-grade face recognition pipeline.
This project adds **target locking**, **stable tracking**, and **facial action detection** (blink, smile, move) to the core recognition engine.

---

## ✨ Features

* 📷 Real-time webcam capture
* 🧠 Face detection
* 🎯 5-point facial landmark extraction
* 🧭 Face alignment (112×112 ArcFace standard)
* 🧬 ArcFace embeddings via ONNX Runtime
* 📦 Face enrollment & database creation
* 🔍 Live face recognition with threshold control
* 🧪 Evaluation of genuine vs impostor distances

---

## 📁 Project Structure

```
face-recognition-5pt/
│
├── data/
│   ├── enroll/          # Raw & aligned enrollment images
│   └── db/              # Face embeddings database
│
├── models/
│   └── embedder_arcface.onnx
│
├── src/
│   ├── init_projects.py # Project generator
│   ├── camera.py        # Webcam test
│   ├── detect.py        # Face detection
│   ├── landmarks.py     # 5-point landmark extraction
│   ├── align.py         # Face alignment
│   ├── embed.py         # ArcFace embedding
│   ├── enroll.py        # Enrollment pipeline
│   ├── recognize.py     # Live recognition
│   ├── evaluate.py      # Threshold evaluation
│   └── haar_5pt.py      # Haar + landmark helpers
│
└── book/                # Reference materials
```

---

## ⚙️ Requirements

* Python **3.9+** (recommended)
* Webcam
* OS: Windows / Linux / macOS

### Python Dependencies

```
opencv-python
numpy
onnxruntime
scipy
tqdm
mediapipe
```

Install all dependencies:

```bash
pip install opencv-python numpy onnxruntime scipy tqdm mediapipe
```

---

## 🚀 Quick Start

### 1️⃣ Create Project Structure

```bash
python src/init_projects.py
```

---

### 2️⃣ Test Webcam

```bash
python -m src.camera
```

Press **q** to exit.

---

### 3️⃣ Face Detection

```bash
python -m src.detect
```

You should see a bounding box around detected faces.

---

### 4️⃣ Landmark Detection (5-point)

```bash
python -m src.landmarks
```

Five facial landmarks should appear:

* Left eye
* Right eye
* Nose
* Left mouth corner
* Right mouth corner

---

### 5️⃣ Face Alignment (Critical Step)

```bash
python -m src.align
```

Outputs a **112×112 aligned face** suitable for ArcFace.

---

## y ArcFace Model Setup

Download the **ArcFace ONNX model** (InsightFace):

```bash
curl -L -o buffalo_l.zip https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download
unzip buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
```

(Optional cleanup)

```bash
rm buffalo_l.zip w600k_r50.onnx
```

---

### Validate Embeddings

```bash
python -m src.embed
```

Expected output:

* Embedding dimension: **512**
* High cosine similarity between same-face frames

---

## 👤 Enrollment

Register known identities into the database.

```bash
python -m src.enroll
```

Controls:

* **SPACE** → capture frame
* **A** → auto capture
* **Q** → quit and save

Enrollment data saved in:

```
data/enroll/
data/db/
```

---

##  Threshold Evaluation

Determine the optimal recognition threshold:

```bash
python -m src.evaluate
```

Outputs:

* Genuine distances
* Impostor distances
* Recommended threshold value

---

##  Live Recognition

```bash
python -m src.recognize
```

Controls:

* **+** increase threshold (more permissive)
* **-** decrease threshold (stricter)
* **Q** quit

---

##  System Pipeline

### Enrollment

```
Camera → Detect → Landmarks → Align → Embed → Average → Save
```

### Recognition

```
Camera → Detect → Landmarks → Align → Embed → Compare → Threshold → Result
```

---

##  Common Pitfalls

* Skipping face alignment
* Enrolling with poor lighting
* Using only one enrollment image
* Changing models without re-enrolling

---

##  Notes

* CPU-only (no GPU required)
* Deterministic and explainable pipeline
* Suitable for attendance systems, access control, exams, and research

---

##  License

This project is provided for **educational and research purposes**.

---

## 🙌 Acknowledgements

* InsightFace / ArcFace
* OpenCV
* MediaPipe

---

---

## 🔒 Face Locking Feature

This system supports **Face Locking** for behavior tracking:

### How Face Locking Works

1. **Manual Face Selection**: At startup, select one enrolled identity to lock (e.g., "Gabi" or "Fani").
2. **Locking**: When the selected face appears and is confidently recognized, the system locks onto it and displays a clear visual indicator (blue bounding box and text overlay).
3. **Stable Tracking**: The system tracks the locked face across frames, tolerates brief recognition failures, and only releases the lock if the face disappears for a set duration (~2 seconds).
4. **Action Detection**: While locked, the system detects and records simple face actions:
	 - Face moved left
	 - Face moved right
	 - Eye blink
	 - Smile or laugh (simple detection)
5. **Action History Recording**: All detected actions are recorded to a timeline file.

### Actions Detected

- **move_left**: Face moved left in the frame
- **move_right**: Face moved right in the frame
- **eye_blink**: Eye blink detected
- **smile**: Smile or laugh detected

### History File Naming and Storage

- History files are named as `<face>_history_<timestamp>.txt` (e.g., `gabi_history_20260129112099.txt`).
- Each record includes:
	- Timestamp
	- Action type
	- Brief description or value
- Files are stored in `data/db/`.

---

## 📖 Example History Record

```
2026-02-01 11:20:59	move_left	Face moved left by 32.0 px
2026-02-01 11:21:01	eye_blink	Left eye blink detected
2026-02-01 11:21:03	smile	Smile/laugh detected
```

---

Happy hacking 🚀
