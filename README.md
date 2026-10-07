# Multimodal Driver Fatigue Detection System

🏆 **Best freshers — Panasonic Equinox'26 Hackathon**

A real-time **AI-powered driver fatigue detection system** that combines **computer vision** and **driving behavior analysis** to estimate driver alertness and trigger safety alerts.

The system monitors both **physiological signals (webcam)** and **driving telemetry (simulator)**, fusing them through a machine learning pipeline to compute a continuous **fatigue probability score**.

When fatigue exceeds safe thresholds, the system triggers **multi-channel alerts** including controller haptics, phone vibration, and visual warnings.

---

#  Problem

Driver fatigue is responsible for a large percentage of road accidents worldwide. Detecting fatigue is difficult because:

* Drivers can **keep their eyes open while microsleeping**
* Steering drift can happen naturally on curved roads
* Single-signal detection systems produce **high false positives**

To solve this, the system uses **multimodal detection** — combining **physiological signals** and **driving behaviour telemetry** to produce a more reliable fatigue estimate.

---

#  System Architecture

```
Webcam + Driving Simulator
        │
        ▼
Preprocessing Layer
(OpenCV + MediaPipe FaceMesh)
        │
        ▼
Feature Extraction
(EAR, MAR, head pose, lane drift, steering instability)
        │
        ▼
Machine Learning Inference
(LightGBM fatigue probability model)
        │
        ▼
Alert Engine
(Haptic feedback + phone vibration + UI alerts)
```

Both input streams are processed **in parallel** and fused into a single feature vector before ML inference.

---

#  Detection Modalities

## 1️) Vision-Based Fatigue Detection

Using the webcam as a dashcam surrogate.

Features extracted from **MediaPipe FaceMesh landmarks**:

* Eye Aspect Ratio (EAR)
* Blink frequency
* Eye closure duration
* Mouth Aspect Ratio (MAR) for yawning
* Head pitch angle
* Gaze direction

These signals detect **microsleep, yawning, and head nodding**.

---

## 2️) Driving Behaviour Analysis

Driving telemetry is streamed from a **browser-based driving simulator**.

Telemetry features include:

* Lane offset
* Steering angle
* Steering correction frequency
* Reaction delay
* Vehicle speed
* Steering reversals

These signals detect **loss of motor control and delayed reactions caused by fatigue**.

---

#  Feature Fusion

Both pipelines produce statistical features over a **30-second sliding window**.

Example feature vector:

```
F = [
EAR_mean,
EAR_std,
Blink_Frequency,
MAR_max,
Head_Pitch,
Lane_Drift_Variance,
Steering_Instability,
Reaction_Delay,
Session_Duration,
Time_of_Day
]
```

Total features: **19**

The feature vector is fed to the machine learning model which outputs:

```
P(fatigue) ∈ [0,1]
```

---

#  Machine Learning Model

Model used:

**LightGBM Gradient Boosting Classifier**

Reasons:

* Excellent performance on tabular features
* Fast inference (<10ms)
* Robust with small datasets
* Easily interpretable

Expected performance targets:

| Metric            | Target |
| ----------------- | ------ |
| AUC               | >0.87  |
| F1 Score          | >0.82  |
| Detection Latency | <4s    |
| Inference Latency | <100ms |

The model outputs a **calibrated fatigue probability score**.

---

#  Fatigue Alert System

Fatigue probability is mapped to four operational states:

| State            | Probability | Response                  |
| ---------------- | ----------- | ------------------------- |
| Normal           | <0.30       | Monitoring only           |
| Mild Fatigue     | 0.30–0.55   | Controller vibration      |
| High Fatigue     | 0.55–0.75   | Sustained warning         |
| Critical Fatigue | >0.75       | Simulation pause + alerts |

Alert channels include:

* 🎮 Controller haptic feedback
* 📱 Smartphone vibration
* 🖥 Visual dashboard warnings

---

#  Tech Stack

### Computer Vision

* OpenCV
* MediaPipe FaceMesh

### Machine Learning

* LightGBM
* Scikit-learn
* NumPy
* Pandas

### Backend

* FastAPI
* WebSockets
* Python

### Frontend

* React
* Recharts
* Three.js driving simulator

### Hardware Integration

* USB Game Controller (haptic alerts)
* Smartphone vibration alerts
* Webcam

---

#  Data Pipeline

```
Webcam (30 FPS)
        │
        ▼
FaceMesh Landmark Detection
        │
        ▼
Vision Feature Extraction
        │
        ▼
Driving Telemetry Stream (10Hz)
        │
        ▼
Sliding Window Aggregation (30s)
        │
        ▼
Feature Fusion
        │
        ▼
LightGBM Inference
        │
        ▼
Fatigue Score
        │
        ▼
Alert Engine
```

---

#  Demo Setup

Required hardware:

* Laptop with webcam
* USB game controller
* Smartphone connected to same WiFi network

Demo flow:

1. Driver starts the simulator
2. Webcam and telemetry streams initialize
3. Fatigue score updates in real time
4. When fatigue increases:

   * Controller vibrates
   * Dashboard warning appears
   * Phone vibrates

---

#  Running Locally

Requires Python 3.12 (MediaPipe doesn't support 3.14 yet) and a webcam.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

**Run it** (one server for everything, port 8000):

```bash
.venv/bin/python backend/main.py
```

Open http://localhost:8000 and follow the three steps: create an account, add an emergency contact, then choose the **camera**, the **driving simulator**, or both. The simulator opens in a new tab. The live fatigue page shows the score, what it is being scored from, and the status of each module.

- **Camera only:** scored from eyes, yawns and head pose (fused model).
- **Simulator only:** scored from lane keeping, steering and reaction time (driving-only model).
- **Both:** fused model; if the camera loses your face, scoring falls back to driving data.

In the simulator, steer with the arrow keys, A/D or a gamepad and counter-steer when a wind gust hits. Press `2` or `3` to let an alert or drowsy autopilot drive. Opening http://localhost:8000/sim/ directly also works: it starts a camera-off session that ends when the tab closes. The telemetry socket also accepts raw 10 Hz samples (`lane_offset`, `steering_angle`, ...) from an external simulator such as the original Godot one.

Useful extras:

- `FATIGUE_SHOW_HUD=0` runs without the OpenCV camera window; `FATIGUE_DEBUG=1` prints feature range warnings.
- `.venv/bin/python scripts/test_stream.py` prints the live score stream; `scripts/test_simulator_connection.py` fakes a simulator.
- `.venv/bin/python backend/ml/train_model.py` retrains both models (fused and driving-only).

---

#  Key Features

* Real-time fatigue probability scoring
* Multimodal signal fusion
* Low-latency ML inference pipeline
* Hardware-integrated alert system
* Real-time fatigue dashboard
* Emergency contact alert capability

---

#  Future Improvements

* Real vehicle telemetry via **OBD-II**
* Edge deployment on **Jetson Nano / Raspberry Pi**
* Temporal models (LSTM / Transformer)
* Fleet-level fatigue analytics
* Wearable physiological sensor integration

---

#  Project Structure

```
backend/
 ├── main.py               FastAPI server: dashboard, simulator, session API, WebSockets
 ├── session_manager.py    session lifecycle, alert levels, CSV logging
 ├── monitor.py            1 Hz scoring loop (camera and/or simulator)
 ├── vision/               webcam, MediaPipe landmarks, EAR/MAR/head pose, 30 s window
 ├── ml/                   feature schema, models, training, EMA smoothing
 └── network/              shared state and telemetry aggregation
dashboard/                 login -> emergency contact -> choose modules -> live score
simulator/index.html       browser driving simulator (telemetry + alerts)
data/training_data.csv     synthetic training data
docs/                      literature review (PDF)
scripts/                   data generators and test clients
```

---

#  Panasonic Equinox'26 Hackathon Timeline

The entire system was built in **60 hours**.

Major milestones included:

* Computer vision pipeline
* Driving simulator telemetry
* Feature extraction
* Machine learning training
* WebSocket integration
* Real-time dashboard
* Hardware alert system

---

#  Team

* Arhaansh Jhingan — Curated the system architecture, the ML and data pipeline, the backend and the integration
* Eklavya Nathani — Created the car simulation engine on godot, designed the frontend and the UI

---

