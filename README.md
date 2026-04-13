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
driver-fatigue-detection/

backend/
 ├── main.py
 ├── fatigue_pipeline.py
 ├── websocket_server.py
 └── model/

frontend/
 ├── simulator/
 ├── dashboard/
 └── components/

ml/
 ├── feature_extraction.py
 ├── training_pipeline.py
 └── model.pkl

hardware/
 ├── controller_alerts.py
 └── phone_alert_service.py
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

