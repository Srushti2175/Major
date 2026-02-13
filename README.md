# 🏥 Elderly Care AI Monitoring System

A comprehensive AI-powered monitoring system for elderly care, featuring **real-time activity detection**, **fall detection**, **emotion recognition**, and **alerting capabilities**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)

## 🌟 Features

### Activity Detection
- **Standing** - Upright posture detection
- **Sitting** - Seated posture with bent knees
- **Walking** - Movement detection with tracking
- **Lying Down** - Horizontal body position
- **Inactive** - No movement for extended period

### Fall Detection
- **Sudden Drop** - Rapid vertical movement
- **Body Orientation** - Transition to horizontal
- **Velocity Analysis** - Impact detection
- **Multi-frame Confirmation** - Reduces false positives

### Emotion Recognition
- **Happy** 😊 - Positive emotions
- **Sad** 😢 - Low mood detection
- **Angry** 😠 - Frustration detection
- **Fearful** 😨 - Anxiety/fear detection
- **Neutral** 😐 - Calm state
- **Tired** 😴 - Fatigue detection
- **Distressed** 😰 - Combined negative indicators

### Alert System
- 🚨 **Fall Alerts** (Critical) - Immediate notification
- ⏰ **Inactivity Alerts** (High) - Extended no-movement
- 😰 **Distress Alerts** (High) - Emotional distress
- 📍 **Missing Alerts** (Medium) - Person left camera view

## 📁 Project Structure

```
video_model/
├── config/
│   └── config.yaml              # All configurable parameters
├── src/
│   ├── detector.py              # Base pose detection
│   ├── activity_classifier.py   # Activity classification
│   ├── fall_detector.py         # Fall detection
│   ├── emotion_detector.py      # Emotion recognition
│   ├── elderly_care_monitor.py  # Main monitoring system
│   └── utils.py                 # Utility functions
├── dashboard/
│   ├── app.py                   # Flask backend with WebSocket
│   ├── templates/
│   │   └── index.html           # Dashboard HTML template
│   └── static/
│       ├── css/style.css        # Dashboard styling
│       └── js/dashboard.js      # Real-time updates & charts
├── outputs/
│   ├── detections/              # JSON detection data
│   └── logs/                    # Session logs & alerts
├── elderly_care_main.py         # CLI entry point
├── run_dashboard.py             # Web dashboard entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Web Dashboard (Recommended)

```bash
python run_dashboard.py
```
Then open http://localhost:5000 in your browser.

### 3. Run Live Monitoring (CLI)

```bash
python elderly_care_main.py live
```

### 4. Analyze Video File

```bash
python elderly_care_main.py video path/to/video.mp4
```

## 🖥️ Web Dashboard

The system includes a modern web dashboard for real-time monitoring.

### Features

| Component | Description |
|-----------|-------------|
| 📹 **Live Video** | Real-time video feed with pose skeleton overlay |
| 👤 **Person Cards** | Status cards for each detected person |
| 🚨 **Alerts Panel** | Real-time alert notifications with sound |
| 📊 **Activity Chart** | Bar chart showing activity distribution |
| 😊 **Emotion Chart** | Doughnut chart of emotion states |
| 🏃 **Movement Graph** | Line chart tracking movement over time |
| 📈 **Summary Stats** | Quick counts for each activity type |

### Running the Dashboard

```bash
# Start dashboard (opens browser automatically)
python run_dashboard.py

# Custom port
python run_dashboard.py --port 8080

# Without opening browser
python run_dashboard.py --no-browser

# Debug mode
python run_dashboard.py --debug
```

### Dashboard Controls

- **Start/Stop** - Control video monitoring
- **Camera Select** - Choose camera source
- **Clear Alerts** - Clear alert history

---

## 📖 Usage Guide

### Live Webcam Monitoring

```bash
# Basic usage (default webcam)
python elderly_care_main.py live

# Use specific camera
python elderly_care_main.py live --camera 1

# Save output video
python elderly_care_main.py live --save

# Run without display (headless server)
python elderly_care_main.py live --no-display --save
```

### Video File Analysis

```bash
# Basic usage
python elderly_care_main.py video video.mp4

# With JSON output
python elderly_care_main.py video video.mp4 --save-json

# Headless processing
python elderly_care_main.py video video.mp4 --no-display --save-json
```

## ⚙️ Configuration

All settings are in `config/config.yaml`:

### Activity Detection Thresholds

```yaml
activity_detection:
  standing_angle_threshold: 30    # Degrees from vertical
  movement_threshold: 10          # Pixels for movement
  inactivity_duration: 30         # Seconds before inactive
```

### Fall Detection Sensitivity

```yaml
fall_detection:
  velocity_threshold: 150         # Pixels/frame for fall
  height_drop_ratio: 0.4          # 40% height drop
  confirmation_frames: 5          # Frames to confirm
```

### Alert Settings

```yaml
alerts:
  inactivity_threshold: 1800      # 30 minutes
  missing_threshold: 600          # 10 minutes
```

## 🎯 Detection Details

### How Activity Detection Works

| Activity | Detection Method |
|----------|-----------------|
| Standing | Vertical torso angle < 30°, straight legs |
| Sitting | Vertical torso, bent knees (60-130°) |
| Walking | Standing posture + movement > threshold |
| Lying | Horizontal torso angle > 60°, low aspect ratio |
| Inactive | No movement for configured duration |

### How Fall Detection Works

1. **Vertical Velocity** - Tracks rapid downward movement
2. **Height Change** - Monitors body height reduction
3. **Orientation Shift** - Detects transition to horizontal
4. **Multi-frame Confirmation** - Prevents false positives

### How Emotion Detection Works

1. **Face Detection** - Locates face from pose keypoints or Haar cascade
2. **Feature Extraction** - Uses FER (Facial Emotion Recognition)
3. **Classification** - Classifies into 7 emotion categories
4. **Mood Tracking** - Tracks emotional patterns over time

## 📊 Output Format

### Status Output (JSON)

```json
{
  "person_id": 1,
  "timestamp": 1703952000.0,
  "position": {
    "bbox": [100, 50, 300, 450],
    "center": {"x": 200, "y": 250}
  },
  "activity": {
    "type": "sitting",
    "confidence": 0.85,
    "duration_seconds": 120,
    "movement_score": 5.2,
    "is_moving": false
  },
  "fall_detection": {
    "status": "normal",
    "fall_detected": false
  },
  "emotion": {
    "type": "neutral",
    "confidence": 0.72,
    "mood_score": 0.0,
    "is_distressed": false
  },
  "alerts": []
}
```

### Alert Output (JSON)

```json
{
  "type": "fall_detected",
  "severity": "critical",
  "person_id": 1,
  "message": "FALL DETECTED! Person 1 has fallen!",
  "timestamp": 1703952000.0,
  "data": {
    "fall_type": "sudden_drop",
    "position": [200, 250]
  }
}
```

## 🔧 Python API

```python
from src.elderly_care_monitor import ElderlyCareMonitor

# Initialize monitor
monitor = ElderlyCareMonitor()

# Custom alert handler
def my_alert_handler(alert):
    print(f"Alert: {alert.message}")
    # Send SMS, email, etc.

monitor.register_alert_callback(my_alert_handler)

# Process video
for frame, statuses, frame_num in monitor.process_video(0, show=True):
    for status in statuses:
        print(f"Person {status.person_id}: {status.activity}")
        if status.fall_detected:
            print("⚠️ FALL DETECTED!")
```

## 🎮 Keyboard Controls

- **Q** or **ESC** - Quit application

## ⚡ Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available
2. **Model Selection**: Use `yolov8n-pose` for speed, `yolov8x-pose` for accuracy
3. **Resolution**: Lower resolution = faster processing
4. **Frame Skip**: Process every nth frame for faster analysis

## 🔮 Future Enhancements

Based on the project document:
- [ ] Audio emotion detection
- [ ] AI Voice Assistant integration
- [ ] Reminder system
- [ ] Multi-camera support
- [ ] Mobile app dashboard
- [ ] SMS/Email/WhatsApp alerts
- [ ] Cloud synchronization

## 📝 Alert Severity Levels

| Level | Description | Actions |
|-------|-------------|---------|
| 🔴 Critical | Fall detected | Immediate notification, all channels |
| 🟠 High | Inactivity, distress | Push + SMS + Dashboard |
| 🟡 Medium | Missing person, negative emotion | Push + Dashboard |
| 🟢 Low | Minor events | Dashboard only |

## 🐛 Troubleshooting

### FER Model Not Loading
```bash
pip install --upgrade fer tensorflow mtcnn
```

### CUDA Not Available
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Camera Not Found
- Check camera index (try 0, 1, 2...)
- Verify camera permissions

## 📚 Documentation

For detailed technical documentation, see:

- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Complete technical details including:
  - System architecture diagrams
  - ML model explanations (YOLOv8, FER)
  - Algorithm documentation (activity classification, fall detection)
  - API reference
  - Configuration guide
  - Performance benchmarks

---

## 📝 License

Educational project for elderly care monitoring.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FER - Facial Emotion Recognition](https://github.com/justinshenk/fer)
- [OpenCV](https://opencv.org/)
