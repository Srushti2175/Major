# 🏥 Elderly Care AI Monitoring System

> A comprehensive, real-time AI-powered monitoring system designed for elderly care facilities and home care. This system combines advanced computer vision, emotion detection, and fall detection to ensure the safety and well-being of elderly individuals.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-green.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.6%2B-green.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Dashboard Features](#-dashboard-features)
- [API Documentation](#-api-documentation)
- [Detection Algorithms](#-detection-algorithms)
- [Database Schema](#-database-schema)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **Elderly Care AI Monitoring System** is a full-stack application that provides 24/7 automated monitoring of elderly individuals using artificial intelligence. The system analyzes video feeds in real-time to detect activities, emotions, falls, and potential emergencies, alerting caregivers immediately when intervention is needed.

### Problem Statement

Elderly individuals, especially those living alone or in care facilities, face several critical challenges:


| Challenge | Impact | Our Solution |
|-----------|--------|--------------|
| **Falls** | Leading cause of injury and death in elderly | Real-time fall detection with instant alerts |
| **Inactivity** | Can indicate health deterioration | Continuous movement monitoring and inactivity alerts |
| **Emotional Distress** | Affects mental health and quality of life | Emotion tracking and distress detection |
| **Delayed Response** | Medical emergencies go unnoticed | Automated alert system with multiple severity levels |
| **Limited Monitoring** | Caregivers can't be everywhere | 24/7 AI-powered monitoring with video feed |

### Solution

Our system provides:
- **Proactive Monitoring**: Detects issues before they become critical
- **Real-time Alerts**: Immediate notifications for falls, distress, and inactivity
- **Historical Analysis**: Track patterns and trends over time
- **Privacy-Focused**: All processing can be done locally
- **Scalable**: Monitor multiple individuals simultaneously

---

## ✨ Key Features

### 🎥 Real-Time Video Monitoring
- **Live Video Feed**: Stream from webcam or IP cameras
- **Multi-Person Tracking**: Monitor multiple individuals simultaneously
- **Pose Estimation**: 17-keypoint skeleton tracking using YOLOv8-Pose
- **Annotated Video**: Visual overlay showing detected activities and alerts

### 🏃 Activity Detection

- **Standing**: Detects upright posture with straight legs
- **Sitting**: Identifies seated position with bent knees
- **Walking**: Recognizes movement patterns while standing
- **Lying Down**: Detects horizontal body orientation
- **Inactive**: Alerts when no movement detected for extended periods

### 🚨 Fall Detection
- **Multi-Indicator System**: Uses 4 simultaneous indicators for accuracy
  - Sudden vertical drop (>150 pixels/frame)
  - Height collapse (>40% decrease)
  - Orientation change (upright to horizontal)
  - High downward velocity
- **Fall Types**: Identifies sudden drops, forward/backward/side falls
- **Confirmation System**: Requires 5 consecutive frames to reduce false positives
- **Cooldown Period**: Prevents alert spam (5-second cooldown)

### 😊 Emotion Detection
- **7 Emotion Categories**: Happy, Sad, Angry, Fearful, Surprised, Neutral, Tired
- **Distress Detection**: Identifies emotional distress states
- **Mood Tracking**: Calculates mood score (-1 to +1 scale)
- **Trend Analysis**: Tracks mood changes over time
- **Face Detection**: Uses DeepFace with multiple backend options

### 📊 Professional Dashboard

- **Real-Time Updates**: WebSocket-based live status updates
- **Video Stream**: Live annotated video feed
- **Alert Management**: View, acknowledge, and resolve alerts
- **Patient Information**: Manage patient records and emergency contacts
- **Historical Data**: Activity and emotion timelines with Chart.js
- **Statistics**: Daily reports and activity summaries
- **Responsive Design**: Works on desktop, tablet, and mobile

### 💾 MongoDB Integration
- **Persistent Storage**: All data stored in MongoDB for long-term analysis
- **Batch Processing**: Efficient batch writes for performance
- **Historical Queries**: Query activities, emotions, and alerts by time range
- **Patient Records**: Store and manage patient information
- **Movement Tracking**: Detailed position and keypoint history

### 🔔 Alert System
- **Multiple Severity Levels**: Critical, High, Medium, Low
- **Alert Types**:
  - Fall Detected (Critical)
  - Inactivity (High)
  - Emotional Distress (High)
  - Negative Emotion (Medium)
  - Left Camera View (Medium)
- **Cooldown Management**: Prevents alert spam
- **Callback System**: Extensible for SMS, email, push notifications

---

## 🏗️ System Architecture


```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Webcam    │  │ IP Camera   │  │ Video File  │                  │
│  │  (OpenCV)   │  │  (RTSP)     │  │  (MP4/AVI)  │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
└─────────┼────────────────┼────────────────┼─────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AI PROCESSING LAYER                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │              YOLOv8-Pose Estimation                         │     │
│  │  - Input: Video Frame (640x480)                             │     │
│  │  - Output: 17 Keypoints per person                          │     │
│  │  - Bounding Box + Confidence Score                          │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│           ┌──────────────────┼──────────────────┐                   │
│           ▼                  ▼                  ▼                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │    Activity     │ │      Fall       │ │    Emotion      │       │
│  │   Classifier    │ │    Detector     │ │    Detector     │       │
│  │                 │ │                 │ │                 │       │
│  │ - Joint Angles  │ │ - Velocity      │ │ - Face Detect   │       │
│  │ - Body Orient.  │ │ - Height Drop   │ │ - DeepFace      │       │
│  │ - Movement      │ │ - Orientation   │ │ - Mood Track    │       │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘       │
│           │                   │                   │                 │
│           └───────────────────┼───────────────────┘                 │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                   ALERT SYSTEM                              │     │
│  │  - Fall Alerts (Critical)                                   │     │
│  │  - Inactivity Alerts (High)                                 │     │
│  │  - Distress Alerts (High)                                   │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼

┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND SERVER (Flask)                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Flask Application                                          │     │
│  │  - REST API (Historical Data)                               │     │
│  │  - WebSocket (Real-time Updates)                            │     │
│  │  - Video Streaming (MJPEG)                                  │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Database Service (MongoDB)                                 │     │
│  │  - Activity Records                                         │     │
│  │  - Emotion Records                                          │     │
│  │  - Alert Records                                            │     │
│  │  - Movement Records                                         │     │
│  │  - Patient Records                                          │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND DASHBOARD                                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  HTML5 / CSS3 / JavaScript                                  │     │
│  │  - Live Video Feed                                          │     │
│  │  - Real-time Status Cards                                   │     │
│  │  - Alert Management                                         │     │
│  │  - Patient Information Table                                │     │
│  │  - Activity & Emotion Charts (Chart.js)                     │     │
│  │  - Event History Timeline                                   │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Primary programming language |
| **PyTorch** | 2.0+ | Deep learning backend |
| **OpenCV** | 4.8+ | Video processing and computer vision |
| **YOLOv8** | Latest | Pose estimation and object detection |


### AI/ML Libraries

| Library | Purpose |
|---------|---------|
| **Ultralytics** | YOLOv8 implementation |
| **DeepFace** | Facial emotion recognition |
| **TensorFlow/Keras** | DeepFace backend |
| **NumPy** | Numerical computations |

### Web Technologies

| Technology | Purpose |
|------------|---------|
| **Flask** | Web server and REST API |
| **Flask-SocketIO** | Real-time WebSocket communication |
| **Chart.js** | Data visualization |
| **HTML5/CSS3/JavaScript** | Frontend interface |

### Database

| Technology | Purpose |
|------------|---------|
| **MongoDB** | NoSQL database for persistent storage |
| **PyMongo** | Python MongoDB driver |

---

## 🔍 How It Works

### 1. Pose Estimation (YOLOv8-Pose)

The system uses YOLOv8-Pose to extract a 17-point skeleton from each person in the video frame:

```
                    0: Nose
                      │
            ┌─────────┴─────────┐
        1: L_Eye             2: R_Eye
            │                     │
        3: L_Ear             4: R_Ear
            
        5: L_Shoulder ─────── 6: R_Shoulder
            │                     │
        7: L_Elbow           8: R_Elbow
            │                     │
        9: L_Wrist          10: R_Wrist
            
       11: L_Hip ─────────── 12: R_Hip
            │                     │
       13: L_Knee           14: R_Knee
            │                     │
       15: L_Ankle          16: R_Ankle
```


### 2. Activity Classification

The system analyzes geometric relationships between keypoints to determine activity:

#### **Standing Detection**
- Torso is mostly vertical (< 30° from vertical)
- Knees are relatively straight (angle > 150°)
- Body is tall (vertical extent > horizontal)

#### **Sitting Detection**
- Torso is upright (< 45° from vertical)
- Knees are bent (60° < angle < 130°)
- Hips and knees at similar height

#### **Lying Down Detection**
- Torso is mostly horizontal (> 60° from vertical)
- Body width > body height (aspect ratio < 0.5)
- Head and feet at similar Y level

#### **Walking Detection**
- Currently in standing posture
- Movement score > 15 pixels/frame
- Consistent position changes over time

#### **Inactive Detection**
- Movement score ≈ 0
- No significant position changes for > 30 seconds

### 3. Fall Detection Algorithm

The system uses **4 simultaneous indicators** to detect falls with high accuracy:


```
┌─────────────────────────────────────────────────────────────┐
│                    FALL DETECTION LOGIC                      │
│                                                              │
│  Indicator 1: SUDDEN DROP                                    │
│  ──────────────────────────                                  │
│  Body center Y increases rapidly (> 150 pixels)              │
│                                                              │
│  Indicator 2: HEIGHT DROP                                    │
│  ───────────────────────                                     │
│  Body height decreases by > 40%                              │
│                                                              │
│  Indicator 3: ORIENTATION CHANGE                             │
│  ──────────────────────────────                              │
│  Body transitions from upright to horizontal (> 60°)         │
│                                                              │
│  Indicator 4: HIGH VELOCITY                                  │
│  ─────────────────────────                                   │
│  Vertical velocity > 150 pixels/frame                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DECISION LOGIC                            │
│                                                              │
│  if (indicators_triggered >= 2):                            │
│      if (consecutive_frames >= 5):                          │
│          → FALL CONFIRMED (Critical Alert)                  │
│      else:                                                   │
│          → FALL SUSPECTED                                    │
│  else:                                                       │
│      → NORMAL                                                │
└─────────────────────────────────────────────────────────────┘
```

### 4. Emotion Detection

The system uses DeepFace to analyze facial expressions:

1. **Face Localization**: Estimates face region from pose keypoints
2. **Preprocessing**: Resize to 48x48, convert to grayscale, normalize
3. **Emotion Classification**: 7 emotion categories with confidence scores
4. **Mood Tracking**: Calculates mood score (-1 to +1) and tracks trends
5. **Distress Detection**: Identifies combined negative emotions


**Emotion Categories & Mood Scores:**

| Emotion | Mood Score | Alert Level |
|---------|------------|-------------|
| 😊 Happy | +1.0 | None |
| 😮 Surprised | +0.3 | Low |
| 😐 Neutral | 0.0 | None |
| 😴 Tired | -0.2 | Low |
| 😢 Sad | -0.6 | Medium |
| 😠 Angry | -0.7 | Medium |
| 😨 Fearful | -0.8 | High |
| 😰 Distressed | -1.0 | High |

---

## 📦 Installation

### Prerequisites

- **Python 3.8 or higher**
- **MongoDB** (Local or Atlas)
- **Webcam or IP Camera**
- **GPU (Optional)**: NVIDIA GPU with CUDA for better performance

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Major
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard

```bash
cd backend
python run_dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:5000`

**Command Options:**
```bash
python run_dashboard.py --port 5000          # Specify port (default: 5000)
python run_dashboard.py --no-browser         # Don't auto-open browser
python run_dashboard.py --debug              # Run in debug mode
```

