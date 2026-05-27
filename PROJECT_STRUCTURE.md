# 📁 Elderly Care AI - Project Structure

## Clean Project Organization

```
Major/
├── backend/                          # Backend application
│   ├── config/                       # Configuration files
│   │   └── config.yaml              # System configuration
│   ├── dashboard/                    # Flask web dashboard
│   │   ├── __init__.py
│   │   └── app.py                   # Main dashboard application
│   ├── src/                         # Core AI modules
│   │   ├── __init__.py
│   │   ├── activity_classifier.py   # Activity detection logic
│   │   ├── database.py              # MongoDB integration
│   │   ├── detector.py              # Base pose detector
│   │   ├── elderly_care_monitor.py  # Main monitoring system
│   │   ├── emotion_detector.py      # Emotion recognition
│   │   ├── fall_detector.py         # Fall detection algorithm
│   │   └── utils.py                 # Utility functions
│   ├── uploads/                     # Upload directory (empty)
│   ├── .env                         # Environment variables (MongoDB)
│   ├── run_dashboard.py             # Dashboard entry point
│   └── yolov8m-pose.pt             # YOLOv8 pose model
│
├── frontend/                        # Frontend web interface
│   ├── static/                      # Static assets
│   │   ├── css/
│   │   │   └── style.css           # Dashboard styles
│   │   └── js/
│   │       ├── audio_dashboard.js   # Audio monitoring UI
│   │       ├── dashboard.js         # Main dashboard UI
│   │       └── video_dashboard.js   # Video monitoring UI
│   └── templates/                   # HTML templates
│       ├── audio_dashboard.html     # Audio monitoring page
│       ├── home.html                # Landing page
│       ├── index.html               # Main dashboard
│       └── video_dashboard.html     # Video monitoring page
│
├── venv/                            # Virtual environment (not in git)
├── .gitignore                       # Git ignore rules
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

## 🎯 Entry Points

### Main Application
```bash
python backend/run_dashboard.py
```
- Starts the Flask web dashboard
- Opens browser automatically at http://localhost:5000
- Provides real-time monitoring interface

## 🔧 Core Modules

### Backend (`backend/src/`)

| Module | Purpose |
|--------|---------|
| `elderly_care_monitor.py` | Main monitoring system orchestrator |
| `detector.py` | YOLOv8 pose detection wrapper |
| `activity_classifier.py` | Classifies activities (sitting, standing, etc.) |
| `fall_detector.py` | Multi-indicator fall detection |
| `emotion_detector.py` | DeepFace emotion recognition |
| `database.py` | MongoDB integration and data models |
| `utils.py` | Configuration and utility functions |

### Dashboard (`backend/dashboard/`)

| File | Purpose |
|------|---------|
| `app.py` | Flask application with REST API and WebSocket |
| `run_dashboard.py` | Entry point with CLI arguments |

### Frontend (`frontend/`)

| Component | Purpose |
|-----------|---------|
| `templates/index.html` | Main monitoring dashboard |
| `templates/video_dashboard.html` | Video-focused monitoring |
| `templates/audio_dashboard.html` | Audio monitoring (simulated) |
| `static/js/dashboard.js` | Main dashboard logic |
| `static/css/style.css` | Dashboard styling |

## 🗄️ Database

**MongoDB Atlas** (configured in `backend/.env`)
- Activities collection
- Emotions collection
- Alerts collection
- Movements collection
- Patients collection

## 📦 Dependencies

See `requirements.txt` for full list. Key dependencies:
- **ultralytics** - YOLOv8 pose estimation
- **deepface** - Emotion detection
- **flask** - Web framework
- **flask-socketio** - Real-time updates
- **pymongo** - MongoDB driver
- **opencv-python** - Video processing

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard**
   ```bash
   cd backend
   python run_dashboard.py
   ```

3. **Access the application**
   - Open http://localhost:5000 in your browser
   - Dashboard will automatically connect to your webcam

## 🧹 Cleaned Up Files

The following unnecessary files have been removed:
- ❌ `yolov8m-pose.pt` (root) - Duplicate model file
- ❌ `backend/check_db_fix.py` - Test script
- ❌ `backend/verify_monitor.py` - Test script
- ❌ `backend/main.py` - Old CLI interface
- ❌ `backend/elderly_care_main.py` - Unused CLI version
- ❌ `backend/src/image_detection.py` - Unused module
- ❌ `backend/src/live_detection.py` - Unused module
- ❌ `backend/src/video_detection.py` - Unused module
- ❌ `Elderly_Care_AI_Project_Complete.txt` - Redundant docs
- ❌ `TECHNICAL_DOCUMENTATION.md` - Redundant docs

## 📝 Notes

- The project now has a single, clear entry point: `run_dashboard.py`
- All core functionality is in `backend/src/`
- Dashboard code is in `backend/dashboard/`
- Frontend assets are properly organized in `frontend/`
- No duplicate or test files remain
