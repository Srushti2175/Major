# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+ (you have Python 3.12.2)
- Webcam
- MongoDB Atlas account (already configured)

## Installation & Setup

### 1. Fix Jinja2 Issue (IMPORTANT - Do This First!)
```cmd
cd d:\Major\Dashboard\new-dash\Major
venv\Scripts\activate
pip uninstall jinja2 -y
pip install jinja2==3.1.4
```

### 2. Verify Installation
```cmd
python -c "import jinja2; print(jinja2.__version__)"
```
Should output: `3.1.4` without errors

### 3. Run the Dashboard
```cmd
cd backend
python run_dashboard.py
```

## What Happens Next?

1. **Terminal Output**:
   ```
   ELDERLY CARE AI DASHBOARD
   Starting Web Dashboard...
   
   ✓ MongoDB connected and session started
   ✓ Camera opened successfully!
   ✓ Elderly Care Dashboard starting...
   ✓ Open in browser: http://localhost:5000
   ```

2. **Browser Opens Automatically** at `http://localhost:5000`

3. **Dashboard Features**:
   - 📹 Live video feed with pose detection
   - 🏃 Real-time activity detection
   - 😊 Emotion recognition
   - 🚨 Fall detection alerts
   - 📊 Historical charts and analytics

## Dashboard Pages

| URL | Description |
|-----|-------------|
| `http://localhost:5000/` | Home/Landing page |
| `http://localhost:5000/dashboard` | Main monitoring dashboard |
| `http://localhost:5000/video_dashboard` | Video-focused monitoring |
| `http://localhost:5000/audio_dashboard` | Audio monitoring (simulated) |

## Command Options

```cmd
# Run on different port
python run_dashboard.py --port 8080

# Don't open browser automatically
python run_dashboard.py --no-browser

# Run in debug mode
python run_dashboard.py --debug

# Combine options
python run_dashboard.py --port 8080 --no-browser
```

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the server.

## Troubleshooting

### Issue: Jinja2 Error
**Error**: `re.error: bad character range \w-- at position 1`

**Solution**:
```cmd
pip uninstall jinja2 -y
pip install jinja2==3.1.4
```

### Issue: Camera Not Opening
**Error**: `Could not open camera source: 0`

**Solutions**:
1. Check if another application is using the webcam
2. Try a different camera index:
   ```cmd
   # In browser, change URL to:
   http://localhost:5000/video_feed?source=1
   ```
3. Grant camera permissions to Python

### Issue: MongoDB Connection Failed
**Error**: `MongoDB initialization failed`

**Solutions**:
1. Check internet connection (using MongoDB Atlas)
2. Verify `.env` file exists in `backend/` directory
3. Check MongoDB credentials in `.env`

### Issue: Module Not Found
**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```cmd
pip install -r requirements.txt
```

## Project Structure

```
Major/
├── backend/
│   ├── run_dashboard.py    ← START HERE
│   ├── src/                ← Core AI modules
│   ├── dashboard/          ← Flask app
│   └── .env                ← MongoDB config
├── frontend/
│   ├── templates/          ← HTML pages
│   └── static/             ← CSS/JS
└── requirements.txt        ← Dependencies
```

## Key Files

| File | Purpose |
|------|---------|
| `backend/run_dashboard.py` | **Main entry point** - Run this! |
| `backend/.env` | MongoDB credentials |
| `backend/yolov8m-pose.pt` | YOLOv8 pose model |
| `requirements.txt` | Python dependencies |

## Next Steps

1. ✅ Fix Jinja2 (see step 1 above)
2. ✅ Run the dashboard
3. ✅ Test with your webcam
4. ✅ Explore the features
5. ✅ Check MongoDB data in Atlas

## Need Help?

- 📖 Read `README.md` for detailed documentation
- 📁 Check `PROJECT_STRUCTURE.md` for code organization
- 🧹 See `CLEANUP_SUMMARY.md` for what was cleaned

## Common Commands

```cmd
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
cd backend
python run_dashboard.py

# Check Python version
python --version

# Check installed packages
pip list

# Deactivate virtual environment
deactivate
```

---

**Ready to start?** Run these commands:

```cmd
cd d:\Major\Dashboard\new-dash\Major
venv\Scripts\activate
pip install --upgrade jinja2
cd backend
python run_dashboard.py
```

🎉 **Your dashboard should now be running!**
