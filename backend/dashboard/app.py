"""
Elderly Care AI Dashboard - Flask Backend with MongoDB Integration

Provides:
- Real-time video streaming with detections from laptop camera
- WebSocket for live status updates (concurrent with video)
- MongoDB storage for all activities, emotions, and alerts
- REST API for historical data
- Alert management with persistence
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables FIRST
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"✓ Loaded environment variables from {env_path}")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠ Failed to load .env file: {e}")

import cv2
import threading
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from moviepy.editor import VideoFileClip
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from src.elderly_care_monitor import ElderlyCareMonitor, Alert
from src.utils import load_config
from src.audio_yamnet import predict_audio_events
from src.audio_alerts import map_yamnet_to_level
from src.database import (
    get_database, close_database, DatabaseService,
    ActivityRecord, EmotionRecord, AlertRecord, MovementRecord,
    PatientRecord, VideoRecord
)


def sanitize_for_json(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable types to Python native types.
    This ensures all data can be safely sent via WebSocket.
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    else:
        # For any other type, try to convert to string
        try:
            return str(obj)
        except:
            return None


# Initialize Flask app
app = Flask(__name__, 
            template_folder='../../frontend/templates',
            static_folder='../../frontend/static')
app.config['SECRET_KEY'] = 'elderly-care-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent.parent / 'uploads')

# Add CORS headers for all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', max_http_buffer_size=500 * 1024 * 1024)

# Global state
monitor = None
db_service: DatabaseService = None
is_monitoring = False
video_capture = None
current_frame = None
frame_lock = threading.Lock()
status_lock = threading.Lock()

# In-memory cache (for fast access, MongoDB is the source of truth)
activity_history = deque(maxlen=100)
emotion_history = deque(maxlen=100)
alerts_history = deque(maxlen=50)
current_statuses = {}

# Configuration
config = None

# Default camera source (0 = laptop webcam)
CAMERA_SOURCE = 0

# Frame counter for movement storage (store every Nth frame to reduce DB load)
frame_counter = 0
MOVEMENT_STORE_INTERVAL = 5  # Store movement every 5 frames


def init_monitor():
    """Initialize the monitoring system with MongoDB."""
    global monitor, config, db_service
    
    config = load_config()
    
    # Initialize database service
    try:
        db_service = get_database(config)
        if db_service.enabled:
            db_service.start_session()
            print(" MongoDB connected and session started")
    except Exception as e:
        print(f" MongoDB initialization failed: {e}")
        print(" Continuing without database persistence")
        db_service = None
    
    # Initialize monitor
    monitor = ElderlyCareMonitor()
    
    # Register alert callback
    def alert_handler(alert: Alert):
        handle_alert(alert)
    
    monitor.register_alert_callback(alert_handler)
    return monitor


def handle_alert(alert: Alert):
    """Handle alert: store to MongoDB and emit via WebSocket."""
    global db_service
    
    # Sanitize alert data for JSON serialization
    alert_data = sanitize_for_json(alert.to_dict())
    
    # Add to in-memory cache
    alerts_history.append(alert_data)
    
    # Store to MongoDB
    if db_service and db_service.enabled:
        try:
            alert_record = AlertRecord(
                person_id=alert.person_id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                message=alert.message,
                data=sanitize_for_json(alert.data),
                timestamp=datetime.fromtimestamp(alert.timestamp)
            )
            db_service.store_alert(alert_record)
        except Exception as e:
            print(f" Failed to store alert to MongoDB: {e}")
    
    # Emit via WebSocket
    socketio.emit('new_alert', alert_data)


def store_status_to_db(status, frame_num: int):
    """Store status data to MongoDB concurrently."""
    global db_service, frame_counter
    
    if not db_service or not db_service.enabled:
        return
    
    try:
        timestamp = datetime.fromtimestamp(status.timestamp)
        
        # Store activity record (every frame for accurate tracking)
        activity_record = ActivityRecord(
            person_id=int(status.person_id),
            activity_type=str(status.activity),
            confidence=float(status.activity_confidence),
            movement_score=float(status.movement_score),
            is_moving=bool(status.is_moving),
            duration_seconds=float(status.activity_duration),
            timestamp=timestamp
        )
        db_service.store_activity(activity_record)
        
        # Store emotion record (every frame for accurate tracking)
        all_scores = {}
        if hasattr(status, 'emotion_scores'):
            all_scores = status.emotion_scores
        
        emotion_record = EmotionRecord(
            person_id=int(status.person_id),
            emotion_type=str(status.emotion),
            confidence=float(status.emotion_confidence),
            mood_score=float(status.mood_score),
            is_distressed=bool(status.is_distressed),
            all_scores=sanitize_for_json(all_scores),
            timestamp=timestamp
        )
        db_service.store_emotion(emotion_record)
        
        # Store movement record (every Nth frame to reduce DB load)
        if frame_num % MOVEMENT_STORE_INTERVAL == 0:
            keypoints_dict = sanitize_for_json(status.keypoints) if status.keypoints else {}
            
            movement_record = MovementRecord(
                person_id=int(status.person_id),
                position_x=float(status.center[0]),
                position_y=float(status.center[1]),
                bbox=[float(v) for v in status.bbox],
                keypoints=keypoints_dict,
                activity=str(status.activity),
                emotion=str(status.emotion),
                fall_status=str(status.fall_status),
                movement_score=float(status.movement_score),
                timestamp=timestamp,
                frame_number=frame_num
            )
            db_service.store_movement(movement_record)
            
    except Exception as e:
        print(f" Failed to store status to MongoDB: {e}")


def emit_status_update(statuses, frame_num):
    """Emit status update to all connected clients and store to MongoDB."""
    try:
        status_data = []
        for status in statuses:
            # Convert to dict and sanitize for JSON serialization
            status_dict = sanitize_for_json(status.to_dict())
            status_data.append(status_dict)
            
            with status_lock:
                current_statuses[status.person_id] = status_dict
            
            # Add to in-memory cache
            activity_history.append({
                'timestamp': float(status.timestamp),
                'person_id': int(status.person_id),
                'activity': str(status.activity),
                'confidence': float(status.activity_confidence)
            })
            
            emotion_history.append({
                'timestamp': float(status.timestamp),
                'person_id': int(status.person_id),
                'emotion': str(status.emotion),
                'confidence': float(status.emotion_confidence),
                'mood_score': float(status.mood_score)
            })
            
            # Store to MongoDB concurrently (in background thread via queue)
            store_status_to_db(status, frame_num)
        
        # Emit status update via WebSocket - sanitize all data
        update_data = sanitize_for_json({
            'frame_num': frame_num,
            'statuses': status_data,
            'alert_count': len(alerts_history),
            'timestamp': time.time()
        })
        socketio.emit('status_update', update_data)
    except Exception as e:
        print(f"Error emitting status: {e}")
        import traceback
        traceback.print_exc()


def generate_frames(source=0):
    """Generate video frames with detections from webcam."""
    global current_frame, is_monitoring, video_capture, frame_counter
    
    if monitor is None:
        init_monitor()
    
    is_monitoring = True
    frame_counter = 0
    
    # Use webcam (0 = default laptop camera)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    print(f" Opening camera source: {source}")
    if sys.platform == 'win32' and source == 0:
        video_capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        video_capture = cv2.VideoCapture(source)
    
    if not video_capture.isOpened():
        print(f" Could not open camera source: {source}")
        is_monitoring = False
        return
    
    # Set camera properties for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    print(f" Camera opened successfully!")
    
    frame_num = 0
    start_time = time.time()
    last_emit_time = 0
    emit_interval = 0.1  # Emit status updates every 100ms for smooth dashboard updates
    
    try:
        while is_monitoring:
            ret, frame = video_capture.read()
            
            if not ret:
                print(" Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Process frame with AI
            timestamp = start_time + frame_num / 30.0
            annotated_frame, statuses = monitor.process_frame(frame, timestamp)
            
            # Update current frame (for other endpoints)
            with frame_lock:
                current_frame = annotated_frame.copy()
            
            # Emit status updates at regular intervals (concurrent with video)
            current_time = time.time()
            if current_time - last_emit_time >= emit_interval:
                emit_status_update(statuses, frame_num)
                last_emit_time = current_time
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            frame_num += 1
            frame_counter = frame_num
    
    except Exception as e:
        print(f" Error in frame generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_monitoring = False
        if video_capture:
            video_capture.release()
            print(" Camera released")


# Routes
@app.route('/')
def home():
    """Informational landing page."""
    return render_template('home.html')


@app.route('/dashboard')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route - uses laptop webcam by default."""
    source = request.args.get('source', CAMERA_SOURCE)
    try:
        source = int(source)
    except ValueError:
        pass  # Keep as string (file path)
    
    print(f" Starting video feed from source: {source}")
    
    return Response(
        generate_frames(source),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )


@app.route('/api/status')
def get_status():
    """Get current monitoring status."""
    return jsonify({
        'is_monitoring': is_monitoring,
        'persons': current_statuses,
        'alert_count': len(alerts_history),
        'db_connected': db_service.enabled if db_service else False
    })


@app.route('/api/alerts')
def get_alerts():
    """Get alert history (from MongoDB if available, otherwise from memory)."""
    if db_service and db_service.enabled:
        try:
            alerts = db_service.get_alerts(limit=50)
            # Convert ObjectId and datetime to JSON-serializable format
            for alert in alerts:
                if '_id' in alert:
                    alert['_id'] = str(alert['_id'])
                # Convert datetime to Unix timestamp
                if 'timestamp' in alert:
                    if hasattr(alert['timestamp'], 'timestamp'):
                        alert['timestamp'] = alert['timestamp'].timestamp()
                    elif not isinstance(alert['timestamp'], (int, float)):
                        alert['timestamp'] = time.time()
            return jsonify(alerts)
        except Exception as e:
            print(f" Failed to fetch alerts from MongoDB: {e}")
    
    return jsonify(list(alerts_history))


@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts from memory (MongoDB alerts are kept for history)."""
    alerts_history.clear()
    return jsonify({'success': True})


@app.route('/api/emotions')
def get_emotions():
    """Get recent emotion history from MongoDB."""
    if db_service and db_service.enabled:
        try:
            emotions = db_service.get_emotions(limit=30)
            result = []
            for e in emotions:
                emotion_record = {
                    'personId': e.get('person_id', 0),
                    'emotion': e.get('emotion_type', 'unknown'),
                    'confidence': e.get('confidence', 0),
                    'timestamp': e.get('timestamp', time.time())
                }
                # Convert datetime to Unix timestamp
                if hasattr(emotion_record['timestamp'], 'timestamp'):
                    emotion_record['timestamp'] = emotion_record['timestamp'].timestamp()
                result.append(emotion_record)
            return jsonify(result)
        except Exception as e:
            print(f" Failed to fetch emotions from MongoDB: {e}")
    
    return jsonify([])


@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    if db_service and db_service.enabled:
        success = db_service.acknowledge_alert(alert_id)
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'Database not available'})


@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolve an alert."""
    if db_service and db_service.enabled:
        success = db_service.resolve_alert(alert_id)
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'Database not available'})


@app.route('/api/activity_history')
def get_activity_history():
    """Get activity history (from MongoDB if available)."""
    person_id = request.args.get('person_id', type=int)
    limit = request.args.get('limit', 100, type=int)
    hours = request.args.get('hours', 24, type=int)
    
    if db_service and db_service.enabled:
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            activities = db_service.get_activities(
                person_id=person_id,
                start_time=start_time,
                limit=limit
            )
            # Convert ObjectId to string
            for activity in activities:
                if '_id' in activity:
                    activity['_id'] = str(activity['_id'])
            return jsonify(sanitize_for_json(activities))
        except Exception as e:
            print(f"Failed to fetch activities from MongoDB: {e}")
    
    return jsonify(sanitize_for_json(list(activity_history)))


@app.route('/api/emotion_history')
def get_emotion_history():
    """Get emotion history (from MongoDB if available)."""
    person_id = request.args.get('person_id', type=int)
    limit = request.args.get('limit', 100, type=int)
    hours = request.args.get('hours', 24, type=int)
    
    if db_service and db_service.enabled:
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            emotions = db_service.get_emotions(
                person_id=person_id,
                start_time=start_time,
                limit=limit
            )
            # Convert ObjectId to string
            for emotion in emotions:
                if '_id' in emotion:
                    emotion['_id'] = str(emotion['_id'])
            return jsonify(sanitize_for_json(emotions))
        except Exception as e:
            print(f"Failed to fetch emotions from MongoDB: {e}")
    
    return jsonify(sanitize_for_json(list(emotion_history)))


@app.route('/api/emotion_stats')
def get_emotion_stats():
    """Get emotion statistics."""
    person_id = request.args.get('person_id', type=int)
    hours = request.args.get('hours', 24, type=int)
    
    if db_service and db_service.enabled:
        try:
            stats = db_service.get_emotion_stats(person_id=person_id, hours=hours)
            return jsonify(sanitize_for_json(stats))
        except Exception as e:
            print(f"Failed to fetch emotion stats from MongoDB: {e}")
    
    return jsonify({})


@app.route('/api/movements')
def get_movements():
    """Get movement history for detailed tracking."""
    person_id = request.args.get('person_id', type=int)
    limit = request.args.get('limit', 500, type=int)
    hours = request.args.get('hours', 1, type=int)
    
    if db_service and db_service.enabled:
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            movements = db_service.get_movements(
                person_id=person_id,
                start_time=start_time,
                limit=limit
            )
            # Convert ObjectId to string
            for movement in movements:
                if '_id' in movement:
                    movement['_id'] = str(movement['_id'])
            return jsonify(sanitize_for_json(movements))
        except Exception as e:
            print(f"Failed to fetch movements from MongoDB: {e}")
    
    return jsonify([])


@app.route('/api/daily_report')
def get_daily_report():
    """Get daily report summary."""
    if db_service and db_service.enabled:
        try:
            report = db_service.get_daily_report()
            return jsonify(sanitize_for_json(report))
        except Exception as e:
            print(f" Failed to generate daily report: {e}")
    
    return jsonify({})


@app.route('/api/activity_summary')
def get_activity_summary():
    """Get activity summary."""
    hours = request.args.get('hours', 24, type=int)
    
    if db_service and db_service.enabled:
        try:
            summary = db_service.get_activity_summary(hours=hours)
            return jsonify(sanitize_for_json(summary))
        except Exception as e:
            print(f" Failed to fetch activity summary: {e}")
    
    return jsonify({})


@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start monitoring."""
    global is_monitoring
    if not is_monitoring:
        is_monitoring = True
        return jsonify({'success': True, 'message': 'Monitoring started'})
    return jsonify({'success': False, 'message': 'Already monitoring'})


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop monitoring and release camera."""
    global is_monitoring, video_capture
    is_monitoring = False
    
    # Release camera if open
    if video_capture is not None:
        try:
            video_capture.release()
        except:
            pass
        video_capture = None
    
    return jsonify({'success': True, 'message': 'Monitoring stopped'})


@app.route('/api/camera/status')
def camera_status():
    """Get camera status."""
    global video_capture, is_monitoring
    is_open = video_capture is not None and video_capture.isOpened() if video_capture else False
    return jsonify({
        'is_monitoring': is_monitoring,
        'camera_open': is_open,
        'source': CAMERA_SOURCE,
        'db_connected': db_service.enabled if db_service else False
    })


@app.route('/api/patients')
def get_patients():
    """Get all patient info from MongoDB."""
    if db_service and db_service.enabled:
        try:
            patients = db_service.get_patients()
            # If no patients, add some demo data
            if not patients:
                demo_patients = [
                    PatientRecord(1, "John Doe", 75, "Male", ["Hypertension"], "555-0101", "Room 101"),
                    PatientRecord(2, "Jane Smith", 82, "Female", ["Arthritis"], "555-0102", "Room 102"),
                    PatientRecord(3, "Robert Brown", 69, "Male", ["Diabetes"], "555-0103", "Room 103")
                ]
                for p in demo_patients:
                    db_service.store_patient(p)
                patients = db_service.get_patients()
            
            # Sanitize for JSON
            for p in patients:
                if '_id' in p:
                    p['_id'] = str(p['_id'])
                if 'created_at' in p and hasattr(p['created_at'], 'isoformat'):
                    p['created_at'] = p['created_at'].isoformat()
            
            return jsonify(patients)
        except Exception as e:
            print(f" Failed to fetch patients: {e}")
    
    return jsonify([])


@app.route('/api/history')
def get_full_history():
    """Get combined history of activities and alerts."""
    limit = request.args.get('limit', 50, type=int)
    results = []
    
    if db_service and db_service.enabled:
        activities = db_service.get_activities(limit=limit)
        alerts = db_service.get_alerts(limit=limit)
        
        for a in activities:
            results.append({
                'type': 'activity',
                'person_id': a.get('person_id'),
                'event': a.get('activity_type'),
                'subtext': f"Confidence: {int(a.get('confidence', 0)*100)}%",
                'timestamp': a.get('timestamp').timestamp() if hasattr(a.get('timestamp'), 'timestamp') else time.time()
            })
            
        for a in alerts:
            results.append({
                'type': 'alert',
                'person_id': a.get('person_id'),
                'event': a.get('alert_type'),
                'subtext': a.get('message'),
                'timestamp': a.get('timestamp').timestamp() if hasattr(a.get('timestamp'), 'timestamp') else time.time()
            })
            
        # Sort by timestamp descending
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(results[:limit])
        
    return jsonify([])


@app.route('/api/config')
def get_config():
    """Get current configuration."""
    return jsonify(config if config else {})


@app.route('/api/summary')
def get_summary():
    """Get monitoring summary."""
    if monitor:
        summary = monitor.get_monitoring_summary()
        
        # Add database stats if available
        if db_service and db_service.enabled:
            try:
                summary['db_stats'] = {
                    'activities_today': len(db_service.get_activities(limit=1000)),
                    'emotions_today': len(db_service.get_emotions(limit=1000)),
                    'alerts_today': len(db_service.get_alerts(limit=1000))
                }
            except:
                pass
        
        return jsonify(sanitize_for_json(summary))
    return jsonify({})


@app.route('/api/db/status')
def db_status():
    """Get database connection status."""
    if db_service:
        return jsonify({
            'connected': db_service.enabled,
            'database_name': db_service.database_name if db_service.enabled else None,
            'connection_string': db_service.connection_string if db_service.enabled else None
        })
    return jsonify({'connected': False})


@app.route('/api/audio_status')
def audio_status():
    """
    Real-time audio status endpoint powered by YAMNet (TensorFlow Hub).

    Flow
    ----
    1. Capture 1 second of audio from the default microphone.
    2. Run YAMNet inference → returns the most likely sound-event class.
    3. Map that class to an alert level (normal / attention / alert).
    4. Return the result as JSON for the dashboard UI.
    """
    try:
        import sounddevice as sd
        from src.audio_yamnet import predict_audio_events
        from src.audio_alerts import map_yamnet_to_level

        # ---- 1️⃣  Capture a short audio snippet from the mic ----------
        DURATION = 1.0          # seconds
        SAMPLE_RATE = 16000     # YAMNet's native rate — avoids resampling
        audio_chunk = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()               # block until recording finishes

        waveform = audio_chunk.squeeze()   # shape → (16000,)

        # ---- 2️⃣  Run YAMNet inference --------------------------------
        yamnet_result = predict_audio_events(waveform, sample_rate=SAMPLE_RATE)

        # ---- 3️⃣  Map the top class to an alert level -----------------
        level = map_yamnet_to_level(yamnet_result["top_class"])

        # ---- 4️⃣  Build the JSON response -----------------------------
        now = time.time()
        return jsonify({
            "timestamp": now,
            "emotion": yamnet_result["top_class"],
            "emotion_confidence": yamnet_result["top_score"],
            "audio_events": yamnet_result.get("all_events", [yamnet_result["top_class"]]),
            "yamnet_scores": yamnet_result["top_dict"],
            "alert_level": level,
            "needs_attention": level != "normal",
        })

    except ImportError as exc:
        # Graceful fallback if sounddevice or TF is not installed
        return jsonify({
            "timestamp": time.time(),
            "error": f"Missing dependency: {exc}",
            "alert_level": "normal",
            "needs_attention": False,
        }), 503

    except Exception as exc:
        # Catch-all so the dashboard never crashes
        print(f"⚠  audio_status error: {exc}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "timestamp": time.time(),
            "error": str(exc),
            "alert_level": "normal",
            "needs_attention": False,
        }), 500


@app.route('/video_dashboard')
def video_dashboard():
    """
    Dedicated dashboard focused on video-captured information
    (movement timelines, activity history, and event list).
    """
    return render_template('video_dashboard.html')


@app.route('/audio_dashboard')
def audio_dashboard():
    """
    Dedicated dashboard for audio monitoring only.
    Shows live audio emotion and keyword-based alerts.
    """
    return render_template('audio_dashboard.html')


# ==================== Video Testing Routes ====================

# Upload configuration
UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/testing')
def testing():
    """Video testing page."""
    return render_template('testing.html')



@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload."""
    try:
        print("Upload request received")
        
        if 'video' not in request.files:
            print("No video in request.files")
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        file = request.files['video']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
        
        # Secure filename
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{original_filename}"
        filepath = UPLOAD_FOLDER / filename
        
        print(f"Saving to: {filepath}")
        
        # Save file
        file.save(str(filepath))
        file_size = filepath.stat().st_size
        
        print(f"File saved successfully. Size: {file_size} bytes")
        
        # Get video duration
        duration = None
        try:
            cap = cv2.VideoCapture(str(filepath))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration = frame_count / fps
            cap.release()
            print(f"Video duration: {duration}s")
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
        
        # Store in database
        video_id = None
        if db_service and db_service.enabled:
            try:
                video_record = VideoRecord(
                    filename=filename,
                    original_filename=original_filename,
                    file_size=file_size,
                    duration=duration,
                    upload_timestamp=datetime.utcnow(),
                    status="uploaded"
                )
                video_id = db_service.store_video(video_record)
                print(f"Video record stored in database: {video_id}")
            except Exception as db_error:
                print(f"Database storage failed: {db_error}")
                # Continue anyway - file is uploaded
        
        return jsonify({
            'success': True,
            'video_id': video_id or 'no_db',
            'filename': filename,
            'original_filename': original_filename,
            'file_size': file_size,
            'duration': duration
        })
            
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/videos')
def get_uploaded_videos():
    """Get list of uploaded videos."""
    try:
        if db_service and db_service.enabled:
            videos = db_service.get_videos(limit=100)
            # Convert ObjectId to string
            for video in videos:
                if '_id' in video:
                    video['_id'] = str(video['_id'])
                # Convert datetime to ISO format
                if 'upload_timestamp' in video and hasattr(video['upload_timestamp'], 'isoformat'):
                    video['upload_timestamp'] = video['upload_timestamp'].isoformat()
                if 'created_at' in video and hasattr(video['created_at'], 'isoformat'):
                    video['created_at'] = video['created_at'].isoformat()
            return jsonify(sanitize_for_json(videos))
        else:
            # If database not available, scan uploads folder
            videos = []
            if UPLOAD_FOLDER.exists():
                for video_file in UPLOAD_FOLDER.glob('*.*'):
                    if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                        videos.append({
                            '_id': video_file.stem,
                            'filename': video_file.name,
                            'original_filename': video_file.name,
                            'file_size': video_file.stat().st_size,
                            'upload_timestamp': datetime.fromtimestamp(video_file.stat().st_mtime).isoformat(),
                            'status': 'uploaded',
                            'duration': None
                        })
            return jsonify(videos)
    except Exception as e:
        print(f"Error getting videos: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([]), 200


@app.route('/api/analyze_video/<video_id>', methods=['POST'])
def analyze_video(video_id):
    """Analyze uploaded video."""
    global db_service
    # If the database wasn't connected at startup, try to initialize again
    if db_service is None or not db_service.enabled:
        init_monitor()
        
    if not db_service or not db_service.enabled:
        return jsonify({'success': False, 'error': 'Database not available. Please ensure your MongoDB is running locally.'}), 500
    
    try:
        # Get video record
        video = db_service.get_video(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Update status to processing
        db_service.update_video_status(video_id, 'processing')
        
        # Start analysis in background thread
        def analyze():
            cap = None
            try:
                filepath = UPLOAD_FOLDER / video['filename']
                if not filepath.exists():
                    db_service.update_video_status(video_id, 'failed', {'error': 'File not found'})
                    return
                
                # Initialize monitor if not already done
                if monitor is None:
                    init_monitor()
                
                # Analyze video and audio
                cap = cv2.VideoCapture(str(filepath))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                
                # Audio Extraction using MoviePy
                try:
                    clip = VideoFileClip(str(filepath))
                    audio = clip.audio
                    if audio:
                        fps_audio = audio.fps
                        audio_array = audio.to_soundarray()
                        has_audio = True
                        audio.close()
                    else:
                        has_audio = False
                        audio_array = None
                        fps_audio = 44100
                    clip.close()
                except Exception as e:
                    print(f"Failed to extract audio: {e}")
                    has_audio = False
                    audio_array = None
                    fps_audio = 44100
                
                activities_count = {}
                emotions_count = {}
                alerts_count = 0
                alert_reasons = set()
                frame_num = 0
                
                last_emit_time = 0
                emit_interval = 0.1  # Update dashboard every 100ms
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    timestamp = frame_num / fps
                    _, statuses = monitor.process_frame(frame, timestamp)
                    
                    for status in statuses:
                        # Count activities
                        activity = status.activity
                        activities_count[activity] = activities_count.get(activity, 0) + 1
                        
                        # Count emotions
                        emotion = status.emotion
                        emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
                        
                        # Count alerts
                        if status.fall_status != 'normal':
                            alerts_count += 1
                    
                    # Audio Analysis for this frame chunk (evaluating 1-second chunks every 1 second to save CPU)
                    audio_level = 'normal'
                    yamnet_label = None
                    yamnet_score = 0.0
                    
                    if has_audio and audio_array is not None:
                        # Process audio strictly once per second to prevent heavy TensorFlow slowdowns
                        if frame_num % int(fps) == 0:
                            start_idx = int(timestamp * fps_audio)
                            end_idx = int((timestamp + 1.0) * fps_audio)
                            chunk = audio_array[start_idx:end_idx]
                            
                            if len(chunk) > 0:
                                try:
                                    res = predict_audio_events(chunk, sample_rate=int(fps_audio))
                                    yamnet_label = res["top_class"]
                                    yamnet_score = res["top_score"]
                                    audio_level = map_yamnet_to_level(res["top_dict"])
                                except Exception as e:
                                    print(f"YAMNet Error: {e}")
                                    audio_level = 'normal'
                    
                    # Combined Alert Logic
                    for status in statuses:
                        if status.fall_status != 'normal' and audio_level in ['alert', 'critical']:
                            alert_reasons.add(f"Video: {status.fall_status} + Audio: {yamnet_label}")
                            # Trigger a combined critical alert via WebSocket
                            socketio.emit('alert', {
                                'type': 'CRITICAL_COMBINED',
                                'message': f'CRITICAL: Fall detected with {yamnet_label} (Confidence: {yamnet_score:.2f}) for person {status.person_id}',
                                'timestamp': current_time,
                                'person_id': status.person_id
                            })
                        elif status.fall_status != 'normal':
                            alert_reasons.add(f"Video: {status.fall_status}")
                        elif audio_level == 'alert':
                            alert_reasons.add(f"Audio: {yamnet_label}")
                            # Just loud noise/alarms
                            socketio.emit('alert', {
                                'type': 'AUDIO_ALERT',
                                'message': f'ATTENTION: {yamnet_label} detected (Confidence: {yamnet_score:.2f})',
                                'timestamp': current_time
                            })
                    
                    # Emit status update so the dashboard shows real-time changes
                    current_time = time.time()
                    if current_time - last_emit_time >= emit_interval:
                        emit_status_update(statuses, frame_num)
                        last_emit_time = current_time
                        
                        # Emit real audio analysis update
                        if has_audio and yamnet_label is not None:
                            socketio.emit('audio_status_update', {
                                'timestamp': current_time,
                                'emotion': yamnet_label,
                                'emotion_confidence': yamnet_score,
                                'keywords_detected': [yamnet_label],
                                'alert_level': audio_level,
                                'needs_attention': audio_level != 'normal',
                                'rms_volume': yamnet_score # Pass confidence in place of RMS for dashboard visualization
                            })
                    
                    frame_num += 1
                    
                    # Sleep to simulate real-time playback for the dashboard
                    time.sleep(1.0 / fps)
                
                cap.release()
                
                # Store analysis results
                analysis_results = {
                    'total_frames': total_frames,
                    'analyzed_frames': frame_num,
                    'activities': activities_count,
                    'emotions': emotions_count,
                    'alerts_detected': alerts_count,
                    'alert_reasons': list(alert_reasons),
                    'analyzed_at': datetime.utcnow().isoformat()
                }
                
                db_service.update_video_status(video_id, 'completed', analysis_results)
                
            except Exception as e:
                db_service.update_video_status(video_id, 'failed', {'error': str(e)})
            finally:
                if cap is not None:
                    cap.release()

        
        # Start analysis thread
        threading.Thread(target=analyze, daemon=True).start()
        
        return jsonify({'success': True, 'message': 'Analysis started'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/video/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a video and its records."""
    global db_service
    if db_service is None or not db_service.enabled:
        return jsonify({'success': False, 'error': 'Database not available'}), 500
        
    try:
        video = db_service.get_video(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
            
        # Delete file
        filepath = UPLOAD_FOLDER / video['filename']
        if filepath.exists():
            filepath.unlink()
            
        # Delete record
        db_service.delete_video(video_id)
        
        return jsonify({'success': True, 'message': 'Video deleted successfully'})
    except Exception as e:
        print(f"Error deleting video: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/video_status/<video_id>')
def get_video_status(video_id):
    """Get video analysis status."""
    if not db_service or not db_service.enabled:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        video = db_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Convert ObjectId and datetime
        if '_id' in video:
            video['_id'] = str(video['_id'])
        if 'upload_timestamp' in video and hasattr(video['upload_timestamp'], 'isoformat'):
            video['upload_timestamp'] = video['upload_timestamp'].isoformat()
        if 'created_at' in video and hasattr(video['created_at'], 'isoformat'):
            video['created_at'] = video['created_at'].isoformat()
        
        return jsonify(sanitize_for_json(video))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {
        'status': 'Connected to Elderly Care Dashboard',
        'db_connected': db_service.enabled if db_service else False
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client."""
    emit('status_update', {
        'statuses': list(current_statuses.values()),
        'alert_count': len(alerts_history),
        'db_connected': db_service.enabled if db_service else False
    })


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run the dashboard server."""
    print(f"\n Elderly Care Dashboard starting...")
    print(f" Open in browser: http://localhost:{port}")
    print(f" Network access: http://{host}:{port}")
    
    init_monitor()
    
    try:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    finally:
        # Cleanup on shutdown
        if db_service:
            db_service.end_session()
            close_database()
            print(" Database connection closed")


if __name__ == '__main__':
    run_dashboard(debug=True)
