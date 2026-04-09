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

import cv2
import json
import time
import random
import threading
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from collections import deque

from src.elderly_care_monitor import ElderlyCareMonitor, Alert
from src.utils import load_config
from src.database import (
    get_database, close_database, DatabaseService,
    ActivityRecord, EmotionRecord, AlertRecord, MovementRecord
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
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = 'elderly-care-secret-key'

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
            print("✅ MongoDB connected and session started")
    except Exception as e:
        print(f"⚠️ MongoDB initialization failed: {e}")
        print("⚠️ Continuing without database persistence")
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
            print(f"⚠️ Failed to store alert to MongoDB: {e}")
    
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
        print(f"⚠️ Failed to store status to MongoDB: {e}")


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
    
    print(f"📷 Opening camera source: {source}")
    if sys.platform == 'win32' and source == 0:
        video_capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        video_capture = cv2.VideoCapture(source)
    
    if not video_capture.isOpened():
        print(f"❌ Could not open camera source: {source}")
        is_monitoring = False
        return
    
    # Set camera properties for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    print(f"✅ Camera opened successfully!")
    
    frame_num = 0
    start_time = time.time()
    last_emit_time = 0
    emit_interval = 0.1  # Emit status updates every 100ms for smooth dashboard updates
    
    try:
        while is_monitoring:
            ret, frame = video_capture.read()
            
            if not ret:
                print("⚠️ Failed to read frame from camera")
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
        print(f"❌ Error in frame generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_monitoring = False
        if video_capture:
            video_capture.release()
            print("📷 Camera released")


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
    
    print(f"🎥 Starting video feed from source: {source}")
    
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
            print(f"⚠️ Failed to fetch alerts from MongoDB: {e}")
    
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
            print(f"⚠️ Failed to fetch emotions from MongoDB: {e}")
    
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
            return jsonify(activities)
        except Exception as e:
            print(f"⚠️ Failed to fetch activities from MongoDB: {e}")
    
    return jsonify(list(activity_history))


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
            return jsonify(emotions)
        except Exception as e:
            print(f"⚠️ Failed to fetch emotions from MongoDB: {e}")
    
    return jsonify(list(emotion_history))


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
            print(f"⚠️ Failed to fetch emotion stats from MongoDB: {e}")
    
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
            return jsonify(movements)
        except Exception as e:
            print(f"⚠️ Failed to fetch movements from MongoDB: {e}")
    
    return jsonify([])


@app.route('/api/daily_report')
def get_daily_report():
    """Get daily report summary."""
    if db_service and db_service.enabled:
        try:
            report = db_service.get_daily_report()
            return jsonify(sanitize_for_json(report))
        except Exception as e:
            print(f"⚠️ Failed to generate daily report: {e}")
    
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
            print(f"⚠️ Failed to fetch activity summary: {e}")
    
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
    Simulated audio monitor backend.
    This can later be swapped with a real audio pipeline.
    """
    # Simple rotating demo data to keep UI alive
    emotions = ['calm', 'happy', 'stressed', 'angry', 'worried']
    keywords = [
        'help', 'pain', 'quiet', 'okay', 'doctor',
        'emergency', 'fall', 'nothing'
    ]
    level = random.choice(['normal', 'attention', 'alert'])
    now = time.time()
    
    return jsonify({
        'timestamp': now,
        'emotion': random.choice(emotions),
        'emotion_confidence': round(random.uniform(0.6, 0.98), 2),
        'keywords_detected': random.sample(keywords, k=2),
        'alert_level': level,
        'needs_attention': level != 'normal'
    })


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
    print(f"\n🏥 Elderly Care Dashboard starting...")
    print(f"📍 Open in browser: http://localhost:{port}")
    print(f"📍 Network access: http://{host}:{port}")
    
    init_monitor()
    
    try:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    finally:
        # Cleanup on shutdown
        if db_service:
            db_service.end_session()
            close_database()
            print("📝 Database connection closed")


if __name__ == '__main__':
    run_dashboard(debug=True)
