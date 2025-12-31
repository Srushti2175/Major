"""
Elderly Care Monitoring System

Integrates all detection modules for comprehensive elderly monitoring:
- Human position detection (pose estimation)
- Activity classification (sitting, standing, walking, lying)
- Fall detection
- Emotion detection
- Inactivity monitoring
- Alert generation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Generator, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import json

from ultralytics import YOLO

from .utils import load_config, get_device, format_keypoints, setup_output_dirs
from .activity_classifier import ActivityClassifier, ActivityType, ActivityState
from .fall_detector import FallDetector, FallStatus, FallEvent
from .emotion_detector import EmotionDetector, EmotionType, EmotionResult


@dataclass
class ElderlyStatus:
    """Complete status for a monitored elderly person."""
    person_id: int
    timestamp: float
    
    # Position & Pose
    bbox: List[float]
    center: Tuple[float, float]
    keypoints: Dict[str, Dict[str, float]]
    
    # Activity
    activity: str
    activity_confidence: float
    movement_score: float
    is_moving: bool
    activity_duration: float
    
    # Fall Detection
    fall_status: str
    fall_detected: bool
    
    # Emotion
    emotion: str
    emotion_confidence: float
    mood_score: float
    is_distressed: bool
    
    # Fields with default values (must come last)
    fall_event: Optional[FallEvent] = None
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Helper to convert numpy types to Python native types
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            elif isinstance(val, (list, tuple)):
                return [to_native(v) for v in val]
            elif isinstance(val, dict):
                return {k: to_native(v) for k, v in val.items()}
            return val
        
        return {
            "person_id": int(self.person_id),
            "timestamp": float(self.timestamp),
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "position": {
                "bbox": [float(v) for v in self.bbox],
                "center": {"x": float(self.center[0]), "y": float(self.center[1])}
            },
            "activity": {
                "type": str(self.activity),
                "confidence": float(self.activity_confidence),
                "duration_seconds": float(self.activity_duration),
                "movement_score": float(self.movement_score),
                "is_moving": bool(self.is_moving)
            },
            "fall_detection": {
                "status": str(self.fall_status),
                "fall_detected": bool(self.fall_detected),
                "fall_event": self.fall_event.fall_type if self.fall_event else None
            },
            "emotion": {
                "type": str(self.emotion),
                "confidence": float(self.emotion_confidence),
                "mood_score": float(self.mood_score),
                "is_distressed": bool(self.is_distressed)
            },
            "alerts": list(self.alerts)
        }


class AlertType:
    """Alert type constants."""
    FALL_DETECTED = "fall_detected"
    INACTIVITY = "inactivity"
    DISTRESS = "distress"
    NEGATIVE_EMOTION = "negative_emotion"
    LEFT_VIEW = "left_camera_view"


class AlertCooldown:
    """Manages alert cooldowns to prevent spam."""
    
    def __init__(self):
        self._last_alert_time: Dict[str, float] = {}
        self._cooldowns = {
            AlertType.FALL_DETECTED: 30.0,    # 30 seconds
            AlertType.INACTIVITY: 300.0,      # 5 minutes
            AlertType.DISTRESS: 60.0,         # 1 minute
            AlertType.NEGATIVE_EMOTION: 120.0, # 2 minutes
            AlertType.LEFT_VIEW: 300.0        # 5 minutes
        }
    
    def can_alert(self, alert_type: str, person_id: int) -> bool:
        """Check if enough time has passed to send another alert."""
        key = f"{alert_type}_{person_id}"
        current_time = time.time()
        
        if key not in self._last_alert_time:
            return True
        
        cooldown = self._cooldowns.get(alert_type, 60.0)
        return (current_time - self._last_alert_time[key]) >= cooldown
    
    def record_alert(self, alert_type: str, person_id: int) -> None:
        """Record that an alert was sent."""
        key = f"{alert_type}_{person_id}"
        self._last_alert_time[key] = time.time()
    
    def reset(self, person_id: int = None) -> None:
        """Reset cooldowns for a person or all."""
        if person_id is None:
            self._last_alert_time.clear()
        else:
            keys_to_remove = [k for k in self._last_alert_time if k.endswith(f"_{person_id}")]
            for k in keys_to_remove:
                del self._last_alert_time[k]


@dataclass
class Alert:
    """Represents an alert event."""
    alert_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    person_id: int
    message: str
    timestamp: float
    data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        # Helper to convert numpy types to Python native types
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            elif isinstance(val, (list, tuple)):
                return [to_native(v) for v in val]
            elif isinstance(val, dict):
                return {k: to_native(v) for k, v in val.items()}
            return val
        
        return {
            "type": str(self.alert_type),
            "severity": str(self.severity),
            "person_id": int(self.person_id),
            "message": str(self.message),
            "timestamp": float(self.timestamp),
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": to_native(self.data)
        }


class ElderlyCareMonitor:
    """
    Comprehensive elderly care monitoring system.
    
    Integrates pose detection, activity classification, fall detection,
    and emotion analysis for complete elderly monitoring.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the elderly care monitor.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        
        # Initialize pose detection model
        self._init_pose_model()
        
        # Initialize sub-modules
        self.activity_classifier = ActivityClassifier(self.config)
        self.fall_detector = FallDetector(self.config)
        self.emotion_detector = EmotionDetector(self.config)
        
        # Alert tracking
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.alert_cooldown = AlertCooldown()
        
        # Person tracking
        self.tracked_persons: Dict[int, ElderlyStatus] = {}
        self.last_seen: Dict[int, float] = {}
        
        # Load alert settings
        alert_config = self.config.get('alerts', {})
        self.inactivity_alert_threshold = alert_config.get('inactivity_threshold', 1800)  # 30 min
        self.missing_person_threshold = alert_config.get('missing_threshold', 600)  # 10 min
        
        # Visualization settings
        self._setup_visualization()
    
    def _init_pose_model(self) -> None:
        """Initialize YOLOv8 pose estimation model."""
        model_config = self.config.get('model', {})
        
        model_name = model_config.get('name', 'yolov8m-pose')
        self.device = get_device(model_config.get('device', 'auto'))
        
        print(f"Loading pose model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.pose_model = YOLO(model_name)
        
        self.conf_threshold = model_config.get('confidence_threshold', 0.5)
        self.iou_threshold = model_config.get('iou_threshold', 0.45)
        
        detection_config = self.config.get('detection', {})
        self.enable_tracking = detection_config.get('enable_tracking', True)
        self.tracker = detection_config.get('tracker', 'bytetrack')
    
    def _setup_visualization(self) -> None:
        """Setup visualization parameters."""
        vis_config = self.config.get('visualization', {})
        keypoint_config = self.config.get('keypoints', {})
        
        self.show_skeleton = vis_config.get('show_skeleton', True)
        self.show_activity = vis_config.get('show_activity', True)
        self.show_emotion = vis_config.get('show_emotion', True)
        self.show_alerts = vis_config.get('show_alerts', True)
        
        self.keypoint_names = keypoint_config.get('names', [])
        self.skeleton_connections = keypoint_config.get('skeleton', [])
        
        # Activity colors
        self.activity_colors = {
            'standing': (0, 255, 0),    # Green
            'sitting': (255, 255, 0),   # Cyan
            'walking': (0, 255, 255),   # Yellow
            'lying_down': (0, 165, 255), # Orange
            'inactive': (0, 0, 255),    # Red
            'unknown': (128, 128, 128)  # Gray
        }
    
    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert and notify callbacks."""
        self.alerts.append(alert)
        
        # Print alert
        severity_colors = {
            'critical': '\033[91m',  # Red
            'high': '\033[93m',      # Yellow
            'medium': '\033[94m',    # Blue
            'low': '\033[92m'        # Green
        }
        reset = '\033[0m'
        color = severity_colors.get(alert.severity, '')
        
        print(f"\n{color}🚨 ALERT [{alert.severity.upper()}]: {alert.message}{reset}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def _check_alerts(self, status: ElderlyStatus) -> List[str]:
        """Check for alert conditions and generate alerts with cooldown."""
        alerts = []
        timestamp = time.time()
        
        # 1. Fall Detection Alert (CRITICAL) - always important
        if status.fall_detected:
            if self.alert_cooldown.can_alert(AlertType.FALL_DETECTED, status.person_id):
                alert = Alert(
                    alert_type=AlertType.FALL_DETECTED,
                    severity="critical",
                    person_id=status.person_id,
                    message=f"FALL DETECTED! Person {status.person_id} has fallen!",
                    timestamp=timestamp,
                    data={
                        "fall_type": status.fall_event.fall_type if status.fall_event else "unknown",
                        "position": status.center
                    }
                )
                self._trigger_alert(alert)
                self.alert_cooldown.record_alert(AlertType.FALL_DETECTED, status.person_id)
            alerts.append("FALL DETECTED")
        
        # 2. Inactivity Alert (HIGH) - with cooldown
        if status.activity == 'inactive' and status.activity_duration > self.inactivity_alert_threshold:
            if self.alert_cooldown.can_alert(AlertType.INACTIVITY, status.person_id):
                alert = Alert(
                    alert_type=AlertType.INACTIVITY,
                    severity="high",
                    person_id=status.person_id,
                    message=f"Inactivity detected! Person {status.person_id} has been inactive for {status.activity_duration/60:.1f} minutes",
                    timestamp=timestamp,
                    data={
                        "duration": status.activity_duration,
                        "last_activity": status.activity
                    }
                )
                self._trigger_alert(alert)
                self.alert_cooldown.record_alert(AlertType.INACTIVITY, status.person_id)
            alerts.append(f"INACTIVE ({status.activity_duration/60:.1f} min)")
        
        # 3. Distress Alert (HIGH) - with cooldown to prevent spam
        # Only trigger if mood is really negative (below -0.5) AND is_distressed
        if status.is_distressed and status.mood_score < -0.5:
            if self.alert_cooldown.can_alert(AlertType.DISTRESS, status.person_id):
                alert = Alert(
                    alert_type=AlertType.DISTRESS,
                    severity="high",
                    person_id=status.person_id,
                    message=f"Distress detected! Person {status.person_id} appears to be in distress",
                    timestamp=timestamp,
                    data={
                        "emotion": status.emotion,
                        "mood_score": status.mood_score
                    }
                )
                self._trigger_alert(alert)
                self.alert_cooldown.record_alert(AlertType.DISTRESS, status.person_id)
            alerts.append("DISTRESSED")
        
        # 4. Prolonged Negative Emotion (MEDIUM) - just track, no alert spam
        elif status.emotion in ['sad', 'angry', 'fearful'] and status.emotion_confidence > 0.7:
            alerts.append(f"Negative emotion: {status.emotion}")
        
        return alerts
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = None
    ) -> Tuple[np.ndarray, List[ElderlyStatus]]:
        """
        Process a single frame for elderly monitoring.
        
        Args:
            frame: Input frame (BGR).
            timestamp: Optional timestamp.
            
        Returns:
            Tuple of (annotated_frame, list of ElderlyStatus).
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Run pose detection
        if self.enable_tracking:
            results = self.pose_model.track(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],  # Person only
                tracker=f"{self.tracker}.yaml",
                persist=True,
                verbose=False
            )
        else:
            results = self.pose_model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                verbose=False
            )
        
        annotated_frame = frame.copy()
        statuses = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            keypoints_data = result.keypoints if hasattr(result, 'keypoints') and result.keypoints is not None else None
            
            for i, box in enumerate(boxes):
                # Get detection info
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                person_id = int(box.id[0].cpu().numpy()) if box.id is not None else i
                
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                # Get keypoints
                raw_kps = None
                formatted_kps = {}
                
                if keypoints_data is not None and i < len(keypoints_data):
                    raw_kps = keypoints_data[i].data[0].cpu().numpy()
                    formatted_kps = format_keypoints(raw_kps, self.keypoint_names)
                
                if raw_kps is None:
                    continue
                
                # Classify activity
                activity_state = self.activity_classifier.classify_activity(
                    person_id, raw_kps, timestamp
                )
                
                # Detect falls
                fall_status, fall_event = self.fall_detector.detect_fall(
                    person_id, raw_kps, timestamp
                )
                
                # Detect emotion
                emotion_result = self.emotion_detector.detect_emotion(
                    frame, person_id, raw_kps, timestamp
                )
                mood_analysis = self.emotion_detector.get_mood_analysis(person_id)
                
                # Create status
                status = ElderlyStatus(
                    person_id=person_id,
                    timestamp=timestamp,
                    bbox=bbox,
                    center=center,
                    keypoints=formatted_kps,
                    activity=activity_state.activity.value,
                    activity_confidence=activity_state.confidence,
                    movement_score=activity_state.movement_score,
                    is_moving=activity_state.is_moving,
                    activity_duration=activity_state.duration,
                    fall_status=fall_status.value,
                    fall_detected=(fall_status == FallStatus.FALL_DETECTED),
                    fall_event=fall_event,
                    emotion=emotion_result.emotion.value,
                    emotion_confidence=emotion_result.confidence,
                    mood_score=mood_analysis['current_mood'],
                    is_distressed=mood_analysis['is_distressed']
                )
                
                # Check for alerts
                status.alerts = self._check_alerts(status)
                
                # Update tracking
                self.tracked_persons[person_id] = status
                self.last_seen[person_id] = timestamp
                
                statuses.append(status)
                
                # Draw visualization
                annotated_frame = self._draw_status(annotated_frame, status, raw_kps, emotion_result)
        
        # Check for missing persons
        self._check_missing_persons(timestamp)
        
        return annotated_frame, statuses
    
    def _draw_status(
        self,
        frame: np.ndarray,
        status: ElderlyStatus,
        keypoints: np.ndarray,
        emotion_result: EmotionResult
    ) -> np.ndarray:
        """Draw monitoring status on frame."""
        x1, y1, x2, y2 = [int(v) for v in status.bbox]
        
        # Get color based on status
        if status.fall_detected:
            box_color = (0, 0, 255)  # Red for fall
        elif status.is_distressed:
            box_color = (0, 0, 200)  # Dark red for distress
        elif status.activity == 'inactive':
            box_color = (0, 165, 255)  # Orange for inactive
        else:
            box_color = self.activity_colors.get(status.activity, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw skeleton
        if self.show_skeleton and keypoints is not None:
            frame = self._draw_skeleton(frame, keypoints)
        
        # Draw info panel
        panel_y = y1 - 80
        if panel_y < 0:
            panel_y = y2 + 5
        
        # Background panel
        cv2.rectangle(
            frame,
            (x1, panel_y),
            (x1 + 200, panel_y + 75),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (x1, panel_y),
            (x1 + 200, panel_y + 75),
            box_color,
            1
        )
        
        # Text info
        y_offset = panel_y + 15
        
        # ID and Activity
        cv2.putText(
            frame,
            f"ID:{status.person_id} | {status.activity.upper()}",
            (x1 + 5, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )
        y_offset += 15
        
        # Movement score
        cv2.putText(
            frame,
            f"Move: {status.movement_score:.1f} | {status.activity_duration:.0f}s",
            (x1 + 5, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )
        y_offset += 15
        
        # Emotion
        if self.show_emotion:
            emotion_color = self.emotion_detector.EMOTION_COLORS.get(
                emotion_result.emotion, (200, 200, 200)
            )
            cv2.putText(
                frame,
                f"Emotion: {status.emotion} ({status.emotion_confidence:.0%})",
                (x1 + 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                emotion_color,
                1
            )
            y_offset += 15
        
        # Fall status
        if status.fall_status != 'normal':
            cv2.putText(
                frame,
                f"⚠️ {status.fall_status.upper()}",
                (x1 + 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
            y_offset += 15
        
        # Alerts
        if self.show_alerts and status.alerts:
            for alert_msg in status.alerts[:2]:  # Show max 2 alerts
                cv2.putText(
                    frame,
                    f"🚨 {alert_msg}",
                    (x1 + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1
                )
                y_offset += 15
        
        # Draw emotion on face if detected
        if emotion_result.face_bbox:
            frame = self.emotion_detector.draw_emotion(frame, emotion_result)
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw pose skeleton on frame."""
        skeleton_colors = [
            (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
            (0, 255, 255), (0, 255, 0), (0, 255, 0),
            (0, 128, 255), (0, 128, 255), (255, 0, 255), (255, 0, 255),
            (255, 255, 0), (0, 255, 0), (0, 255, 0), (0, 128, 255), (0, 128, 255)
        ]
        
        # Draw connections
        for idx, (start_idx, end_idx) in enumerate(self.skeleton_connections):
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                
                if len(start_kp) > 2 and len(end_kp) > 2:
                    if start_kp[2] > 0.3 and end_kp[2] > 0.3:
                        start = (int(start_kp[0]), int(start_kp[1]))
                        end = (int(end_kp[0]), int(end_kp[1]))
                        color = skeleton_colors[idx % len(skeleton_colors)]
                        cv2.line(frame, start, end, color, 2)
        
        # Draw keypoints
        for kp in keypoints:
            if len(kp) > 2 and kp[2] > 0.3:
                point = (int(kp[0]), int(kp[1]))
                cv2.circle(frame, point, 4, (0, 255, 255), -1)
                cv2.circle(frame, point, 4, (0, 0, 0), 1)
        
        return frame
    
    def _check_missing_persons(self, current_time: float) -> None:
        """Check for persons who have left the camera view."""
        for person_id, last_time in list(self.last_seen.items()):
            if current_time - last_time > self.missing_person_threshold:
                alert = Alert(
                    alert_type=AlertType.LEFT_VIEW,
                    severity="medium",
                    person_id=person_id,
                    message=f"Person {person_id} has left camera view for {(current_time - last_time)/60:.1f} minutes",
                    timestamp=current_time
                )
                self._trigger_alert(alert)
                
                # Remove from tracking
                self.last_seen.pop(person_id, None)
                self.tracked_persons.pop(person_id, None)
    
    def process_video(
        self,
        source,
        output_path: Optional[str] = None,
        show: bool = True,
        callback: Optional[Callable] = None
    ) -> Generator[Tuple[np.ndarray, List[ElderlyStatus], int], None, None]:
        """
        Process video for elderly monitoring.
        
        Args:
            source: Video file path or camera index.
            output_path: Path to save output video.
            show: Whether to display video.
            callback: Optional callback for each frame.
            
        Yields:
            Tuple of (annotated_frame, statuses, frame_number).
        """
        video_config = self.config.get('video', {})
        
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        target_width = video_config.get('width')
        target_height = video_config.get('height')
        
        if target_width and target_height:
            width, height = target_width, target_height
        
        writer = None
        if output_path:
            codec = self.config.get('output', {}).get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if target_width and target_height:
                    frame = cv2.resize(frame, (target_width, target_height))
                
                # Process frame
                timestamp = start_time + (frame_num / fps)
                annotated_frame, statuses = self.process_frame(frame, timestamp)
                
                # Add frame info
                info_text = f"Frame: {frame_num} | Persons: {len(statuses)} | Alerts: {len(self.alerts)}"
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                if writer:
                    writer.write(annotated_frame)
                
                if show:
                    cv2.imshow("Elderly Care Monitor", annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                
                if callback:
                    callback(annotated_frame, statuses, frame_num)
                
                yield annotated_frame, statuses, frame_num
                
                frame_num += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
    
    def get_monitoring_summary(self) -> Dict:
        """Get summary of monitoring session."""
        return {
            "tracked_persons": len(self.tracked_persons),
            "total_alerts": len(self.alerts),
            "alerts_by_type": self._count_alerts_by_type(),
            "persons": {
                pid: status.to_dict() 
                for pid, status in self.tracked_persons.items()
            }
        }
    
    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count alerts by type."""
        counts = {}
        for alert in self.alerts:
            counts[alert.alert_type] = counts.get(alert.alert_type, 0) + 1
        return counts
    
    def save_session_log(self, output_path: str) -> None:
        """Save session log to JSON file."""
        log_data = {
            "session_end": datetime.now().isoformat(),
            "summary": self.get_monitoring_summary(),
            "alerts": [alert.to_dict() for alert in self.alerts]
        }
        
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Session log saved to: {output_path}")

