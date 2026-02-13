"""
Advanced Facial Emotion Detection Module for Elderly Care

Uses DeepFace for accurate emotion detection with multiple backend support.
Supports emotion categories:
- Happy
- Sad  
- Angry
- Fearful
- Surprised
- Neutral
- Tired/Exhausted (inferred)
- Distressed (computed from negative emotions)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time
import threading


class EmotionType(Enum):
    """Enumeration of detectable emotions."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    TIRED = "tired"
    DISTRESSED = "distressed"
    DISGUST = "disgust"
    UNKNOWN = "unknown"


@dataclass
class EmotionResult:
    """Represents emotion detection result."""
    emotion: EmotionType
    confidence: float
    all_scores: Dict[str, float]
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    timestamp: float = 0.0
    raw_dominant: str = ""  # Raw dominant emotion from model


# Check for DeepFace availability
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded for accurate emotion detection")
except ImportError:
    print("⚠️ DeepFace not installed. Install with: pip install deepface")


class EmotionDetector:
    """
    Advanced emotion detector using DeepFace.
    
    Features:
    - Multiple backend support (RetinaFace, MTCNN, OpenCV, etc.)
    - Ensemble emotion analysis for accuracy
    - Temporal smoothing to reduce flickering
    - Mood tracking over time
    """
    
    # Emotion colors for visualization (BGR)
    EMOTION_COLORS = {
        EmotionType.HAPPY: (0, 255, 0),      # Green
        EmotionType.SAD: (255, 0, 0),        # Blue
        EmotionType.ANGRY: (0, 0, 255),      # Red
        EmotionType.FEARFUL: (255, 0, 255),  # Magenta
        EmotionType.SURPRISED: (0, 255, 255), # Yellow
        EmotionType.NEUTRAL: (128, 128, 128), # Gray
        EmotionType.TIRED: (128, 0, 128),    # Purple
        EmotionType.DISTRESSED: (0, 0, 200), # Dark Red
        EmotionType.DISGUST: (0, 128, 128),  # Olive
        EmotionType.UNKNOWN: (200, 200, 200)  # Light Gray
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize emotion detector.
        
        Args:
            config: Configuration dictionary.
        """
        config = config or {}
        emotion_config = config.get('emotion_detection', {})
        
        # Configuration from config file
        self.detector_backend = emotion_config.get('detector_backend', 'opencv')
        self.model_name = emotion_config.get('model_name', 'Emotion')
        self.confidence_threshold = emotion_config.get('confidence_threshold', 0.4)
        self.history_length = emotion_config.get('history_length', 30)
        self.smoothing_window = emotion_config.get('smoothing_window', 5)
        self.distress_threshold = emotion_config.get('distress_threshold', 0.6)
        self.min_face_size = emotion_config.get('min_face_size', 40)
        
        # Initialize face detector for fallback
        self._init_face_detector()
        
        # Emotion history for each person (for temporal smoothing)
        self.emotion_history: Dict[int, deque] = {}
        self.raw_scores_history: Dict[int, deque] = {}
        self.mood_scores: Dict[int, deque] = {}
        
        # Distress tracking
        self.negative_emotion_count: Dict[int, int] = {}
        
        # Cache for performance
        self._last_detection_time: Dict[int, float] = {}
        self._cached_results: Dict[int, EmotionResult] = {}
        self._detection_interval = emotion_config.get('detection_interval', 0.1)  # 100ms
        
        # Thread lock for thread-safe access
        self._lock = threading.Lock()
        
        # DeepFace status
        self.deepface_available = DEEPFACE_AVAILABLE
        if self.deepface_available:
            print(f"🎭 Emotion detector initialized with backend: {self.detector_backend}")
        else:
            print("⚠️ Using fallback emotion detection (less accurate)")
    
    def _init_face_detector(self) -> None:
        """Initialize OpenCV face detector as fallback."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar Cascade (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        return [tuple(face) for face in faces]
    
    def _detect_face_from_keypoints(
        self, 
        keypoints: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Estimate face region from pose keypoints.
        Uses nose, eyes, and ears to estimate face bounding box.
        """
        # COCO keypoint indices
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_EAR = 3
        RIGHT_EAR = 4
        
        points = []
        for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]:
            if idx < len(keypoints):
                kp = keypoints[idx]
                if len(kp) > 2 and kp[2] > 0.3:
                    points.append((kp[0], kp[1]))
        
        if len(points) < 2:
            return None
        
        # Calculate bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Expand box to include full face
        width = max_x - min_x
        height = max_y - min_y
        
        # Face is roughly 2x the eye-to-eye distance
        face_width = max(width * 2.5, height * 2)
        face_height = face_width * 1.3  # Face is taller than wide
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2 + face_height * 0.15  # Shift down slightly
        
        x = int(center_x - face_width / 2)
        y = int(center_y - face_height / 2)
        w = int(face_width)
        h = int(face_height)
        
        # Clip to frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_shape[1] - x)
        h = min(h, frame_shape[0] - y)
        
        if w < self.min_face_size or h < self.min_face_size:
            return None
        
        return (x, y, w, h)
    
    def _analyze_emotion_deepface(
        self, 
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Dict[str, float], str, Optional[Tuple[int, int, int, int]]]:
        """
        Analyze emotion using DeepFace.
        
        Returns:
            Tuple of (emotion_scores, dominant_emotion, face_bbox)
        """
        if not self.deepface_available:
            return {}, "unknown", None
        
        try:
            # If we have a face bbox, crop the image for faster processing
            if face_bbox:
                x, y, w, h = face_bbox
                # Add some padding
                pad = int(min(w, h) * 0.2)
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(w + 2*pad, frame.shape[1] - x)
                h = min(h + 2*pad, frame.shape[0] - y)
                face_img = frame[y:y+h, x:x+w]
            else:
                face_img = frame
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            
            # Handle list or dict result
            if isinstance(result, list):
                if len(result) == 0:
                    return {}, "unknown", None
                result = result[0]
            
            emotion_scores = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'unknown')
            
            # Get face region from DeepFace if available
            region = result.get('region', {})
            if region and not face_bbox:
                detected_bbox = (
                    region.get('x', 0),
                    region.get('y', 0),
                    region.get('w', 0),
                    region.get('h', 0)
                )
                if all(v > 0 for v in detected_bbox):
                    face_bbox = detected_bbox
            
            # Normalize scores to 0-1 range
            normalized_scores = {}
            for emotion, score in emotion_scores.items():
                normalized_scores[emotion.lower()] = score / 100.0
            
            return normalized_scores, dominant.lower(), face_bbox
            
        except Exception as e:
            # DeepFace can throw various exceptions
            return {}, "unknown", face_bbox
    
    def _analyze_emotion_fallback(
        self, 
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Dict[str, float]:
        """
        Fallback emotion analysis using image features.
        Returns a basic neutral emotion when DeepFace is unavailable.
        """
        x, y, w, h = face_bbox
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return {'neutral': 1.0}
        
        # Convert to grayscale and analyze basic features
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Very basic heuristics
        scores = {
            'happy': 0.1,
            'sad': 0.1,
            'angry': 0.1,
            'fear': 0.1,
            'surprise': 0.1,
            'neutral': 0.5
        }
        
        # Higher contrast might indicate more expression
        if std_intensity > 50:
            scores['neutral'] = 0.3
            scores['happy'] = 0.3
        
        return scores
    
    def _smooth_emotions(
        self, 
        person_id: int, 
        current_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply temporal smoothing to reduce flickering.
        Uses exponential moving average over recent detections.
        """
        with self._lock:
            if person_id not in self.raw_scores_history:
                self.raw_scores_history[person_id] = deque(maxlen=self.smoothing_window)
            
            self.raw_scores_history[person_id].append(current_scores)
            
            if len(self.raw_scores_history[person_id]) < 2:
                return current_scores
            
            # Calculate smoothed scores
            smoothed = {}
            all_emotions = set()
            for scores in self.raw_scores_history[person_id]:
                all_emotions.update(scores.keys())
            
            for emotion in all_emotions:
                values = [s.get(emotion, 0) for s in self.raw_scores_history[person_id]]
                # Weighted average (more recent = higher weight)
                weights = [1.5 ** i for i in range(len(values))]
                smoothed[emotion] = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            
            return smoothed
    
    def _map_emotion_type(self, emotion: str, scores: Dict[str, float]) -> EmotionType:
        """Map raw emotion string to EmotionType with special handling."""
        # Direct mapping
        emotion_map = {
            'happy': EmotionType.HAPPY,
            'sad': EmotionType.SAD,
            'angry': EmotionType.ANGRY,
            'fear': EmotionType.FEARFUL,
            'fearful': EmotionType.FEARFUL,
            'surprise': EmotionType.SURPRISED,
            'surprised': EmotionType.SURPRISED,
            'neutral': EmotionType.NEUTRAL,
            'disgust': EmotionType.DISGUST
        }
        
        emotion_type = emotion_map.get(emotion.lower(), EmotionType.NEUTRAL)
        
        # Check for distress (combination of negative emotions)
        negative_score = (
            scores.get('fear', 0) + 
            scores.get('sad', 0) + 
            scores.get('angry', 0) +
            scores.get('disgust', 0)
        )
        if negative_score > self.distress_threshold:
            emotion_type = EmotionType.DISTRESSED
        
        # Check for tiredness (neutral with low engagement indicators)
        if emotion_type == EmotionType.NEUTRAL:
            happy_score = scores.get('happy', 0)
            surprise_score = scores.get('surprise', 0)
            sad_score = scores.get('sad', 0)
            
            # Low expression overall with slight sadness suggests tiredness
            if happy_score < 0.1 and surprise_score < 0.1 and sad_score > 0.1:
                emotion_type = EmotionType.TIRED
        
        return emotion_type
    
    def detect_emotion(
        self,
        frame: np.ndarray,
        person_id: int = 0,
        keypoints: np.ndarray = None,
        timestamp: float = None
    ) -> EmotionResult:
        """
        Detect emotion from frame.
        
        Args:
            frame: Input frame (BGR).
            person_id: Tracking ID for person.
            keypoints: Optional pose keypoints for face localization.
            timestamp: Optional timestamp.
            
        Returns:
            EmotionResult with detected emotion.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Check cache to avoid excessive processing
        if person_id in self._last_detection_time:
            time_since_last = timestamp - self._last_detection_time[person_id]
            if time_since_last < self._detection_interval and person_id in self._cached_results:
                # Return cached result with updated timestamp
                cached = self._cached_results[person_id]
                return EmotionResult(
                    emotion=cached.emotion,
                    confidence=cached.confidence,
                    all_scores=cached.all_scores,
                    face_bbox=cached.face_bbox,
                    timestamp=timestamp,
                    raw_dominant=cached.raw_dominant
                )
        
        # Detect face
        face_bbox = None
        
        # First try to get face from keypoints (more reliable when pose is detected)
        if keypoints is not None:
            face_bbox = self._detect_face_from_keypoints(keypoints, frame.shape[:2])
        
        # Analyze emotion
        if self.deepface_available:
            scores, dominant, detected_bbox = self._analyze_emotion_deepface(frame, face_bbox)
            if detected_bbox:
                face_bbox = detected_bbox
        else:
            # Fallback: detect face with OpenCV first
            if face_bbox is None:
                faces = self._detect_faces_opencv(frame)
                if faces:
                    face_bbox = faces[0]
            
            if face_bbox:
                scores = self._analyze_emotion_fallback(frame, face_bbox)
                dominant = max(scores, key=scores.get) if scores else "unknown"
            else:
                scores = {}
                dominant = "unknown"
        
        # Handle no detection
        if not scores or dominant == "unknown":
            result = EmotionResult(
                emotion=EmotionType.UNKNOWN,
                confidence=0.0,
                all_scores={},
                face_bbox=face_bbox,
                timestamp=timestamp,
                raw_dominant="unknown"
            )
            self._cached_results[person_id] = result
            self._last_detection_time[person_id] = timestamp
            return result
        
        # Apply temporal smoothing
        smoothed_scores = self._smooth_emotions(person_id, scores)
        
        # Find dominant emotion from smoothed scores
        dominant_emotion = max(smoothed_scores, key=smoothed_scores.get)
        confidence = smoothed_scores[dominant_emotion]
        
        # Map to EmotionType
        emotion_type = self._map_emotion_type(dominant_emotion, smoothed_scores)
        
        result = EmotionResult(
            emotion=emotion_type,
            confidence=confidence,
            all_scores=smoothed_scores,
            face_bbox=face_bbox,
            timestamp=timestamp,
            raw_dominant=dominant_emotion
        )
        
        # Update history and cache
        self._update_history(person_id, result)
        self._cached_results[person_id] = result
        self._last_detection_time[person_id] = timestamp
        
        return result
    
    def _update_history(self, person_id: int, result: EmotionResult) -> None:
        """Update emotion history for a person."""
        with self._lock:
            if person_id not in self.emotion_history:
                self.emotion_history[person_id] = deque(maxlen=self.history_length)
                self.mood_scores[person_id] = deque(maxlen=self.history_length)
                self.negative_emotion_count[person_id] = 0
            
            self.emotion_history[person_id].append(result)
            
            # Calculate mood score (-1 to 1)
            mood = self._calculate_mood_score(result)
            self.mood_scores[person_id].append(mood)
            
            # Track negative emotions for distress detection
            if result.emotion in [EmotionType.SAD, EmotionType.ANGRY, 
                                  EmotionType.FEARFUL, EmotionType.DISTRESSED,
                                  EmotionType.DISGUST]:
                self.negative_emotion_count[person_id] += 1
            else:
                # Decay negative count over time
                self.negative_emotion_count[person_id] = max(0, 
                    self.negative_emotion_count[person_id] - 1)
    
    def _calculate_mood_score(self, result: EmotionResult) -> float:
        """
        Calculate mood score from emotion.
        Returns value from -1 (very negative) to 1 (very positive).
        """
        mood_weights = {
            EmotionType.HAPPY: 1.0,
            EmotionType.SURPRISED: 0.3,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.TIRED: -0.2,
            EmotionType.SAD: -0.6,
            EmotionType.DISGUST: -0.5,
            EmotionType.ANGRY: -0.7,
            EmotionType.FEARFUL: -0.8,
            EmotionType.DISTRESSED: -1.0,
            EmotionType.UNKNOWN: 0.0
        }
        
        return mood_weights.get(result.emotion, 0.0) * result.confidence
    
    def get_mood_analysis(self, person_id: int) -> Dict:
        """
        Get mood analysis for a person.
        
        Returns:
            Dictionary with mood statistics.
        """
        with self._lock:
            if person_id not in self.mood_scores or not self.mood_scores[person_id]:
                return {
                    "current_mood": 0.0,
                    "average_mood": 0.0,
                    "mood_trend": "stable",
                    "negative_emotion_streak": 0,
                    "dominant_emotion": "unknown",
                    "is_distressed": False,
                    "emotion_distribution": {}
                }
            
            scores = list(self.mood_scores[person_id])
            history = list(self.emotion_history[person_id])
            
            current_mood = scores[-1] if scores else 0.0
            average_mood = float(np.mean(scores))
            
            # Calculate trend
            if len(scores) >= 5:
                recent = float(np.mean(scores[-5:]))
                older = float(np.mean(scores[:-5])) if len(scores) > 5 else average_mood
                
                if recent > older + 0.1:
                    trend = "improving"
                elif recent < older - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            # Find dominant emotion
            emotion_counts = {}
            for result in history:
                e = result.emotion.value
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            total = sum(emotion_counts.values())
            emotion_distribution = {k: v/total for k, v in emotion_counts.items()} if total > 0 else {}
            
            dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"
            
            # Check distress - require more evidence before flagging
            negative_count = self.negative_emotion_count.get(person_id, 0)
            is_distressed = (
                (negative_count >= 15 and current_mood < -0.6) or  # 15+ consecutive negative emotions
                (current_mood < -0.7 and len(history) >= 10) or    # Very negative mood over time
                emotion_counts.get('distressed', 0) > len(history) * 0.4  # 40%+ distressed
            )
            
            return {
                "current_mood": current_mood,
                "average_mood": average_mood,
                "mood_trend": trend,
                "negative_emotion_streak": self.negative_emotion_count.get(person_id, 0),
                "dominant_emotion": dominant,
                "is_distressed": is_distressed,
                "emotion_distribution": emotion_distribution
            }
    
    def draw_emotion(
        self, 
        frame: np.ndarray, 
        result: EmotionResult,
        show_scores: bool = True,
        show_mood: bool = False,
        person_id: int = 0
    ) -> np.ndarray:
        """
        Draw emotion detection result on frame.
        
        Args:
            frame: Input frame.
            result: EmotionResult to visualize.
            show_scores: Whether to show all emotion scores.
            show_mood: Whether to show mood indicator.
            person_id: Person ID for mood analysis.
            
        Returns:
            Frame with visualization.
        """
        if result.face_bbox is None:
            return frame
        
        x, y, w, h = result.face_bbox
        color = self.EMOTION_COLORS.get(result.emotion, (200, 200, 200))
        
        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label
        label = f"{result.emotion.value}: {result.confidence:.0%}"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, 
            (x, y - text_h - 10), 
            (x + text_w + 10, y),
            color, -1
        )
        
        # Text
        cv2.putText(
            frame, label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
        
        # Show top emotion scores
        if show_scores and result.all_scores:
            y_offset = y + h + 20
            sorted_scores = sorted(result.all_scores.items(), key=lambda x: -x[1])[:4]
            for emotion, score in sorted_scores:
                text = f"{emotion}: {score:.0%}"
                cv2.putText(
                    frame, text,
                    (x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1
                )
                y_offset += 15
        
        # Show mood indicator
        if show_mood:
            mood_analysis = self.get_mood_analysis(person_id)
            mood_score = mood_analysis['current_mood']
            
            # Draw mood bar
            bar_x = x + w + 10
            bar_y = y
            bar_h = h
            bar_w = 10
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            
            # Mood level (centered at middle)
            mid_y = bar_y + bar_h // 2
            mood_height = int((bar_h / 2) * abs(mood_score))
            
            if mood_score >= 0:
                mood_color = (0, 255, 0)  # Green for positive
                cv2.rectangle(frame, (bar_x, mid_y - mood_height), (bar_x + bar_w, mid_y), mood_color, -1)
            else:
                mood_color = (0, 0, 255)  # Red for negative
                cv2.rectangle(frame, (bar_x, mid_y), (bar_x + bar_w, mid_y + mood_height), mood_color, -1)
            
            # Center line
            cv2.line(frame, (bar_x, mid_y), (bar_x + bar_w, mid_y), (255, 255, 255), 1)
        
        return frame
    
    def reset_person(self, person_id: int) -> None:
        """Reset tracking for a person."""
        with self._lock:
            self.emotion_history.pop(person_id, None)
            self.raw_scores_history.pop(person_id, None)
            self.mood_scores.pop(person_id, None)
            self.negative_emotion_count.pop(person_id, None)
            self._cached_results.pop(person_id, None)
            self._last_detection_time.pop(person_id, None)
    
    def get_all_emotion_scores(self, person_id: int) -> Dict[str, float]:
        """Get all emotion scores for a person from the last detection."""
        if person_id in self._cached_results:
            return self._cached_results[person_id].all_scores
        return {}
