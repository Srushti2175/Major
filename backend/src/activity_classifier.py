"""
Activity Classification Module for Elderly Care

Classifies human activities based on pose keypoints:
- Standing
- Sitting
- Walking
- Lying Down
- Inactive (no movement)

Uses geometric analysis of body keypoints to determine posture.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time


class ActivityType(Enum):
    """Enumeration of detectable activities."""
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    LYING_DOWN = "lying_down"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


@dataclass
class ActivityState:
    """Represents the current activity state of a person."""
    activity: ActivityType
    confidence: float
    duration: float  # seconds in this state
    movement_score: float  # 0-100, higher = more movement
    is_moving: bool
    last_position: Optional[Tuple[float, float]] = None


class ActivityClassifier:
    """
    Classifies human activities based on pose keypoints.
    
    Uses geometric analysis of skeleton to determine:
    - Body orientation (vertical vs horizontal)
    - Joint angles (knee, hip)
    - Movement patterns over time
    """
    
    # Keypoint indices (COCO format)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self, config: Dict = None):
        """
        Initialize the activity classifier.
        
        Args:
            config: Configuration dictionary with activity settings.
        """
        config = config or {}
        activity_config = config.get('activity_detection', {})
        
        # Thresholds from config
        self.standing_angle_threshold = activity_config.get('standing_angle_threshold', 30)
        self.sitting_hip_knee_ratio = activity_config.get('sitting_hip_knee_ratio', 0.6)
        self.lying_aspect_ratio = activity_config.get('lying_aspect_ratio', 0.5)
        self.movement_threshold = activity_config.get('movement_threshold', 10)
        self.inactivity_duration = activity_config.get('inactivity_duration', 30)  # seconds
        self.walking_movement_threshold = activity_config.get('walking_movement_threshold', 15)
        
        # History tracking for each person
        self.person_history: Dict[int, deque] = {}
        self.activity_states: Dict[int, ActivityState] = {}
        self.last_update_time: Dict[int, float] = {}
        
        # Movement history for walking detection
        self.position_history: Dict[int, deque] = {}
        self.history_length = activity_config.get('history_length', 30)  # frames
        
    def _get_keypoint(
        self, 
        keypoints: np.ndarray, 
        idx: int, 
        conf_threshold: float = 0.3
    ) -> Optional[Tuple[float, float]]:
        """
        Get keypoint coordinates if confidence is sufficient.
        
        Args:
            keypoints: Array of keypoints [N, 3].
            idx: Keypoint index.
            conf_threshold: Minimum confidence threshold.
            
        Returns:
            Tuple (x, y) or None if not confident.
        """
        if idx >= len(keypoints):
            return None
        
        kp = keypoints[idx]
        if len(kp) > 2 and kp[2] < conf_threshold:
            return None
        
        return (float(kp[0]), float(kp[1]))
    
    def _calculate_angle(
        self, 
        p1: Tuple[float, float], 
        p2: Tuple[float, float], 
        p3: Tuple[float, float]
    ) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        
        Args:
            p1, p2, p3: Points forming the angle.
            
        Returns:
            Angle in degrees.
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        return np.degrees(np.arccos(cos_angle))
    
    def _calculate_vertical_angle(
        self, 
        p_top: Tuple[float, float], 
        p_bottom: Tuple[float, float]
    ) -> float:
        """
        Calculate angle from vertical axis.
        
        Args:
            p_top: Top point.
            p_bottom: Bottom point.
            
        Returns:
            Angle from vertical in degrees.
        """
        dx = p_bottom[0] - p_top[0]
        dy = p_bottom[1] - p_top[1]
        
        # Angle from vertical (0 = perfectly vertical)
        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
        return angle
    
    def _get_body_center(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Calculate the center of the body.
        
        Args:
            keypoints: Array of keypoints.
            
        Returns:
            Center point (x, y) or None.
        """
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        
        if left_hip and right_hip:
            return ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        
        # Fallback to shoulder center
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        
        if left_shoulder and right_shoulder:
            return ((left_shoulder[0] + right_shoulder[0]) / 2, 
                    (left_shoulder[1] + right_shoulder[1]) / 2)
        
        return None
    
    def _calculate_movement(self, person_id: int, current_center: Tuple[float, float]) -> float:
        """
        Calculate movement score based on position history.
        
        Args:
            person_id: Person tracking ID.
            current_center: Current body center.
            
        Returns:
            Movement score (0-100).
        """
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=self.history_length)
        
        history = self.position_history[person_id]
        history.append(current_center)
        
        if len(history) < 2:
            return 0.0
        
        # Calculate total movement over history
        total_movement = 0.0
        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]
            dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            total_movement += dist
        
        # Normalize to 0-100 scale
        avg_movement = total_movement / len(history)
        movement_score = min(100, avg_movement * 2)  # Scale factor
        
        return movement_score
    
    def _is_standing(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if person is standing.
        
        Criteria:
        - Torso is mostly vertical
        - Knees are relatively straight
        - Body is tall (vertical extent > horizontal)
        """
        # Get key points
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        left_knee = self._get_keypoint(keypoints, self.LEFT_KNEE)
        right_knee = self._get_keypoint(keypoints, self.RIGHT_KNEE)
        left_ankle = self._get_keypoint(keypoints, self.LEFT_ANKLE)
        right_ankle = self._get_keypoint(keypoints, self.RIGHT_ANKLE)
        
        confidence = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Check torso verticality
        if left_shoulder and left_hip:
            angle = self._calculate_vertical_angle(left_shoulder, left_hip)
            if angle < self.standing_angle_threshold:
                checks_passed += 1
            total_checks += 1
            
        if right_shoulder and right_hip:
            angle = self._calculate_vertical_angle(right_shoulder, right_hip)
            if angle < self.standing_angle_threshold:
                checks_passed += 1
            total_checks += 1
        
        # Check leg straightness (knee angle close to 180)
        if left_hip and left_knee and left_ankle:
            knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            if knee_angle > 150:  # Relatively straight
                checks_passed += 1
            total_checks += 1
            
        if right_hip and right_knee and right_ankle:
            knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            if knee_angle > 150:
                checks_passed += 1
            total_checks += 1
        
        if total_checks > 0:
            confidence = checks_passed / total_checks
        
        return confidence > 0.5, confidence
    
    def _is_sitting(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if person is sitting.
        
        Criteria:
        - Torso is vertical
        - Knees are bent (angle < 120 degrees)
        - Hips are at similar height to knees
        """
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        left_knee = self._get_keypoint(keypoints, self.LEFT_KNEE)
        right_knee = self._get_keypoint(keypoints, self.RIGHT_KNEE)
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        left_ankle = self._get_keypoint(keypoints, self.LEFT_ANKLE)
        right_ankle = self._get_keypoint(keypoints, self.RIGHT_ANKLE)
        
        confidence = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Check if torso is vertical (not lying)
        if left_shoulder and left_hip:
            angle = self._calculate_vertical_angle(left_shoulder, left_hip)
            if angle < 45:  # Torso reasonably upright
                checks_passed += 1
            total_checks += 1
        
        # Check knee bend
        if left_hip and left_knee and left_ankle:
            knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            if 60 < knee_angle < 130:  # Bent knee
                checks_passed += 1
            total_checks += 1
            
        if right_hip and right_knee and right_ankle:
            knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            if 60 < knee_angle < 130:
                checks_passed += 1
            total_checks += 1
        
        # Check hip-knee height relationship
        if left_hip and left_knee:
            hip_knee_diff = abs(left_hip[1] - left_knee[1])
            if hip_knee_diff < 50:  # Hips and knees at similar height
                checks_passed += 1
            total_checks += 1
        
        if total_checks > 0:
            confidence = checks_passed / total_checks
        
        return confidence > 0.5, confidence
    
    def _is_lying_down(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if person is lying down.
        
        Criteria:
        - Body is mostly horizontal
        - Width > Height (aspect ratio)
        - Head and feet at similar Y level
        """
        # Get bounding points
        nose = self._get_keypoint(keypoints, self.NOSE)
        left_ankle = self._get_keypoint(keypoints, self.LEFT_ANKLE)
        right_ankle = self._get_keypoint(keypoints, self.RIGHT_ANKLE)
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        
        confidence = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Check torso angle (should be more horizontal)
        if left_shoulder and left_hip:
            angle = self._calculate_vertical_angle(left_shoulder, left_hip)
            if angle > 60:  # More horizontal than vertical
                checks_passed += 1
            total_checks += 1
        
        if right_shoulder and right_hip:
            angle = self._calculate_vertical_angle(right_shoulder, right_hip)
            if angle > 60:
                checks_passed += 1
            total_checks += 1
        
        # Check if head and ankles are at similar Y level
        if nose and (left_ankle or right_ankle):
            ankle_y = left_ankle[1] if left_ankle else right_ankle[1]
            head_ankle_diff = abs(nose[1] - ankle_y)
            
            # Calculate body length for normalization
            body_points = [p for p in [nose, left_shoulder, right_shoulder, 
                                       left_hip, right_hip, left_ankle, right_ankle] if p]
            if len(body_points) >= 2:
                y_values = [p[1] for p in body_points]
                body_height = max(y_values) - min(y_values)
                
                if body_height > 0 and head_ankle_diff < body_height * 0.3:
                    checks_passed += 1
            total_checks += 1
        
        # Calculate aspect ratio
        all_points = []
        for idx in range(17):
            point = self._get_keypoint(keypoints, idx)
            if point:
                all_points.append(point)
        
        if len(all_points) >= 4:
            x_values = [p[0] for p in all_points]
            y_values = [p[1] for p in all_points]
            
            width = max(x_values) - min(x_values)
            height = max(y_values) - min(y_values)
            
            if height > 0:
                aspect_ratio = height / width
                if aspect_ratio < self.lying_aspect_ratio:  # Wider than tall
                    checks_passed += 1
            total_checks += 1
        
        if total_checks > 0:
            confidence = checks_passed / total_checks
        
        return confidence > 0.5, confidence
    
    def classify_activity(
        self, 
        person_id: int, 
        keypoints: np.ndarray,
        timestamp: float = None
    ) -> ActivityState:
        """
        Classify the current activity of a person.
        
        Args:
            person_id: Tracking ID of the person.
            keypoints: Array of pose keypoints.
            timestamp: Current timestamp (uses time.time() if None).
            
        Returns:
            ActivityState with classification results.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Get body center for movement tracking
        body_center = self._get_body_center(keypoints)
        
        # Calculate movement score
        movement_score = 0.0
        if body_center:
            movement_score = self._calculate_movement(person_id, body_center)
        
        is_moving = movement_score > self.movement_threshold
        
        # Check each activity type
        is_lying, lying_conf = self._is_lying_down(keypoints)
        is_sitting, sitting_conf = self._is_sitting(keypoints)
        is_standing, standing_conf = self._is_standing(keypoints)
        
        # Determine activity based on confidence scores
        activity = ActivityType.UNKNOWN
        confidence = 0.0
        
        # Priority: Lying > Sitting > Standing (based on confidence)
        if is_lying and lying_conf > 0.4:
            activity = ActivityType.LYING_DOWN
            confidence = lying_conf
        elif is_sitting and sitting_conf > sitting_conf * 0.8:
            activity = ActivityType.SITTING
            confidence = sitting_conf
        elif is_standing and standing_conf > 0.4:
            # Check if walking (standing + moving)
            if movement_score > self.walking_movement_threshold:
                activity = ActivityType.WALKING
            else:
                activity = ActivityType.STANDING
            confidence = standing_conf
        else:
            # Default based on highest confidence
            confidences = {
                ActivityType.LYING_DOWN: lying_conf,
                ActivityType.SITTING: sitting_conf,
                ActivityType.STANDING: standing_conf
            }
            activity = max(confidences, key=confidences.get)
            confidence = confidences[activity]
        
        # Calculate duration in current state
        duration = 0.0
        if person_id in self.activity_states:
            prev_state = self.activity_states[person_id]
            if prev_state.activity == activity:
                prev_time = self.last_update_time.get(person_id, timestamp)
                duration = prev_state.duration + (timestamp - prev_time)
        
        # Check for inactivity
        if not is_moving and duration > self.inactivity_duration:
            activity = ActivityType.INACTIVE
        
        # Create new state
        state = ActivityState(
            activity=activity,
            confidence=confidence,
            duration=duration,
            movement_score=movement_score,
            is_moving=is_moving,
            last_position=body_center
        )
        
        # Update tracking
        self.activity_states[person_id] = state
        self.last_update_time[person_id] = timestamp
        
        return state
    
    def get_activity_summary(self, person_id: int) -> Dict:
        """
        Get summary of person's activity state.
        
        Args:
            person_id: Tracking ID.
            
        Returns:
            Dictionary with activity information.
        """
        if person_id not in self.activity_states:
            return {
                "activity": "unknown",
                "confidence": 0.0,
                "duration": 0.0,
                "movement_score": 0.0,
                "is_moving": False
            }
        
        state = self.activity_states[person_id]
        return {
            "activity": state.activity.value,
            "confidence": state.confidence,
            "duration": state.duration,
            "movement_score": state.movement_score,
            "is_moving": state.is_moving,
            "position": state.last_position
        }
    
    def reset_person(self, person_id: int) -> None:
        """Reset tracking for a person."""
        self.activity_states.pop(person_id, None)
        self.position_history.pop(person_id, None)
        self.last_update_time.pop(person_id, None)

