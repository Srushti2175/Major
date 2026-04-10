"""
Fall Detection Module for Elderly Care

Detects falls using multiple indicators:
- Sudden change in body center height
- Rapid transition from standing/sitting to lying
- Velocity of body movement
- Body orientation changes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time


class FallStatus(Enum):
    """Fall detection status."""
    NORMAL = "normal"
    FALL_DETECTED = "fall_detected"
    FALL_SUSPECTED = "fall_suspected"  # Possible fall, needs confirmation
    RECOVERING = "recovering"  # Person getting up after fall


@dataclass
class FallEvent:
    """Represents a detected fall event."""
    timestamp: float
    person_id: int
    confidence: float
    fall_type: str  # 'sudden_drop', 'forward_fall', 'backward_fall', 'side_fall'
    velocity: float
    position: Tuple[float, float]
    is_confirmed: bool = False


class FallDetector:
    """
    Detects falls using pose estimation and motion analysis.
    
    Detection Methods:
    1. Vertical velocity - Rapid downward movement
    2. Height ratio change - Sudden decrease in body height
    3. Orientation change - Transition to horizontal position
    4. Impact detection - Sudden stop after rapid movement
    """
    
    # Keypoint indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self, config: Dict = None):
        """
        Initialize fall detector.
        
        Args:
            config: Configuration dictionary.
        """
        config = config or {}
        fall_config = config.get('fall_detection', {})
        
        # Detection thresholds (all configurable)
        self.velocity_threshold = fall_config.get('velocity_threshold', 150)  # pixels/frame
        self.height_drop_ratio = fall_config.get('height_drop_ratio', 0.4)  # 40% height drop
        self.orientation_threshold = fall_config.get('orientation_threshold', 60)  # degrees
        self.confirmation_frames = fall_config.get('confirmation_frames', 5)
        self.cooldown_duration = fall_config.get('cooldown_duration', 5.0)  # seconds
        
        # History tracking
        self.history_length = fall_config.get('history_length', 15)
        self.position_history: Dict[int, deque] = {}
        self.height_history: Dict[int, deque] = {}
        self.orientation_history: Dict[int, deque] = {}
        self.timestamp_history: Dict[int, deque] = {}
        
        # Fall state tracking
        self.fall_states: Dict[int, FallStatus] = {}
        self.fall_events: Dict[int, List[FallEvent]] = {}
        self.suspected_fall_frames: Dict[int, int] = {}
        self.last_fall_time: Dict[int, float] = {}
        
    def _get_keypoint(
        self, 
        keypoints: np.ndarray, 
        idx: int, 
        conf_threshold: float = 0.3
    ) -> Optional[Tuple[float, float]]:
        """Get keypoint if confidence is sufficient."""
        if idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        if len(kp) > 2 and kp[2] < conf_threshold:
            return None
        return (float(kp[0]), float(kp[1]))
    
    def _get_body_center(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate body center from keypoints."""
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        
        if left_hip and right_hip:
            return ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        
        if left_shoulder and right_shoulder:
            return ((left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2)
        return None
    
    def _get_body_height(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Calculate the vertical extent of the body.
        
        Returns distance from highest to lowest visible keypoint.
        """
        y_values = []
        for idx in range(17):
            kp = self._get_keypoint(keypoints, idx)
            if kp:
                y_values.append(kp[1])
        
        if len(y_values) < 3:
            return None
        
        return max(y_values) - min(y_values)
    
    def _get_body_orientation(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Calculate body orientation angle.
        
        Returns angle from vertical (0 = upright, 90 = horizontal).
        """
        # Use shoulder-hip line for orientation
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        
        shoulder_center = None
        hip_center = None
        
        if left_shoulder and right_shoulder:
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2)
        if left_hip and right_hip:
            hip_center = ((left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2)
        
        if not shoulder_center or not hip_center:
            return None
        
        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        
        # Angle from vertical
        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
        return angle
    
    def _calculate_velocity(self, person_id: int) -> Tuple[float, float]:
        """
        Calculate velocity from position history.
        
        Returns (horizontal_velocity, vertical_velocity).
        """
        if person_id not in self.position_history:
            return (0.0, 0.0)
        
        history = self.position_history[person_id]
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Use last few frames for velocity calculation
        frames_to_use = min(5, len(history))
        
        velocities_x = []
        velocities_y = []
        
        history_list = list(history)
        for i in range(-frames_to_use + 1, 0):
            prev = history_list[i - 1]
            curr = history_list[i]
            velocities_x.append(curr[0] - prev[0])
            velocities_y.append(curr[1] - prev[1])
        
        avg_vx = np.mean(velocities_x) if velocities_x else 0.0
        avg_vy = np.mean(velocities_y) if velocities_y else 0.0
        
        return (avg_vx, avg_vy)
    
    def _check_sudden_drop(self, person_id: int, current_center: Tuple[float, float]) -> bool:
        """Check for sudden vertical drop."""
        if person_id not in self.position_history:
            return False
        
        history = self.position_history[person_id]
        if len(history) < 5:
            return False
        
        # Compare current Y with average Y from recent history
        recent_y = [p[1] for p in list(history)[-10:]]
        avg_y = np.mean(recent_y)
        
        # In image coordinates, Y increases downward
        drop = current_center[1] - avg_y
        
        return drop > self.velocity_threshold
    
    def _check_height_change(self, person_id: int, current_height: float) -> bool:
        """Check for sudden decrease in body height."""
        if person_id not in self.height_history:
            return False
        
        history = self.height_history[person_id]
        if len(history) < 5:
            return False
        
        # Get average height from history
        avg_height = np.mean(list(history))
        
        if avg_height == 0:
            return False
        
        height_ratio = current_height / avg_height
        
        return height_ratio < (1 - self.height_drop_ratio)
    
    def _check_orientation_change(self, person_id: int, current_angle: float) -> bool:
        """Check for sudden change to horizontal orientation."""
        if person_id not in self.orientation_history:
            return False
        
        history = self.orientation_history[person_id]
        if len(history) < 5:
            return False
        
        # Get average orientation
        avg_angle = np.mean(list(history))
        
        # Check if changed from upright to horizontal
        was_upright = avg_angle < 30
        is_horizontal = current_angle > self.orientation_threshold
        
        return was_upright and is_horizontal
    
    def _determine_fall_type(
        self, 
        velocity: Tuple[float, float],
        orientation: float
    ) -> str:
        """Determine the type of fall based on motion characteristics."""
        vx, vy = velocity
        
        if abs(vy) > abs(vx) * 2:
            return "sudden_drop"
        elif vx > 0 and abs(vx) > abs(vy):
            return "side_fall"
        elif vx < 0 and abs(vx) > abs(vy):
            return "side_fall"
        elif vy > 0:
            return "forward_fall" if orientation > 45 else "backward_fall"
        else:
            return "unknown"
    
    def detect_fall(
        self,
        person_id: int,
        keypoints: np.ndarray,
        timestamp: float = None
    ) -> Tuple[FallStatus, Optional[FallEvent]]:
        """
        Detect if a fall has occurred.
        
        Args:
            person_id: Tracking ID.
            keypoints: Pose keypoints array.
            timestamp: Current timestamp.
            
        Returns:
            Tuple of (FallStatus, FallEvent or None).
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize histories if needed
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=self.history_length)
            self.height_history[person_id] = deque(maxlen=self.history_length)
            self.orientation_history[person_id] = deque(maxlen=self.history_length)
            self.timestamp_history[person_id] = deque(maxlen=self.history_length)
            self.fall_states[person_id] = FallStatus.NORMAL
            self.fall_events[person_id] = []
        
        # Check cooldown
        if person_id in self.last_fall_time:
            if timestamp - self.last_fall_time[person_id] < self.cooldown_duration:
                return self.fall_states[person_id], None
        
        # Get current measurements
        body_center = self._get_body_center(keypoints)
        body_height = self._get_body_height(keypoints)
        orientation = self._get_body_orientation(keypoints)
        
        if body_center is None:
            return self.fall_states[person_id], None
        
        # Update histories
        self.position_history[person_id].append(body_center)
        self.timestamp_history[person_id].append(timestamp)
        
        if body_height is not None:
            self.height_history[person_id].append(body_height)
        if orientation is not None:
            self.orientation_history[person_id].append(orientation)
        
        # Calculate velocity
        velocity = self._calculate_velocity(person_id)
        
        # Check fall indicators
        indicators = []
        
        # 1. Sudden vertical drop
        if self._check_sudden_drop(person_id, body_center):
            indicators.append("sudden_drop")
        
        # 2. Body height decrease
        if body_height and self._check_height_change(person_id, body_height):
            indicators.append("height_drop")
        
        # 3. Orientation change
        if orientation and self._check_orientation_change(person_id, orientation):
            indicators.append("orientation_change")
        
        # 4. High downward velocity
        if velocity[1] > self.velocity_threshold:
            indicators.append("high_velocity")
        
        # Determine fall status
        fall_event = None
        
        if len(indicators) >= 2:
            # Multiple indicators - likely fall
            if person_id not in self.suspected_fall_frames:
                self.suspected_fall_frames[person_id] = 0
            
            self.suspected_fall_frames[person_id] += 1
            
            if self.suspected_fall_frames[person_id] >= self.confirmation_frames:
                # Confirmed fall
                confidence = min(1.0, len(indicators) / 4.0 + 0.5)
                fall_type = self._determine_fall_type(velocity, orientation or 0)
                
                fall_event = FallEvent(
                    timestamp=timestamp,
                    person_id=person_id,
                    confidence=confidence,
                    fall_type=fall_type,
                    velocity=np.sqrt(velocity[0]**2 + velocity[1]**2),
                    position=body_center,
                    is_confirmed=True
                )
                
                self.fall_states[person_id] = FallStatus.FALL_DETECTED
                self.fall_events[person_id].append(fall_event)
                self.last_fall_time[person_id] = timestamp
                self.suspected_fall_frames[person_id] = 0
            else:
                self.fall_states[person_id] = FallStatus.FALL_SUSPECTED
        
        elif len(indicators) == 1:
            # Single indicator - possible fall
            if person_id not in self.suspected_fall_frames:
                self.suspected_fall_frames[person_id] = 0
            self.suspected_fall_frames[person_id] += 1
            self.fall_states[person_id] = FallStatus.FALL_SUSPECTED
        
        else:
            # No indicators - normal
            self.suspected_fall_frames[person_id] = 0
            
            # Check if recovering from fall
            if self.fall_states[person_id] == FallStatus.FALL_DETECTED:
                if orientation and orientation < 30:  # Back to upright
                    self.fall_states[person_id] = FallStatus.RECOVERING
                    
            elif self.fall_states[person_id] == FallStatus.RECOVERING:
                self.fall_states[person_id] = FallStatus.NORMAL
            else:
                self.fall_states[person_id] = FallStatus.NORMAL
        
        return self.fall_states[person_id], fall_event
    
    def get_fall_history(self, person_id: int) -> List[FallEvent]:
        """Get fall event history for a person."""
        return self.fall_events.get(person_id, [])
    
    def get_status(self, person_id: int) -> Dict:
        """Get current fall detection status."""
        return {
            "status": self.fall_states.get(person_id, FallStatus.NORMAL).value,
            "suspected_frames": self.suspected_fall_frames.get(person_id, 0),
            "total_falls": len(self.fall_events.get(person_id, [])),
            "last_fall_time": self.last_fall_time.get(person_id)
        }
    
    def reset_person(self, person_id: int) -> None:
        """Reset tracking for a person."""
        self.position_history.pop(person_id, None)
        self.height_history.pop(person_id, None)
        self.orientation_history.pop(person_id, None)
        self.timestamp_history.pop(person_id, None)
        self.fall_states.pop(person_id, None)
        self.suspected_fall_frames.pop(person_id, None)

