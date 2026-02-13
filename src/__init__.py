# Elderly Care AI Monitoring System Package
from .detector import HumanPositionDetector
from .activity_classifier import ActivityClassifier, ActivityType, ActivityState
from .fall_detector import FallDetector, FallStatus, FallEvent
from .emotion_detector import EmotionDetector, EmotionType, EmotionResult
from .elderly_care_monitor import ElderlyCareMonitor, ElderlyStatus, Alert
from .utils import load_config, setup_output_dirs

__all__ = [
    # Core detectors
    "HumanPositionDetector",
    "ActivityClassifier",
    "FallDetector",
    "EmotionDetector",
    "ElderlyCareMonitor",
    
    # Data classes
    "ActivityType",
    "ActivityState",
    "FallStatus",
    "FallEvent",
    "EmotionType",
    "EmotionResult",
    "ElderlyStatus",
    "Alert",
    
    # Utilities
    "load_config",
    "setup_output_dirs"
]

__version__ = "1.0.0"
