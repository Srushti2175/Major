"""
Utility functions for Human Position Detection
"""

import os
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
        
    Returns:
        Dictionary containing configuration values.
    """
    if config_path is None:
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create output directories based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary with paths to output directories.
    """
    output_config = config.get('output', {})
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create output directories
    dirs = {}
    
    video_dir = project_root / output_config.get('output_dir', 'outputs')
    video_dir.mkdir(parents=True, exist_ok=True)
    dirs['video'] = video_dir
    
    json_dir = project_root / output_config.get('json_dir', 'outputs/detections')
    json_dir.mkdir(parents=True, exist_ok=True)
    dirs['json'] = json_dir
    
    return dirs


def get_device(device_config: str) -> str:
    """
    Determine the appropriate device for model inference.
    
    Args:
        device_config: Device configuration ('auto', 'cuda', 'cpu', or device id).
        
    Returns:
        Device string for PyTorch.
    """
    import torch
    
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    return str(device_config)


def generate_output_filename(prefix: str, extension: str) -> str:
    """
    Generate a unique output filename with timestamp.
    
    Args:
        prefix: Filename prefix.
        extension: File extension (without dot).
        
    Returns:
        Unique filename string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def save_detection_data(
    detections: list,
    output_path: Path,
    frame_number: int,
    timestamp: float
) -> None:
    """
    Save detection data to JSON file.
    
    Args:
        detections: List of detection dictionaries.
        output_path: Path to save JSON file.
        frame_number: Current frame number.
        timestamp: Video timestamp in seconds.
    """
    data = {
        "frame": frame_number,
        "timestamp": timestamp,
        "detections": detections
    }
    
    # Append to existing file or create new
    json_file = output_path / "detections.json"
    
    existing_data = []
    if json_file.exists():
        with open(json_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    
    existing_data.append(data)
    
    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=2)


def format_keypoints(keypoints, keypoint_names: list) -> Dict[str, Dict[str, float]]:
    """
    Format keypoints into a readable dictionary.
    
    Args:
        keypoints: Raw keypoints array from model.
        keypoint_names: List of keypoint names.
        
    Returns:
        Dictionary mapping keypoint names to coordinates and confidence.
    """
    formatted = {}
    
    for i, name in enumerate(keypoint_names):
        if i < len(keypoints):
            kp = keypoints[i]
            formatted[name] = {
                "x": float(kp[0]),
                "y": float(kp[1]),
                "confidence": float(kp[2]) if len(kp) > 2 else 1.0
            }
    
    return formatted


def calculate_center(bbox: list) -> tuple:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2].
        
    Returns:
        Tuple (center_x, center_y).
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y).
        point2: Second point (x, y).
        
    Returns:
        Distance as float.
    """
    import math
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

