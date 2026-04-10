"""
Human Position Detection Module

This module provides the core functionality for detecting human positions
and pose estimation using YOLOv8.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Generator
from dataclasses import dataclass, field

from ultralytics import YOLO

from .utils import load_config, get_device, format_keypoints


@dataclass
class Detection:
    """Represents a single human detection with pose information."""
    
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    center: Tuple[float, float]
    keypoints: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_keypoints: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "center": {"x": self.center[0], "y": self.center[1]},
            "keypoints": self.keypoints
        }


class HumanPositionDetector:
    """
    Human Position Detection and Pose Estimation using YOLOv8.
    
    This class provides methods to detect humans and estimate their pose
    in images, video files, and live camera feeds.
    
    Attributes:
        config: Configuration dictionary.
        model: YOLOv8 pose estimation model.
        device: Device for inference (cuda/cpu).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Human Position Detector.
        
        Args:
            config_path: Path to configuration file. Uses default if None.
        """
        self.config = load_config(config_path)
        self._setup_model()
        self._setup_visualization()
    
    def _setup_model(self) -> None:
        """Initialize the YOLOv8 pose estimation model."""
        model_config = self.config.get('model', {})
        
        model_name = model_config.get('name', 'yolov8m-pose')
        self.device = get_device(model_config.get('device', 'auto'))
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load YOLOv8 pose model
        self.model = YOLO(model_name)
        
        # Store thresholds
        self.conf_threshold = model_config.get('confidence_threshold', 0.5)
        self.iou_threshold = model_config.get('iou_threshold', 0.45)
        
        # Detection settings
        detection_config = self.config.get('detection', {})
        self.classes = detection_config.get('classes', [0])  # 0 = person
        self.max_detections = detection_config.get('max_detections', 100)
        self.enable_tracking = detection_config.get('enable_tracking', True)
        self.tracker = detection_config.get('tracker', 'bytetrack')
    
    def _setup_visualization(self) -> None:
        """Setup visualization parameters."""
        vis_config = self.config.get('visualization', {})
        pose_config = self.config.get('pose', {})
        keypoint_config = self.config.get('keypoints', {})
        
        # Colors (convert from list to tuple)
        self.bbox_color = tuple(vis_config.get('bbox_color', [0, 255, 0]))
        self.text_color = tuple(vis_config.get('text_color', [255, 255, 255]))
        
        # Thicknesses and sizes
        self.bbox_thickness = vis_config.get('bbox_thickness', 2)
        self.font_scale = vis_config.get('font_scale', 0.6)
        
        # Pose visualization
        self.draw_skeleton = pose_config.get('draw_skeleton', True)
        self.draw_keypoints = pose_config.get('draw_keypoints', True)
        self.skeleton_thickness = pose_config.get('skeleton_thickness', 2)
        self.keypoint_radius = pose_config.get('keypoint_radius', 5)
        self.keypoint_conf_threshold = pose_config.get('keypoint_confidence', 0.5)
        
        # Display options
        self.show_confidence = vis_config.get('show_confidence', True)
        self.show_id = vis_config.get('show_id', True)
        
        # Keypoint configuration
        self.keypoint_names = keypoint_config.get('names', [])
        self.skeleton_connections = keypoint_config.get('skeleton', [])
        
        # Skeleton colors (gradient from blue to red for different body parts)
        self.skeleton_colors = [
            (255, 128, 0),   # Head connections (cyan-ish)
            (255, 128, 0),
            (255, 128, 0),
            (255, 128, 0),
            (0, 255, 255),   # Shoulders (yellow)
            (0, 255, 0),     # Left arm (green)
            (0, 255, 0),
            (0, 128, 255),   # Right arm (orange)
            (0, 128, 255),
            (255, 0, 255),   # Torso (magenta)
            (255, 0, 255),
            (255, 255, 0),   # Hips (cyan)
            (0, 255, 0),     # Left leg (green)
            (0, 255, 0),
            (0, 128, 255),   # Right leg (orange)
            (0, 128, 255),
        ]
    
    def detect_frame(
        self,
        frame: np.ndarray,
        track: bool = None
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detect humans and their poses in a single frame.
        
        Args:
            frame: Input frame (BGR format).
            track: Whether to use tracking. If None, uses config value.
            
        Returns:
            Tuple of (annotated_frame, list of Detection objects).
        """
        if track is None:
            track = self.enable_tracking
        
        # Run inference
        if track:
            results = self.model.track(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                max_det=self.max_detections,
                tracker=f"{self.tracker}.yaml",
                persist=True,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                max_det=self.max_detections,
                verbose=False
            )
        
        # Process results
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            keypoints_data = result.keypoints if hasattr(result, 'keypoints') and result.keypoints is not None else None
            
            for i, box in enumerate(boxes):
                # Get bounding box
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                
                # Get tracking ID if available
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else i
                
                # Calculate center
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                # Get keypoints if available
                raw_kps = None
                formatted_kps = {}
                
                if keypoints_data is not None and i < len(keypoints_data):
                    raw_kps = keypoints_data[i].data[0].cpu().numpy()
                    formatted_kps = format_keypoints(raw_kps, self.keypoint_names)
                
                # Create detection object
                detection = Detection(
                    id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    center=center,
                    keypoints=formatted_kps,
                    raw_keypoints=raw_kps
                )
                detections.append(detection)
                
                # Draw on frame
                annotated_frame = self._draw_detection(annotated_frame, detection)
        
        return annotated_frame, detections
    
    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """
        Draw detection visualization on frame.
        
        Args:
            frame: Input frame.
            detection: Detection object to visualize.
            
        Returns:
            Frame with detection visualization.
        """
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
        
        # Prepare label
        label_parts = []
        if self.show_id:
            label_parts.append(f"ID:{detection.id}")
        if self.show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                self.bbox_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                2
            )
        
        # Draw pose skeleton
        if detection.raw_keypoints is not None:
            frame = self._draw_pose(frame, detection.raw_keypoints)
        
        return frame
    
    def _draw_pose(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Input frame.
            keypoints: Array of keypoints [N, 3] (x, y, confidence).
            
        Returns:
            Frame with pose visualization.
        """
        # Draw skeleton connections
        if self.draw_skeleton:
            for idx, (start_idx, end_idx) in enumerate(self.skeleton_connections):
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    
                    # Check confidence
                    start_conf = start_kp[2] if len(start_kp) > 2 else 1.0
                    end_conf = end_kp[2] if len(end_kp) > 2 else 1.0
                    
                    if start_conf >= self.keypoint_conf_threshold and end_conf >= self.keypoint_conf_threshold:
                        start_point = (int(start_kp[0]), int(start_kp[1]))
                        end_point = (int(end_kp[0]), int(end_kp[1]))
                        
                        color = self.skeleton_colors[idx % len(self.skeleton_colors)]
                        cv2.line(frame, start_point, end_point, color, self.skeleton_thickness)
        
        # Draw keypoints
        if self.draw_keypoints:
            for i, kp in enumerate(keypoints):
                conf = kp[2] if len(kp) > 2 else 1.0
                
                if conf >= self.keypoint_conf_threshold:
                    point = (int(kp[0]), int(kp[1]))
                    
                    # Color based on confidence
                    intensity = int(conf * 255)
                    color = (intensity, intensity, 255)  # Red-ish with intensity
                    
                    cv2.circle(frame, point, self.keypoint_radius, color, -1)
                    cv2.circle(frame, point, self.keypoint_radius, (0, 0, 0), 1)
        
        return frame
    
    def process_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        show: bool = True,
        callback: Optional[callable] = None
    ) -> Generator[Tuple[np.ndarray, List[Detection], int], None, None]:
        """
        Process video file or camera stream.
        
        Args:
            source: Video file path or camera index (0, 1, etc.).
            output_path: Path to save output video. None to skip saving.
            show: Whether to display the video in a window.
            callback: Optional callback function called for each frame.
                     Signature: callback(frame, detections, frame_num)
                     
        Yields:
            Tuple of (annotated_frame, detections, frame_number).
        """
        video_config = self.config.get('video', {})
        
        # Open video source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Resize if configured
        target_width = video_config.get('width')
        target_height = video_config.get('height')
        
        if target_width and target_height:
            width, height = target_width, target_height
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            output_config = self.config.get('output', {})
            codec = output_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        fps_limit = video_config.get('fps_limit')
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Resize if needed
                if target_width and target_height:
                    frame = cv2.resize(frame, (target_width, target_height))
                
                # Process frame
                annotated_frame, detections = self.detect_frame(frame)
                
                # Write to output
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                if show:
                    # Add FPS info
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_num} | Detections: {len(detections)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imshow("Human Position Detection", annotated_frame)
                    
                    # Check for exit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                
                # Callback
                if callback:
                    callback(annotated_frame, detections, frame_num)
                
                yield annotated_frame, detections, frame_num
                
                frame_num += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
    
    def detect_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = False
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detect humans in a single image.
        
        Args:
            image_path: Path to input image.
            output_path: Path to save annotated image. None to skip saving.
            show: Whether to display the image.
            
        Returns:
            Tuple of (annotated_image, list of Detection objects).
        """
        # Read image
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Process
        annotated_frame, detections = self.detect_frame(frame, track=False)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"Saved annotated image to: {output_path}")
        
        # Display if requested
        if show:
            cv2.imshow("Human Position Detection", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_frame, detections
    
    def get_position_summary(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Get a summary of detected human positions.
        
        Args:
            detections: List of Detection objects.
            
        Returns:
            Dictionary with position summary.
        """
        summary = {
            "total_humans": len(detections),
            "positions": []
        }
        
        for det in detections:
            position = {
                "id": det.id,
                "center": {"x": det.center[0], "y": det.center[1]},
                "bbox": {
                    "x1": det.bbox[0],
                    "y1": det.bbox[1],
                    "x2": det.bbox[2],
                    "y2": det.bbox[3],
                    "width": det.bbox[2] - det.bbox[0],
                    "height": det.bbox[3] - det.bbox[1]
                },
                "confidence": det.confidence,
                "keypoints": det.keypoints
            }
            summary["positions"].append(position)
        
        return summary

