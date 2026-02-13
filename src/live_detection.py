"""
Live Video Human Position Detection

This script captures video from a webcam and performs real-time
human position detection and pose estimation.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import HumanPositionDetector
from src.utils import load_config, setup_output_dirs, generate_output_filename


def main():
    """Main function for live detection."""
    parser = argparse.ArgumentParser(
        description="Live Human Position Detection from Webcam"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index (default: from config or 0)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't show video window"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detection data to JSON"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    camera_source = args.camera if args.camera is not None else config.get('video', {}).get('source', 0)
    
    # Setup output directories
    output_dirs = setup_output_dirs(config)
    
    # Initialize detector
    print("=" * 60)
    print("HUMAN POSITION DETECTION - LIVE MODE")
    print("=" * 60)
    
    detector = HumanPositionDetector(args.config)
    
    # Setup output video path
    output_path = None
    if args.save:
        output_filename = generate_output_filename("live_detection", "mp4")
        output_path = str(output_dirs['video'] / output_filename)
        print(f"Output will be saved to: {output_path}")
    
    # Detection data storage for JSON
    all_detections = []
    
    def detection_callback(frame, detections, frame_num):
        """Callback for each processed frame."""
        if args.save_json and detections:
            frame_data = {
                "frame": frame_num,
                "timestamp": datetime.now().isoformat(),
                "detections": [det.to_dict() for det in detections]
            }
            all_detections.append(frame_data)
    
    print("\nStarting live detection...")
    print("Press 'Q' or 'ESC' to quit\n")
    
    try:
        # Process video stream
        for frame, detections, frame_num in detector.process_video(
            source=camera_source,
            output_path=output_path,
            show=not args.no_display,
            callback=detection_callback
        ):
            # Print detection summary every 30 frames
            if frame_num % 30 == 0 and detections:
                summary = detector.get_position_summary(detections)
                print(f"Frame {frame_num}: {summary['total_humans']} humans detected")
                
                for pos in summary['positions']:
                    print(f"  - ID {pos['id']}: Center ({pos['center']['x']:.1f}, {pos['center']['y']:.1f})")
    
    except KeyboardInterrupt:
        print("\n\nDetection stopped by user.")
    
    finally:
        # Save JSON data
        if args.save_json and all_detections:
            json_filename = generate_output_filename("live_detections", "json")
            json_path = output_dirs['json'] / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(all_detections, f, indent=2)
            
            print(f"\nDetection data saved to: {json_path}")
        
        print("\nLive detection ended.")


if __name__ == "__main__":
    main()

