"""
Video File Human Position Detection

This script processes video files and performs human position detection
and pose estimation on each frame.
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
    """Main function for video file detection."""
    parser = argparse.ArgumentParser(
        description="Human Position Detection from Video File"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: auto-generated in outputs folder)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't show video window during processing"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detection data to JSON"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directories
    output_dirs = setup_output_dirs(config)
    
    # Initialize detector
    print("=" * 60)
    print("HUMAN POSITION DETECTION - VIDEO FILE MODE")
    print("=" * 60)
    print(f"Input video: {video_path}")
    
    detector = HumanPositionDetector(args.config)
    
    # Setup output video path
    output_path = None
    if not args.no_save:
        if args.output:
            output_path = args.output
        else:
            output_filename = f"detected_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = str(output_dirs['video'] / output_filename)
        print(f"Output will be saved to: {output_path}")
    
    # Detection data storage for JSON
    all_detections = []
    frame_count = 0
    total_humans = 0
    
    def detection_callback(frame, detections, frame_num):
        """Callback for each processed frame."""
        nonlocal total_humans
        
        if detections:
            total_humans += len(detections)
        
        if args.save_json and detections:
            frame_data = {
                "frame": frame_num,
                "detections": [det.to_dict() for det in detections]
            }
            all_detections.append(frame_data)
    
    print("\nProcessing video...")
    if not args.no_display:
        print("Press 'Q' or 'ESC' to stop\n")
    
    try:
        # Process video file
        for frame, detections, frame_num in detector.process_video(
            source=str(video_path),
            output_path=output_path,
            show=not args.no_display,
            callback=detection_callback
        ):
            frame_count = frame_num + 1
            
            # Print progress every 100 frames
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames... ({len(detections)} detections in current frame)")
    
    except KeyboardInterrupt:
        print("\n\nProcessing stopped by user.")
    
    finally:
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total human detections: {total_humans}")
        print(f"Average detections per frame: {total_humans / max(frame_count, 1):.2f}")
        
        if output_path and not args.no_save:
            print(f"Output video saved to: {output_path}")
        
        # Save JSON data
        if args.save_json and all_detections:
            json_filename = f"detections_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_path = output_dirs['json'] / json_filename
            
            summary = {
                "source_video": str(video_path),
                "total_frames": frame_count,
                "total_detections": total_humans,
                "processed_at": datetime.now().isoformat(),
                "frames": all_detections
            }
            
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Detection data saved to: {json_path}")


if __name__ == "__main__":
    main()

