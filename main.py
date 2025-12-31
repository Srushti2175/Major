"""
Human Position Detection - Main Entry Point

This script provides a unified interface for all detection modes:
- Live webcam detection
- Video file processing
- Image detection
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Human Position Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live webcam detection
  python main.py live
  
  # Process video file
  python main.py video path/to/video.mp4
  
  # Detect in image
  python main.py image path/to/image.jpg
  
  # Live with options
  python main.py live --camera 1 --save
  
  # Video with JSON output
  python main.py video video.mp4 --save-json
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Detection mode")
    
    # Live detection subparser
    live_parser = subparsers.add_parser("live", help="Live webcam detection")
    live_parser.add_argument("--config", type=str, help="Config file path")
    live_parser.add_argument("--camera", type=int, default=0, help="Camera index")
    live_parser.add_argument("--save", action="store_true", help="Save output video")
    live_parser.add_argument("--no-display", action="store_true", help="No display window")
    live_parser.add_argument("--save-json", action="store_true", help="Save JSON data")
    
    # Video detection subparser
    video_parser = subparsers.add_parser("video", help="Video file processing")
    video_parser.add_argument("video_path", type=str, help="Input video path")
    video_parser.add_argument("--config", type=str, help="Config file path")
    video_parser.add_argument("--output", type=str, help="Output video path")
    video_parser.add_argument("--no-display", action="store_true", help="No display window")
    video_parser.add_argument("--no-save", action="store_true", help="Don't save output")
    video_parser.add_argument("--save-json", action="store_true", help="Save JSON data")
    
    # Image detection subparser
    image_parser = subparsers.add_parser("image", help="Image detection")
    image_parser.add_argument("image_path", type=str, help="Input image path")
    image_parser.add_argument("--config", type=str, help="Config file path")
    image_parser.add_argument("--output", type=str, help="Output image path")
    image_parser.add_argument("--show", action="store_true", help="Display result")
    image_parser.add_argument("--save-json", action="store_true", help="Save JSON data")
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        print("\n⚠️  Please specify a mode: live, video, or image")
        sys.exit(1)
    
    # Convert args to list for subprocess-style calling
    if args.mode == "live":
        from src.live_detection import main as live_main
        # Reconstruct sys.argv for the module
        sys.argv = ["live_detection.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.camera != 0:
            sys.argv.extend(["--camera", str(args.camera)])
        if args.save:
            sys.argv.append("--save")
        if args.no_display:
            sys.argv.append("--no-display")
        if args.save_json:
            sys.argv.append("--save-json")
        live_main()
        
    elif args.mode == "video":
        from src.video_detection import main as video_main
        sys.argv = ["video_detection.py", args.video_path]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.output:
            sys.argv.extend(["--output", args.output])
        if args.no_display:
            sys.argv.append("--no-display")
        if args.no_save:
            sys.argv.append("--no-save")
        if args.save_json:
            sys.argv.append("--save-json")
        video_main()
        
    elif args.mode == "image":
        from src.image_detection import main as image_main
        sys.argv = ["image_detection.py", args.image_path]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.output:
            sys.argv.extend(["--output", args.output])
        if args.show:
            sys.argv.append("--show")
        if args.save_json:
            sys.argv.append("--save-json")
        image_main()


if __name__ == "__main__":
    main()

