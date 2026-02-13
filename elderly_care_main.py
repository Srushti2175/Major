"""
Elderly Care AI Monitoring System - Main Entry Point

This script provides a comprehensive elderly monitoring solution with:
- Real-time video monitoring
- Activity detection (sitting, standing, walking, lying)
- Fall detection
- Emotion detection
- Inactivity alerts
- Complete logging and export
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.elderly_care_monitor import ElderlyCareMonitor, Alert
from src.utils import load_config, setup_output_dirs, generate_output_filename


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🏥 ELDERLY CARE AI MONITORING SYSTEM 🏥                          ║
║                                                                      ║
║     Features:                                                        ║
║     ✓ Real-time Activity Detection                                   ║
║     ✓ Fall Detection with Alerts                                     ║
║     ✓ Emotion Recognition                                            ║
║     ✓ Inactivity Monitoring                                          ║
║     ✓ Comprehensive Logging                                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def alert_callback(alert: Alert):
    """Callback function for handling alerts."""
    # This can be extended to send notifications via:
    # - SMS (Twilio)
    # - Email (SMTP)
    # - Push notifications
    # - WhatsApp
    # - Database logging
    
    # For now, just log to file
    log_file = Path("outputs/logs/alerts.json")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    alerts = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                alerts = json.load(f)
            except json.JSONDecodeError:
                alerts = []
    
    alerts.append(alert.to_dict())
    
    with open(log_file, 'w') as f:
        json.dump(alerts, f, indent=2)


def run_live_monitoring(args, config):
    """Run live webcam monitoring."""
    print("\n🎥 Starting Live Monitoring...")
    print("Press 'Q' or 'ESC' to stop\n")
    
    # Initialize monitor
    monitor = ElderlyCareMonitor(args.config)
    monitor.register_alert_callback(alert_callback)
    
    # Setup output
    output_dirs = setup_output_dirs(config)
    
    output_path = None
    if args.save:
        output_filename = generate_output_filename("elderly_care_live", "mp4")
        output_path = str(output_dirs['video'] / output_filename)
        print(f"📁 Output will be saved to: {output_path}")
    
    # Run monitoring
    frame_count = 0
    total_detections = 0
    
    try:
        for frame, statuses, frame_num in monitor.process_video(
            source=args.camera,
            output_path=output_path,
            show=not args.no_display
        ):
            frame_count = frame_num
            total_detections += len(statuses)
            
            # Print status every 60 frames
            if frame_num % 60 == 0 and statuses:
                print(f"\n📊 Frame {frame_num} Status:")
                for status in statuses:
                    print(f"   Person {status.person_id}: "
                          f"{status.activity.upper()} | "
                          f"Emotion: {status.emotion} | "
                          f"Movement: {status.movement_score:.1f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring stopped by user")
    
    finally:
        # Print summary
        summary = monitor.get_monitoring_summary()
        print("\n" + "="*60)
        print("📈 MONITORING SESSION SUMMARY")
        print("="*60)
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Total alerts: {summary['total_alerts']}")
        print(f"   Alerts by type: {summary['alerts_by_type']}")
        
        # Save session log
        if args.save_log:
            log_filename = generate_output_filename("session_log", "json")
            log_path = output_dirs['json'] / log_filename
            monitor.save_session_log(str(log_path))


def run_video_analysis(args, config):
    """Analyze a video file."""
    video_path = Path(args.video_path)
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"\n🎬 Analyzing Video: {video_path}")
    print("Press 'Q' or 'ESC' to stop\n")
    
    # Initialize monitor
    monitor = ElderlyCareMonitor(args.config)
    monitor.register_alert_callback(alert_callback)
    
    # Setup output
    output_dirs = setup_output_dirs(config)
    
    output_path = None
    if not args.no_save:
        output_filename = f"analyzed_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = str(output_dirs['video'] / output_filename)
        print(f"📁 Output will be saved to: {output_path}")
    
    # Run analysis
    frame_count = 0
    all_statuses = []
    
    try:
        for frame, statuses, frame_num in monitor.process_video(
            source=str(video_path),
            output_path=output_path,
            show=not args.no_display
        ):
            frame_count = frame_num
            
            for status in statuses:
                all_statuses.append(status.to_dict())
            
            # Print progress
            if frame_num % 100 == 0:
                print(f"   Processed: {frame_num} frames")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis stopped by user")
    
    finally:
        # Print summary
        summary = monitor.get_monitoring_summary()
        print("\n" + "="*60)
        print("📈 VIDEO ANALYSIS SUMMARY")
        print("="*60)
        print(f"   Video: {video_path.name}")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total alerts: {summary['total_alerts']}")
        print(f"   Alerts by type: {summary['alerts_by_type']}")
        
        # Save detailed analysis
        if args.save_json:
            json_filename = f"analysis_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_path = output_dirs['json'] / json_filename
            
            analysis_data = {
                "video_file": str(video_path),
                "analyzed_at": datetime.now().isoformat(),
                "total_frames": frame_count,
                "summary": summary,
                "detections": all_statuses
            }
            
            with open(json_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            print(f"   Analysis saved to: {json_path}")
        
        # Save session log
        if args.save_log:
            log_filename = f"session_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = output_dirs['json'] / log_filename
            monitor.save_session_log(str(log_path))


def main():
    """Main entry point."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Elderly Care AI Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Monitoring mode")
    
    # Live monitoring subparser
    live_parser = subparsers.add_parser("live", help="Live webcam monitoring")
    live_parser.add_argument("--config", type=str, help="Config file path")
    live_parser.add_argument("--camera", type=int, default=0, help="Camera index")
    live_parser.add_argument("--save", action="store_true", help="Save output video")
    live_parser.add_argument("--no-display", action="store_true", help="Disable display")
    live_parser.add_argument("--save-log", action="store_true", default=True, help="Save session log")
    
    # Video analysis subparser
    video_parser = subparsers.add_parser("video", help="Analyze video file")
    video_parser.add_argument("video_path", type=str, help="Input video path")
    video_parser.add_argument("--config", type=str, help="Config file path")
    video_parser.add_argument("--output", type=str, help="Output video path")
    video_parser.add_argument("--no-display", action="store_true", help="Disable display")
    video_parser.add_argument("--no-save", action="store_true", help="Don't save output")
    video_parser.add_argument("--save-json", action="store_true", help="Save JSON analysis")
    video_parser.add_argument("--save-log", action="store_true", default=True, help="Save session log")
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        print("\n⚠️  Please specify a mode: live or video")
        print("\nExamples:")
        print("  python elderly_care_main.py live")
        print("  python elderly_care_main.py live --camera 1 --save")
        print("  python elderly_care_main.py video path/to/video.mp4")
        print("  python elderly_care_main.py video path/to/video.mp4 --save-json")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Run appropriate mode
    if args.mode == "live":
        run_live_monitoring(args, config)
    elif args.mode == "video":
        run_video_analysis(args, config)


if __name__ == "__main__":
    main()

