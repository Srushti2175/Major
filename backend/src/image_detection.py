"""
Image Human Position Detection

This script processes single images and performs human position detection
and pose estimation.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import HumanPositionDetector
from src.utils import load_config, setup_output_dirs


def main():
    """Main function for image detection."""
    parser = argparse.ArgumentParser(
        description="Human Position Detection from Image"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image file"
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
        help="Output image path (default: auto-generated)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the result image"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detection data to JSON"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directories
    output_dirs = setup_output_dirs(config)
    
    # Initialize detector
    print("=" * 60)
    print("HUMAN POSITION DETECTION - IMAGE MODE")
    print("=" * 60)
    print(f"Input image: {image_path}")
    
    detector = HumanPositionDetector(args.config)
    
    # Setup output image path
    if args.output:
        output_path = args.output
    else:
        output_filename = f"detected_{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{image_path.suffix}"
        output_path = str(output_dirs['video'] / output_filename)
    
    print(f"Output will be saved to: {output_path}")
    
    # Process image
    print("\nProcessing image...")
    
    annotated_image, detections = detector.detect_image(
        str(image_path),
        output_path=output_path,
        show=args.show
    )
    
    # Get position summary
    summary = detector.get_position_summary(detections)
    
    # Print results
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"Total humans detected: {summary['total_humans']}")
    
    for pos in summary['positions']:
        print(f"\nPerson ID {pos['id']}:")
        print(f"  Position: Center ({pos['center']['x']:.1f}, {pos['center']['y']:.1f})")
        print(f"  Bounding Box: ({pos['bbox']['x1']:.1f}, {pos['bbox']['y1']:.1f}) to ({pos['bbox']['x2']:.1f}, {pos['bbox']['y2']:.1f})")
        print(f"  Size: {pos['bbox']['width']:.1f} x {pos['bbox']['height']:.1f}")
        print(f"  Confidence: {pos['confidence']:.2%}")
        
        if pos['keypoints']:
            print("  Keypoints detected:")
            for kp_name, kp_data in pos['keypoints'].items():
                if kp_data['confidence'] > 0.5:
                    print(f"    - {kp_name}: ({kp_data['x']:.1f}, {kp_data['y']:.1f}) [{kp_data['confidence']:.2%}]")
    
    # Save JSON data
    if args.save_json:
        json_filename = f"detection_{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = output_dirs['json'] / json_filename
        
        json_data = {
            "source_image": str(image_path),
            "processed_at": datetime.now().isoformat(),
            "summary": summary
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nDetection data saved to: {json_path}")
    
    print(f"\nOutput image saved to: {output_path}")


if __name__ == "__main__":
    main()

