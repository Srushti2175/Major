
import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.elderly_care_monitor import ElderlyCareMonitor

def test_monitor():
    print("Testing ElderlyCareMonitor initialization...")
    try:
        monitor = ElderlyCareMonitor()
        print(" Monitor initialized successfully")
        
        # Create a dummy frame (black image)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("Testing process_frame...")
        annotated, statuses = monitor.process_frame(frame)
        print(f" process_frame executed. Statuses: {len(statuses)}")
        
        print(" Validation passed!")
        
    except Exception as e:
        print(f" Tests failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monitor()
