"""
Run Elderly Care AI Dashboard

This script starts the web dashboard for monitoring elderly individuals.
Features:
- Real-time video feed with detection overlays
- Activity and emotion status display
- Alert notifications
- Historical charts and analytics
"""

import argparse
import webbrowser
import threading
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard.app import run_dashboard


def open_browser(port):
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{port}')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Elderly Care AI Dashboard Server"
    )
    parser.add_argument(
        '--host', 
        type=str, 
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='Port to run on (default: 5000)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    args = parser.parse_args()
    
    print("""

                                                                      
      ELDERLY CARE AI DASHBOARD                                   
                                                                      
     Starting Web Dashboard...                                        
                                                                      

    """)
    
    # Open browser automatically
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()
    
    # Run dashboard
    run_dashboard(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

