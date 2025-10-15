#!/usr/bin/env python3
"""
Server Startup Script
Starts both the Eye Tracking Server (port 5001) and Main Application (port 5000)
"""

import subprocess
import time
import sys
import signal
import os

# Store process references
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nShutting down servers...')
    for process in processes:
        process.terminate()
    sys.exit(0)

def start_eye_tracking_server():
    """Start the Eye Tracking Server on port 5001"""
    print("Starting Eye Tracking Server on port 5001...")
    process = subprocess.Popen(
        [sys.executable, 'server/eye_tracking_server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(process)
    return process

def start_main_app():
    """Start the Main Application on port 5000"""
    print("Starting Main Application on port 5000...")
    process = subprocess.Popen(
        [sys.executable, 'server/main_app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(process)
    return process

def main():
    """Main startup function"""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("Eye Exercise Tracker - Starting Servers")
    print("=" * 60)
    
    # Start Eye Tracking Server first
    tracking_server = start_eye_tracking_server()
    
    # Wait a moment for it to initialize
    time.sleep(2)
    
    # Start Main Application
    main_app = start_main_app()
    
    print("\n" + "=" * 60)
    print("Servers started successfully!")
    print("=" * 60)
    print(f"Eye Tracking Server: http://localhost:5001")
    print(f"Main Application: http://localhost:5000")
    print("=" * 60)
    print("\nPress Ctrl+C to stop servers")
    print("=" * 60 + "\n")
    
    # Monitor processes and print their output
    try:
        while True:
            # Check if processes are still running
            for process in processes:
                if process.poll() is not None:
                    print(f"\nWarning: A server process has stopped unexpectedly")
                    return
            
            # Print output from both processes
            for process in processes:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()
