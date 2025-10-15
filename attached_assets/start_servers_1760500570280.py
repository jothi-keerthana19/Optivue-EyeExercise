import subprocess
import time
import threading
import os

def start_enhanced_eye_tracking_server():
    """Start the enhanced eye tracking server"""
    print("Starting Enhanced Eye Tracking Server...")
    try:
        # Change to the project directory
        os.chdir("d:\\EyeCareAI")

        # Start the enhanced eye tracking server using venv python
        process = subprocess.Popen(
            [".venv\\Scripts\\python", "enhanced_eye_tracking_server.py"],
            text=True
        )

        print("Enhanced Eye Tracking Server started on port 5001")
        return process
    except Exception as e:
        print(f"Error starting enhanced eye tracking server: {e}")
        return None

def start_main_application():
    """Start the main application"""
    print("Starting Main Application...")
    try:
        # Change to the project directory
        os.chdir("d:\\EyeCareAI")
        
        # Start the main application
        process = subprocess.Popen(
            ["python", "app.py"],
            text=True
        )
        
        print("Main Application started on port 5000")
        return process
    except Exception as e:
        print(f"Error starting main application: {e}")
        return None

def main():
    print("Starting EyeCareAI servers...")
    
    # Start the enhanced eye tracking server
    eye_tracking_process = start_enhanced_eye_tracking_server()
    
    # Wait a moment for the first server to start
    time.sleep(3)
    
    # Start the main application
    main_app_process = start_main_application()
    
    # Wait for user input to stop servers
    try:
        input("Press Enter to stop servers...")
    except KeyboardInterrupt:
        pass
    
    # Terminate processes
    if eye_tracking_process:
        eye_tracking_process.terminate()
        print("Enhanced Eye Tracking Server stopped")
    
    if main_app_process:
        main_app_process.terminate()
        print("Main Application stopped")

if __name__ == "__main__":
    main()