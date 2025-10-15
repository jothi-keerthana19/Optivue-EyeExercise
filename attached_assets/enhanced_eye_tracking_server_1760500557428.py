from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
from enhanced_eye_tracker import EnhancedEyeTracker
import cv2
import numpy as np
import threading
import time
import base64
import random
from io import BytesIO
from PIL import Image
from typing import Tuple, Any, Optional

class EnhancedEyeTrackingServer:
    def __init__(self):
        print("EnhancedEyeTrackingServer __init__ start")
        try:
            print("Before Flask app creation")
            self.app = Flask(__name__)
            print("After Flask app creation")
            # Enhanced CORS configuration to allow all origins, methods, and headers
            print("Before CORS")
            # CORS(self.app,
            #      origins=['http://localhost:5000', 'http://localhost:5001', 'http://127.0.0.1:5000', 'http://127.0.0.1:5001'],
            #      methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            #      allow_headers=['Content-Type', 'Authorization', 'Accept'],
            #      supports_credentials=True)
            print("After CORS")
            print("Before eye tracker")
            self.eye_tracker = EnhancedEyeTracker()
            print("After eye tracker")
            self.cap = None
            self.camera_active = False
            self.tracking_active = False
            self.current_frame = None
            self.frame_lock = threading.Lock()
            self.target_position = [0.5, 0.5]  # Default target position
            self.session_active = False
            self.simulation_mode = False

            # Start frame reading thread
            self.frame_thread = None
            self.frame_thread_active = False

            print("Before setup_routes")
            self.setup_routes()
            print("After setup_routes")
        except Exception as e:
            print(f"Error in __init__: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('eye_exercises.html')
        
        @self.app.route('/api/enhanced-eye-tracking/status')
        def status():
            return jsonify({
                'status': 'running',
                'camera_status': 'active' if self.camera_active else 'inactive',
                'tracking_active': self.tracking_active,
                'session_active': self.session_active,
                'calibrated': self.eye_tracker.calibration_complete if self.eye_tracker else False
            })
        
        @self.app.route('/api/enhanced-eye-tracking/start_camera', methods=['GET', 'POST'])
        def start_camera():
            try:
                if not self.camera_active:
                    print("Attempting to initialize real camera...")
                    # Try different backends and camera indices
                    backends = [
                        (cv2.CAP_DSHOW, "DSHOW"),
                        (cv2.CAP_MSMF, "MSMF"),
                        (cv2.CAP_ANY, "ANY")
                    ]
                    camera_indices = [0, 1, 2, 3, 4]

                    for backend, backend_name in backends:
                        for index in camera_indices:
                            print(f"Trying camera index {index} with {backend_name} backend...")
                            try:
                                self.cap = cv2.VideoCapture(index, backend)
                                if self.cap.isOpened():
                                    print(f"Camera {index} opened with {backend_name} backend")
                                    # Set camera properties for better performance
                                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                                    # Try to read a test frame
                                    ret, frame = self.cap.read()
                                    if ret and frame is not None:
                                        print(f"Test frame successful - shape: {frame.shape}")
                                        self.camera_active = True
                                        self.simulation_mode = False
                                        # Start frame reading thread
                                        self.frame_thread_active = True
                                        self.frame_thread = threading.Thread(target=self._read_frames)
                                        self.frame_thread.daemon = True
                                        self.frame_thread.start()
                                        print(f"Real camera started successfully with index {index}, backend {backend_name}")
                                        return jsonify({'success': True, 'message': f'Real camera mode enabled (index {index}, {backend_name})'})
                                    else:
                                        print(f"Failed to read test frame from camera {index} with {backend_name}")
                                        self.cap.release()
                                        self.cap = None
                                else:
                                    print(f"Failed to open camera {index} with {backend_name}")
                            except Exception as e:
                                print(f"Error trying camera {index} with {backend_name}: {e}")
                                if self.cap:
                                    self.cap.release()
                                    self.cap = None

                    # No camera worked - return error
                    print("ERROR: No real camera available - cannot start camera")
                    return jsonify({'success': False, 'message': 'No camera available - check camera connection and permissions'})

                return jsonify({'success': True, 'message': 'Camera already active'})
            except Exception as e:
                print(f"Error in start_camera: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/stop_camera', methods=['POST'])
        def stop_camera():
            try:
                if self.camera_active:
                    self.camera_active = False
                    self.tracking_active = False
                    if self.frame_thread_active:
                        self.frame_thread_active = False
                        if self.frame_thread:
                            self.frame_thread.join(timeout=1)
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    return jsonify({'success': True, 'message': 'Camera stopped'})
                return jsonify({'success': True, 'message': 'Camera already stopped'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/start_tracking', methods=['GET', 'POST'])
        def start_tracking():
            try:
                print(f"Start tracking called. Camera active: {self.camera_active}")
                if not self.camera_active:
                    # Try to start camera first with real camera
                    print("Camera not active, trying to start camera")
                    # Try different backends and camera indices
                    backends = [
                        (cv2.CAP_DSHOW, "DSHOW"),
                        (cv2.CAP_MSMF, "MSMF"),
                        (cv2.CAP_ANY, "ANY")
                    ]
                    camera_indices = [0, 1, 2, 3, 4]

                    for backend, backend_name in backends:
                        for index in camera_indices:
                            print(f"Trying camera index {index} with {backend_name} backend for tracking...")
                            try:
                                self.cap = cv2.VideoCapture(index, backend)
                                if self.cap.isOpened():
                                    print(f"Camera {index} opened with {backend_name} backend for tracking")
                                    # Try to read a test frame
                                    ret, frame = self.cap.read()
                                    if ret and frame is not None:
                                        print(f"Test frame successful for tracking - shape: {frame.shape}")
                                        self.camera_active = True
                                        self.simulation_mode = False
                                        # Start frame reading thread
                                        self.frame_thread_active = True
                                        self.frame_thread = threading.Thread(target=self._read_frames)
                                        self.frame_thread.daemon = True
                                        self.frame_thread.start()
                                        print(f"Real camera started for tracking with index {index}, backend {backend_name}")
                                        break
                                    else:
                                        print(f"Failed to read test frame from camera {index} with {backend_name} for tracking")
                                        self.cap.release()
                                else:
                                    print(f"Failed to open camera {index} with {backend_name} for tracking")
                            except Exception as e:
                                print(f"Error trying camera {index} with {backend_name} for tracking: {e}")
                                if self.cap:
                                    self.cap.release()
                        if self.camera_active:
                            break

                    if not self.camera_active:
                        # No camera worked
                        print("ERROR: No real camera available for tracking - all backends and indices failed")
                        return jsonify({'success': False, 'message': 'No camera available for tracking - check camera connection and permissions'}), 500

                self.tracking_active = True
                print("Tracking started with real camera")
                return jsonify({'success': True, 'message': 'Tracking started'})
            except Exception as e:
                print(f"Error in start_tracking: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/stop_tracking', methods=['POST'])
        def stop_tracking():
            try:
                self.tracking_active = False
                return jsonify({'success': True, 'message': 'Tracking stopped'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/start_session', methods=['POST'])
        def start_session():
            try:
                if self.eye_tracker:
                    self.eye_tracker.start_session()
                    self.session_active = True
                    return jsonify({'success': True, 'message': 'Session started'})
                else:
                    return jsonify({'success': False, 'message': 'Eye tracker not initialized'}), 500
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/stop_session', methods=['POST'])
        def stop_session():
            try:
                self.session_active = False
                return jsonify({'success': True, 'message': 'Session stopped'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/get_gaze_data', methods=['GET'])
        def get_gaze_data():
            try:
                # Return current gaze data
                if self.eye_tracker:
                    metrics = self.eye_tracker.get_session_metrics()
                    calibration_status = self.eye_tracker.get_calibration_status()
                else:
                    metrics = {}
                    calibration_status = 'unavailable'
                
                return jsonify({
                    'gaze_x': 50,  # Default center position
                    'gaze_y': 50,  # Default center position
                    'blink_detected': False,
                    'timestamp': time.time(),
                    'metrics': metrics,
                    'calibration_status': calibration_status
                })
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/get_enhanced_gaze', methods=['GET'])
        def get_enhanced_gaze():
            if not self.tracking_active or self.current_frame is None:
                return jsonify({'success': True, 'face_detected': False, 'is_diverted': True, 'is_drowsy': False})

            with self.frame_lock:
                frame_copy = self.current_frame.copy()

            result = self.eye_tracker.process_frame(frame_copy, None, [640, 480])

            # Return the new focus flags
            response_data = {
                'success': True,
                'face_detected': result.get('face_detected', False),
                'is_diverted': result.get('is_diverted', True),
                'is_drowsy': result.get('is_drowsy', False),
                'blink_count': result.get('blink_count', 0)
            }
            return jsonify(response_data)
        
        @self.app.route('/api/enhanced-eye-tracking/set_target_position', methods=['POST'])
        def set_target_position():
            try:
                data = request.get_json()
                if 'x' in data and 'y' in data:
                    self.target_position = [float(data['x']), float(data['y'])]
                    return jsonify({'success': True, 'message': 'Target position updated'})
                return jsonify({'success': False, 'message': 'Invalid target position data'}), 400
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/calibrate', methods=['POST'])
        def calibrate():
            try:
                # For now, we'll just mark calibration as complete
                # In a real implementation, you would collect calibration points
                if self.eye_tracker:
                    self.eye_tracker.calibration_complete = True
                return jsonify({'success': True, 'message': 'Calibration completed'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/reset_session', methods=['POST'])
        def reset_session():
            try:
                if self.eye_tracker:
                    self.eye_tracker.reset_session()
                return jsonify({'success': True, 'message': 'Session reset'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/get_session_metrics', methods=['GET'])
        def get_session_metrics():
            try:
                if self.eye_tracker:
                    metrics = self.eye_tracker.get_session_metrics()
                else:
                    metrics = {}
                return jsonify({'success': True, 'metrics': metrics})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/enhanced-eye-tracking/video_feed')
        def video_feed():
            def generate_frames():
                print("Starting video feed generation")
                # Wait a bit for the first frame to be generated in simulation mode
                wait_count = 0
                while self.camera_active and self.current_frame is None and wait_count < 30:  # Wait up to 1 second
                    time.sleep(0.033)  # ~30 FPS
                    wait_count += 1
                
                frame_counter = 0
                while self.camera_active:
                    with self.frame_lock:
                        if self.current_frame is not None:
                            # Encode frame as JPEG
                            ret, buffer = cv2.imencode('.jpg', self.current_frame)
                            if ret:
                                # Convert to bytes and yield
                                frame_bytes = buffer.tobytes()
                                # Log every 30 frames
                                frame_counter += 1
                                if frame_counter % 30 == 0:
                                    print(f"Sending frame {frame_counter}, size: {len(frame_bytes)} bytes")
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        else:
                            # If no frame is available, generate a blank frame
                            print("No frame available, sending blank frame")
                            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            ret, buffer = cv2.imencode('.jpg', blank_frame)
                            if ret:
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.033)  # ~30 FPS
            
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def _read_frames(self):
        """Read frames from camera in a separate thread"""
        print("Starting frame reading thread")
        frame_count = 0
        while self.frame_thread_active and self.camera_active:
            try:
                if self.cap and self.cap.isOpened():
                    print(f"Reading frame {frame_count}...")
                    capture_result: Tuple[bool, Any] = self.cap.read()
                    print(f"Capture result: {capture_result is not None}, length: {len(capture_result) if capture_result else 'N/A'}")
                    if capture_result and len(capture_result) == 2:
                        ret, frame = capture_result
                        print(f"Frame read result - ret: {ret}, frame is None: {frame is None}")
                        if ret and frame is not None:
                            with self.frame_lock:
                                # Process frame with eye tracker for gaze detection
                                if self.tracking_active and self.eye_tracker:
                                    canvas_size = [640.0, 480.0]
                                    result = self.eye_tracker.process_frame(
                                        frame, 
                                        self.target_position, 
                                        canvas_size
                                    )
                                    # We can draw gaze visualization on the frame here if needed
                                self.current_frame = frame.copy()
                                
                            # Log frame capture for debugging (every 30 frames)
                            frame_count += 1
                            if frame_count % 30 == 0:
                                print(f"Captured frame {frame_count}, shape: {frame.shape if frame is not None else 'None'}")
                        else:
                            print("Failed to capture frame from camera")
                else:
                    print("Camera not opened or cap is None")
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Error reading frame: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        print("Frame reading thread stopped")
    
    def _simulate_frames(self):
        """Simulate camera frames for testing when no camera is available"""
        print("Starting simulation mode - WARNING: This is not real eye tracking!")
        frame_count = 0
        while self.frame_thread_active and self.camera_active:
            try:
                # Create a simulated frame with a moving pattern
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Draw a moving circle to simulate eye movement
                center_x = int(320 + 200 * np.sin(frame_count * 0.1))
                center_y = int(240 + 150 * np.cos(frame_count * 0.1))
                cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
                
                # Add some noise to make it more realistic
                noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Error simulating frame: {e}")
                time.sleep(0.1)
    
    def run(self, host='0.0.0.0', port=5002, debug=True):
        """Run the Flask server"""
        print(f"Starting Enhanced Eye Tracking Server on {host}:{port}")
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            print(f"Error starting server: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    print("Starting enhanced eye tracking server main")
    server = EnhancedEyeTrackingServer()
    print("Server instantiated")
    try:
        server.run()
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        # Clean up
        if server.cap:
            server.cap.release()