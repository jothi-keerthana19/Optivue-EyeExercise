"""
Eye Tracking API Server
Background service dedicated to computer vision processing.
Manages webcam feed and provides face detection API.
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from simplified_eye_tracker import SimplifiedEyeTracker
import cv2
import numpy as np
import threading
import time
from typing import Optional


class EyeTrackingServer:
    """API server for eye tracking and face detection."""
    
    def __init__(self, port: int = 5001):
        """Initialize the eye tracking server."""
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for cross-origin requests
        
        self.port = port
        self.eye_tracker = SimplifiedEyeTracker()
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_active = False
        self.tracking_active = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Frame reading thread
        self.frame_thread = None
        self.frame_thread_active = False
        
        self.setup_routes()
    
    def setup_routes(self):
        """Set up API endpoints."""
        
        @self.app.route('/api/status', methods=['GET'])
        def status():
            """Get server status."""
            return jsonify({
                'status': 'running',
                'camera_active': self.camera_active,
                'tracking_active': self.tracking_active
            })
        
        @self.app.route('/api/start_camera', methods=['POST'])
        def start_camera():
            """Initialize and open the webcam."""
            try:
                if self.camera_active:
                    return jsonify({'success': True, 'message': 'Camera already active'})
                
                # Try to open camera
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    return jsonify({
                        'success': False,
                        'message': 'Failed to open camera'
                    }), 500
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.camera_active = True
                
                # Start frame reading thread
                self.frame_thread_active = True
                self.frame_thread = threading.Thread(target=self._read_frames)
                self.frame_thread.daemon = True
                self.frame_thread.start()
                
                return jsonify({
                    'success': True,
                    'message': 'Camera started successfully'
                })
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/start_tracking', methods=['POST'])
        def start_tracking():
            """Enable video frame processing."""
            try:
                if not self.camera_active:
                    # Try to start camera first
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        return jsonify({
                            'success': False,
                            'message': 'Camera not available'
                        }), 500
                    
                    self.camera_active = True
                    self.frame_thread_active = True
                    self.frame_thread = threading.Thread(target=self._read_frames)
                    self.frame_thread.daemon = True
                    self.frame_thread.start()
                
                self.tracking_active = True
                
                return jsonify({
                    'success': True,
                    'message': 'Tracking started'
                })
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/stop_camera', methods=['POST'])
        def stop_camera():
            """Release the webcam."""
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
                
                return jsonify({
                    'success': True,
                    'message': 'Camera stopped'
                })
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/get_enhanced_gaze', methods=['GET'])
        def get_enhanced_gaze():
            """
            Main polling endpoint for face detection.
            Returns simple JSON with face_detected boolean.
            """
            try:
                if not self.tracking_active or not self.camera_active:
                    return jsonify({
                        'success': False,
                        'face_detected': False,
                        'message': 'Tracking not active'
                    })
                
                with self.frame_lock:
                    if self.current_frame is None:
                        return jsonify({
                            'success': False,
                            'face_detected': False,
                            'message': 'No frame available'
                        })
                    
                    frame = self.current_frame.copy()
                
                # Process frame for face detection
                result = self.eye_tracker.process_frame(frame)
                
                return jsonify({
                    'success': result.get('success', False),
                    'face_detected': result.get('face_detected', False)
                })
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'face_detected': False,
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/video_feed', methods=['GET'])
        def video_feed():
            """Stream live camera feed."""
            def generate_frames():
                while self.camera_active:
                    with self.frame_lock:
                        if self.current_frame is None:
                            continue
                        frame = self.current_frame.copy()
                    
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                    
                    frame_bytes = buffer.tobytes()
                    
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    time.sleep(0.033)  # ~30 FPS
            
            return Response(
                generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
    
    def _read_frames(self):
        """Background thread to continuously read frames from camera."""
        while self.frame_thread_active and self.camera_active:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Mirror the frame for natural viewing
                    frame = cv2.flip(frame, 1)
                    
                    with self.frame_lock:
                        self.current_frame = frame
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def run(self):
        """Start the Flask server."""
        print(f"Eye Tracking Server starting on port {self.port}...")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


if __name__ == '__main__':
    server = EyeTrackingServer(port=5001)
    server.run()
