from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from enhanced_eye_tracker import EnhancedEyeTracker
import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict, Any

class EnhancedEyeTrackingServer:
    def __init__(self):
        self.app = Flask(__name__)
        # Configure CORS to allow requests from any origin to your API endpoints
        CORS(self.app, resources={r"/api/*": {"origins": "*"}})
        
        # Initialize with the new high-accuracy tracker
        self.eye_tracker = EnhancedEyeTracker(min_detection_confidence=0.7)
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_active = False
        self.tracking_active = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        self.last_detection_result: Optional[Dict[str, Any]] = None
        
        self.frame_thread: Optional[threading.Thread] = None
        self.frame_thread_active = False

        self.setup_routes()
    
    def _read_frames(self):
        """
        A dedicated background thread that continuously reads frames from the camera,
        processes them using the eye tracker, and updates the shared state.
        This is highly efficient as processing happens only once per frame.
        """
        print("Starting frame reading and processing thread.")
        failed_reads = 0
        max_failed_reads = 10
        
        while self.frame_thread_active and self.camera_active:
            if not (self.cap and self.cap.isOpened()):
                print("Camera not opened, waiting...")
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                failed_reads += 1
                print(f"Failed to capture frame ({failed_reads}/{max_failed_reads})")
                
                if failed_reads >= max_failed_reads:
                    print("Too many failed reads, stopping camera...")
                    self.camera_active = False
                    break
                    
                time.sleep(0.1)
                continue
            
            # Reset failed reads counter on successful read
            failed_reads = 0
            
            # Flip for a natural, mirror-like view
            frame = cv2.flip(frame, 1)
            
            annotated_frame = frame
            
            if self.tracking_active:
                # This single, efficient call gets both data and the visualized frame
                result, annotated_frame = self.eye_tracker.process_and_draw_frame(frame)
                self.last_detection_result = result
            else:
                # Clear old data if tracking is turned off
                self.last_detection_result = None
            
            # Safely update the frame that will be streamed to the frontend
            with self.frame_lock:
                self.current_frame = annotated_frame
            
            time.sleep(0.033) # Maintain a steady ~30 FPS
        print("Frame reading thread has stopped.")

    def setup_routes(self):
        """Defines all the API endpoints for the frontend to interact with."""

        @self.app.route('/api/enhanced-eye-tracking/status', methods=['GET'])
        def status():
            """Provides the current status of the server."""
            return jsonify({
                'status': 'running',
                'camera_status': 'active' if self.camera_active else 'inactive',
                'tracking_active': self.tracking_active
            })

        @self.app.route('/api/enhanced-eye-tracking/start_camera', methods=['POST'])
        def start_camera():
            """Starts the camera and the background frame-reading thread."""
            if self.camera_active:
                return jsonify({'success': True, 'message': 'Camera is already active'})
            
            # Try multiple camera indices for better compatibility
            camera_indices = [0, 1, 2]
            for idx in camera_indices:
                print(f"Trying camera index {idx}...")
                self.cap = cv2.VideoCapture(idx)
                if self.cap and self.cap.isOpened():
                    # Verify we can actually read a frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"Camera {idx} opened successfully")
                        self.camera_active = True
                        self.frame_thread_active = True
                        self.frame_thread = threading.Thread(target=self._read_frames)
                        self.frame_thread.daemon = True
                        self.frame_thread.start()
                        return jsonify({'success': True, 'message': f'Camera started successfully on index {idx}'})
                    else:
                        self.cap.release()
                        print(f"Camera {idx} opened but couldn't read frames")
                else:
                    if self.cap:
                        self.cap.release()
                    print(f"Failed to open camera {idx}")
            
            # No camera worked
            print("ERROR: No camera available. This may be a Replit environment without camera access.")
            return jsonify({
                'success': False, 
                'message': 'No camera detected. Camera hardware may not be available in this environment.'
            }), 500

        @self.app.route('/api/enhanced-eye-tracking/stop_camera', methods=['POST'])
        def stop_camera():
            """Stops the camera and cleans up resources."""
            if self.camera_active:
                self.frame_thread_active = False # Signal the thread to stop
                if self.frame_thread: self.frame_thread.join(timeout=1)
                if self.cap: self.cap.release()
                
                self.camera_active = False
                self.tracking_active = False
                self.cap = None
            return jsonify({'success': True, 'message': 'Camera stopped successfully'})

        @self.app.route('/api/enhanced-eye-tracking/start_tracking', methods=['POST'])
        def start_tracking():
            """Enables face detection processing on the video stream."""
            if not self.camera_active:
                return jsonify({'success': False, 'message': 'Camera is not active'}), 400
            self.tracking_active = True
            return jsonify({'success': True, 'message': 'Face detection tracking started'})

        @self.app.route('/api/enhanced-eye-tracking/stop_tracking', methods=['POST'])
        def stop_tracking():
            """Disables face detection processing."""
            self.tracking_active = False
            return jsonify({'success': True, 'message': 'Face detection tracking stopped'})

        @self.app.route('/api/enhanced-eye-tracking/get_enhanced_gaze', methods=['GET'])
        def get_enhanced_gaze():
            """Returns the latest face detection result."""
            if not self.tracking_active or self.last_detection_result is None:
                # Return a default "not detected" status
                return jsonify({'success': True, 'face_detected': False})
            return jsonify(self.last_detection_result)

        @self.app.route('/api/enhanced-eye-tracking/video_feed')
        def video_feed():
            """Streams the visually annotated video feed to the frontend."""
            def generate_frames():
                while self.camera_active:
                    frame_to_send = None
                    with self.frame_lock:
                        if self.current_frame is not None:
                            frame_to_send = self.current_frame
                    
                    if frame_to_send is not None:
                        ret, buffer = cv2.imencode('.jpg', frame_to_send)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.033) # Stream at ~30 FPS
            
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def run(self, host='0.0.0.0', port=5002, debug=False):
        """Starts the Flask web server."""
        print(f"Starting Enhanced Eye Tracking Server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)

if __name__ == '__main__':
    server = EnhancedEyeTrackingServer()
    server.run()
