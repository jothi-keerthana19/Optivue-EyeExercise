
"""
Eye Tracking API Server
Background service dedicated to computer vision processing.
Manages webcam feed and provides face detection API with visual feedback.
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from simplified_eye_tracker import SimplifiedEyeTracker
import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict, Any


class EyeTrackingServer:
    """API server for eye tracking and face detection."""

    def __init__(self, port: int = 5001):
        """Initialize the eye tracking server."""
        self.app = Flask(__name__)
        CORS(self.app, resources={r"/api/*": {"origins": "*"}})

        self.port = port
        # Use stricter confidence threshold of 0.7 and full-range model (1) for better accuracy
        self.eye_tracker = SimplifiedEyeTracker(model_selection=1, min_detection_confidence=0.7)
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_active = False
        self.tracking_active = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.last_detection_result: Optional[Dict[str, Any]] = None

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

                self.cap = cv2.VideoCapture(0)

                if not self.cap.isOpened():
                    return jsonify({
                        'success': False,
                        'message': 'No camera detected. Please connect a webcam and try again.'
                    }), 500

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                self.camera_active = True

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
                if not self.tracking_active or self.last_detection_result is None:
                    return jsonify({
                        'success': True,
                        'face_detected': False
                    })

                return jsonify(self.last_detection_result)

            except Exception as e:
                return jsonify({
                    'success': False,
                    'face_detected': False,
                    'message': str(e)
                }), 500

        @self.app.route('/api/detect_face', methods=['POST'])
        def detect_face():
            """
            Receive frame from browser and detect face using MediaPipe.
            """
            try:
                if 'frame' not in request.files:
                    return jsonify({'success': False, 'face_detected': False, 'message': 'No frame provided'}), 400

                file = request.files['frame']
                file_bytes = file.read()

                if len(file_bytes) == 0:
                    return jsonify({'success': False, 'face_detected': False, 'message': 'Empty frame'}), 400

                npimg = np.frombuffer(file_bytes, np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

                if frame is None:
                    return jsonify({'success': False, 'face_detected': False, 'message': 'Invalid frame'}), 400

                result = self.eye_tracker.process_frame(frame)

                return jsonify({
                    'success': True,
                    'face_detected': result.get('face_detected', False)
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'face_detected': False,
                    'message': str(e)
                }), 500

        @self.app.route('/api/video_feed', methods=['GET'])
        def video_feed():
            """Stream live camera feed with debug info."""
            def generate_frames():
                while self.camera_active:
                    with self.frame_lock:
                        if self.current_frame is None:
                            continue
                        frame = self.current_frame.copy()

                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue

                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                    time.sleep(0.033)

            return Response(
                generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

    def _read_frames(self):
        """
        Background thread that continuously reads frames from the camera,
        processes them using the eye tracker, and updates the shared state.
        """
        print("Starting frame reading and processing thread.")
        while self.frame_thread_active and self.camera_active:
            if not (self.cap and self.cap.isOpened()):
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
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
            
            time.sleep(0.033)  # Maintain a steady ~30 FPS
        print("Frame reading thread has stopped.")

    def run(self):
        """Start the Flask server."""
        print(f"Eye Tracking Server starting on port {self.port}...")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


if __name__ == '__main__':
    server = EyeTrackingServer(port=5001)
    server.run()
