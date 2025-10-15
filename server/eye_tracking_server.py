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
        CORS(self.app)

        self.port = port
        # Use stricter confidence threshold of 0.7 for better accuracy
        self.eye_tracker = SimplifiedEyeTracker(min_detection_confidence=0.7)
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_active = False
        self.tracking_active = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

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
        """Background thread to continuously read frames and draw debug info."""
        while self.frame_thread_active and self.camera_active:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)

                    # Process frame to get detection results
                    result = self.eye_tracker.process_frame(frame)

                    # Draw bounding box on the original frame if face detected
                    if result.get('face_detected') and result.get('detections'):
                        for detection in result['detections']:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            x = int(bboxC.xmin * iw)
                            y = int(bboxC.ymin * ih)
                            w = int(bboxC.width * iw)
                            h = int(bboxC.height * ih)

                            # Draw green box around detected face
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # Show confidence score
                            confidence = detection.score[0] if detection.score else 0
                            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # Indicate no face detected
                        cv2.putText(frame, "No Face Detected", (20, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    with self.frame_lock:
                        self.current_frame = frame

            time.sleep(0.01)

    def run(self):
        """Start the Flask server."""
        print(f"Eye Tracking Server starting on port {self.port}...")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


if __name__ == '__main__':
    server = EyeTrackingServer(port=5001)
    server.run()