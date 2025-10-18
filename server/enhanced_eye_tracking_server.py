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
        CORS(self.app, resources={r"/api/*": {"origins": "*"}})

        # Use higher confidence threshold for better accuracy
        self.eye_tracker = EnhancedEyeTracker(
            model_selection=1,  # Full-range model for better accuracy
            min_detection_confidence=0.75  # Higher threshold for more reliable detections
        )

        self.last_processed_frame: Optional[np.ndarray] = None
        self.last_detection_result: Optional[Dict[str, Any]] = None
        self.frame_lock = threading.Lock()

        self.setup_routes()

    def setup_routes(self):
        """Defines API endpoints for browser-based camera processing."""

        @self.app.route('/api/enhanced-eye-tracking/status', methods=['GET'])
        def status():
            return jsonify({'status': 'running', 'message': 'Ready to process frames from browser.'})

        @self.app.route('/api/enhanced-eye-tracking/process_frame', methods=['POST'])
        def process_frame():
            """
            Receives a frame from the browser, processes it for face detection,
            and returns the detection results.
            """
            if 'frame' not in request.files:
                print("ERROR: No 'frame' in request.files")
                return jsonify({'error': "No 'frame' file in request."}), 400

            file = request.files['frame']
            
            # Read file content
            file_content = file.read()
            if len(file_content) == 0:
                print("ERROR: Empty file received")
                return jsonify({'error': 'Empty frame data received'}), 400

            try:
                # Decode image from buffer
                np_img = np.frombuffer(file_content, np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                if frame is None or frame.size == 0:
                    print(f"ERROR: Failed to decode image. Buffer size: {len(file_content)}, np_img shape: {np_img.shape}")
                    return jsonify({'error': 'Failed to decode image.'}), 400
                
                # Validate frame dimensions
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"ERROR: Invalid decoded frame shape: {frame.shape}")
                    return jsonify({'error': f'Invalid frame shape: {frame.shape}'}), 400
                
                print(f"Successfully decoded frame: shape={frame.shape}, dtype={frame.dtype}, contiguous={frame.flags['C_CONTIGUOUS']}")
                
                # Ensure frame is writable and contiguous
                if not frame.flags['WRITEABLE']:
                    frame = frame.copy()
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                
                print(f"Frame prepared for processing: writable={frame.flags['WRITEABLE']}, contiguous={frame.flags['C_CONTIGUOUS']}")

                # Process the frame for face detection
                result, annotated_frame = self.eye_tracker.process_and_draw_frame(frame)

                # Store the latest annotated frame and result
                with self.frame_lock:
                    self.last_processed_frame = annotated_frame
                    self.last_detection_result = result

                return jsonify(result)

            except Exception as e:
                print(f"ERROR processing frame: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        @self.app.route('/api/enhanced-eye-tracking/get_enhanced_gaze', methods=['GET'])
        def get_enhanced_gaze():
            """Returns the latest face detection result."""
            with self.frame_lock:
                if self.last_detection_result is None:
                    return jsonify({'success': True, 'face_detected': False})
                return jsonify(self.last_detection_result)

        @self.app.route('/api/enhanced-eye-tracking/video_feed')
        def video_feed():
            """
            Streams the processed frames with bounding boxes drawn on them.
            """
            def generate_frames():
                while True:
                    frame_to_send = None
                    with self.frame_lock:
                        if self.last_processed_frame is not None:
                            frame_to_send = self.last_processed_frame.copy()

                    if frame_to_send is not None:
                        ret, buffer = cv2.imencode('.jpg', frame_to_send)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                    time.sleep(0.033)  # ~30 FPS

            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self, host='0.0.0.0', port=5002, debug=False):
        print(f"Starting Enhanced Eye Tracking Server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)

if __name__ == '__main__':
    server = EnhancedEyeTrackingServer()
    server.run()