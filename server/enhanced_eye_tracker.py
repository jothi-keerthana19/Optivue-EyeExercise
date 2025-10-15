import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Any

class EnhancedEyeTracker:
    """
    A robust eye tracker using MediaPipe's dedicated FaceDetection model for high accuracy.
    This class processes a video frame to detect a face and annotates the frame with
    visual feedback (bounding box, keypoints, and confidence score).
    """

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.7) -> None:
        """
        Initializes the tracker with the MediaPipe FaceDetection model.

        Args:
            model_selection (int): 0 for short-range model (2 meters), 1 for full-range (5 meters).
                                   1 is generally more versatile and accurate.
            min_detection_confidence (float): Minimum confidence value (from 0.0 to 1.0) for a
                                              detection to be considered successful. A higher value
                                              like 0.7 increases accuracy by filtering out weak detections.
        """
        self.mp_face_detection = mp.solutions.face_detection

        # Initialize the FaceDetection model with the specified confidence
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        print(f"MediaPipe FaceDetection initialized with confidence={min_detection_confidence}")

    def process_and_draw_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Detects faces in a frame and draws detailed visual feedback.

        Args:
            frame (np.ndarray): The input image frame from the camera (in BGR format).

        Returns:
            A tuple containing:
            - A dictionary with detection results ('face_detected', 'success').
            - The annotated frame with a bounding box, keypoints, and score.
        """
        # Validate frame
        if frame is None or frame.size == 0:
            print("ERROR: Invalid frame - frame is None or empty")
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty_frame, "Invalid Frame", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return {'face_detected': False, 'success': False, 'error': 'Invalid frame'}, empty_frame
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"ERROR: Invalid frame shape: {frame.shape}")
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty_frame, "Invalid Frame Shape", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return {'face_detected': False, 'success': False, 'error': 'Invalid frame shape'}, empty_frame
        
        annotated_frame = frame.copy()
        frame_height, frame_width, _ = annotated_frame.shape

        # Convert the BGR image to RGB as MediaPipe expects this format
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe requires specific image properties - create a clean copy
        # Ensure uint8 data type
        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)
        
        # Ensure the frame is writable and contiguous
        rgb_frame = np.ascontiguousarray(rgb_frame)
        rgb_frame.flags.writeable = True

        # Process the frame to find faces
        try:
            results = self.face_detection.process(rgb_frame)
        except ValueError as e:
            if "Empty packets" in str(e) or "Graph has errors" in str(e):
                print(f"MediaPipe processing error: {e}")
                print(f"Frame info - shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}, contiguous: {rgb_frame.flags['C_CONTIGUOUS']}, writable: {rgb_frame.flags['WRITEABLE']}")
                print(f"Frame data range: min={rgb_frame.min()}, max={rgb_frame.max()}, mean={rgb_frame.mean()}")
                # Return a frame with error message
                cv2.putText(annotated_frame, "MediaPipe Error - Check Console", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                return {'face_detected': False, 'success': False, 'error': str(e)}, annotated_frame
            raise

        face_detected = False

        if results.detections:
            face_detected = True
            # Loop through each detected face
            for detection in results.detections:
                # --- 1. Draw the Bounding Box ---
                bbox_data = detection.location_data.relative_bounding_box
                face_rect = np.multiply(
                    [bbox_data.xmin, bbox_data.ymin, bbox_data.width, bbox_data.height],
                    [frame_width, frame_height, frame_width, frame_height]
                ).astype(int)

                # Convert to top-left and bottom-right coordinates for cv2.rectangle
                top_left = (face_rect[0], face_rect[1])
                bottom_right = (face_rect[0] + face_rect[2], face_rect[1] + face_rect[3])

                # Draw a white rectangle around the face
                cv2.rectangle(annotated_frame, top_left, bottom_right, color=(255, 255, 255), thickness=2)

                # --- 2. Draw the Confidence Score ---
                confidence_score = detection.score[0]
                score_text = f"Confidence: {confidence_score:.2%}"
                cv2.putText(annotated_frame, score_text, (face_rect[0], face_rect[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # --- 3. Draw the 6 Key Facial Keypoints ---
                keypoints = detection.location_data.relative_keypoints
                for keypoint in keypoints:
                    keypoint_px = (int(keypoint.x * frame_width), int(keypoint.y * frame_height))
                    # Draw a small circle for each keypoint
                    cv2.circle(annotated_frame, keypoint_px, 4, (0, 255, 0), -1)
        else:
            # If no face is found, display a clear message
            cv2.putText(annotated_frame, "Face Not Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        data = {
            'face_detected': face_detected,
            'success': True
        }

        return data, annotated_frame

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and return face detection results.

        Args:
            frame: Input BGR image from camera

        Returns:
            Dictionary with detection results including bounding box
        """
        if self.face_detection is None:
            return {'face_detected': False}

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            return {
                'face_detected': True,
                'num_faces': len(results.detections),
                'bbox': {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                }
            }

        return {'face_detected': False}