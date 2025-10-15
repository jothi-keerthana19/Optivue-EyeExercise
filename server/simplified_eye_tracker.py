
"""
Enhanced Eye Tracker - High Accuracy Face Detection
Uses MediaPipe's dedicated FaceDetection model for robust face detection.
Provides visual feedback with bounding boxes, keypoints, and confidence scores.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any

try:
    import mediapipe as mp
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    mp = None
    FACE_DETECTION_AVAILABLE = False


class SimplifiedEyeTracker:
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
        self.face_detection = None
        self.mp_face_detection = None
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Face Detection."""
        if not FACE_DETECTION_AVAILABLE:
            print("MediaPipe not available")
            return
        
        try:
            self.mp_face_detection = mp.solutions.face_detection
            
            # Initialize the FaceDetection model with the specified confidence
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_detection_confidence
            )
            print(f"MediaPipe FaceDetection initialized with confidence={self.min_detection_confidence}, model={self.model_selection}")
        except Exception as e:
            self.face_detection = None
            print(f"Failed to initialize MediaPipe Face Detection: {e}")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame to detect face presence.
        
        Args:
            frame: Input video frame (BGR format from OpenCV)
        
        Returns:
            Dictionary with detection results and metadata
        """
        if self.face_detection is None:
            return {
                'face_detected': False,
                'success': False,
                'error': 'Face detection not initialized'
            }
        
        if frame is None:
            return {
                'face_detected': False,
                'success': False,
                'error': 'Input frame is None'
            }
        
        try:
            # Convert the BGR image to RGB as MediaPipe expects this format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to find faces
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                return {
                    'face_detected': True,
                    'success': True,
                    'detections': results.detections
                }
            else:
                return {
                    'face_detected': False,
                    'success': True
                }
        
        except Exception as e:
            return {
                'face_detected': False,
                'success': False,
                'error': str(e)
            }

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
        annotated_frame = frame.copy()
        frame_height, frame_width, _ = annotated_frame.shape

        # Convert the BGR image to RGB as MediaPipe expects this format
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find faces
        results = self.face_detection.process(rgb_frame)
        
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
                
                # Draw a white rectangle around the face
                cv2.rectangle(annotated_frame, 
                            (face_rect[0], face_rect[1]), 
                            (face_rect[0] + face_rect[2], face_rect[1] + face_rect[3]), 
                            color=(255, 255, 255), thickness=2)

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
    
    def cleanup(self) -> None:
        """Release resources."""
        if self.face_detection:
            self.face_detection.close()
