
"""
Simplified Eye Tracker - Face Detection with Confidence Thresholds
This module uses MediaPipe Face Detection for robust face presence detection.
"""

import cv2
import numpy as np
from typing import Dict

try:
    import mediapipe as mp
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    mp = None
    FACE_DETECTION_AVAILABLE = False


class SimplifiedEyeTracker:
    """
    Simplified eye tracker that detects face presence with confidence thresholds.
    Uses MediaPipe Face Detection for robust detection.
    """
    
    def __init__(self, min_detection_confidence: float = 0.7) -> None:
        """
        Initialize the face detector with confidence threshold.
        
        Args:
            min_detection_confidence: Minimum confidence (0.0-1.0) for valid detection
                                     Higher values (0.7) make detection stricter and more accurate
        """
        self.face_detection = None
        self.min_detection_confidence = min_detection_confidence
        self._init_mediapipe()
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Face Detection."""
        if not FACE_DETECTION_AVAILABLE:
            print("MediaPipe not available")
            return
        
        try:
            # Initialize Face Detection with stricter confidence
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=self.min_detection_confidence,
                model_selection=0  # 0 for short range (within 2 meters)
            )
            print(f"MediaPipe Face Detection initialized with confidence threshold: {self.min_detection_confidence}")
        except Exception as e:
            self.face_detection = None
            print(f"Failed to initialize MediaPipe Face Detection: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Process a single video frame to detect face presence.
        
        Args:
            frame: Input video frame (BGR format from OpenCV)
        
        Returns:
            Dictionary with:
                - face_detected (bool): True if face detected above confidence threshold
                - success (bool): True if processing succeeded
                - detections: List of detections (for visual debugging)
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
            # Mark image as not writeable for performance
            frame.flags.setflags(write=False)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_detection.process(rgb_frame)
            
            # Restore writeable flag
            frame.flags.setflags(write=True)
            
            # CRITICAL CHECK: results.detections is None or empty if no face found
            # Only faces above min_detection_confidence threshold are included
            if results.detections:
                return {
                    'face_detected': True,
                    'success': True,
                    'detections': results.detections  # Include for visual debugging
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
    
    def cleanup(self) -> None:
        """Release resources."""
        if self.face_detection:
            self.face_detection.close()
