"""
Simplified Eye Tracker - Face Detection Only
This module contains only basic face presence detection using MediaPipe Face Mesh.
All head pose estimation and drowsiness detection logic has been removed.
"""

import cv2
import numpy as np
from typing import Dict, Optional

try:
    import mediapipe as mp
    FACE_MESH_AVAILABLE = True
except ImportError:
    mp = None
    FACE_MESH_AVAILABLE = False


class SimplifiedEyeTracker:
    """
    Simplified eye tracker that only detects face presence.
    No head pose estimation or drowsiness detection.
    """
    
    def __init__(self) -> None:
        """Initialize the face mesh detector with basic settings."""
        self.face_mesh = None
        self._init_mediapipe()
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Face Mesh with minimal configuration."""
        if not FACE_MESH_AVAILABLE:
            print("MediaPipe not available")
            return
        
        try:
            # Initialize with optimized settings for face detection
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.3,  # Lower threshold for better detection
                min_tracking_confidence=0.3
            )
        except Exception as e:
            self.face_mesh = None
            print(f"Failed to initialize MediaPipe FaceMesh: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Process a single video frame to detect face presence.
        
        Args:
            frame: Input video frame (BGR format from OpenCV)
        
        Returns:
            Dictionary with:
                - face_detected (bool): True if face landmarks found, False otherwise
                - success (bool): True if processing succeeded
        """
        if self.face_mesh is None:
            return {
                'face_detected': False,
                'success': False,
                'error': 'Face mesh not initialized'
            }
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(rgb_frame)
            
            # Check if any face landmarks were detected
            face_detected = False
            if results.multi_face_landmarks:
                # Face landmarks found - face is present
                face_detected = True
            
            return {
                'face_detected': face_detected,
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
        if self.face_mesh:
            self.face_mesh.close()
