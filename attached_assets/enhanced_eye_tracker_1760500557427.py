import cv2
import numpy as np
import time
from typing import Dict, Optional, List, Tuple
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
    FACE_MESH_AVAILABLE = True
except ImportError:
    mp = None
    FACE_MESH_AVAILABLE = False

class EnhancedEyeTracker:
    def __init__(self) -> None:
        """Initializes tracker with head pose and blink detection attributes."""
        self.face_mesh = None
        self._init_mediapipe()
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: np.ndarray = np.zeros((4, 1))
        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype=np.float64)

        # Attributes for blink detection
        self.ear_consecutive_frames = 3 # Frames the eye must be closed to count as a blink
        self.ear_counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        self.blink_rate = 0

    def _init_mediapipe(self) -> None:
        if not FACE_MESH_AVAILABLE: return
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        except Exception as e:
            self.face_mesh = None
            print(f"Failed to initialize MediaPipe FaceMesh: {e}")

    def _calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.3

    def process_frame(self, frame: np.ndarray, target_position: Optional[List[float]], canvas_size: List[float]) -> Dict[str, object]:
        if self.face_mesh is None:
            return {'face_detected': False, 'is_diverted': True, 'is_drowsy': False, 'success': False}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {'face_detected': False, 'is_diverted': True, 'is_drowsy': False, 'success': True}

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        if self.camera_matrix is None:
            self.camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)

        # --- Head Pose Estimation for Diversion Detection ---
        face_2d_points = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in [1, 199, 33, 263, 61, 291]], dtype=np.float64)
        success, rotation_vector, _ = cv2.solvePnP(self.face_3d_model_points, face_2d_points, self.camera_matrix, self.dist_coeffs)
        
        is_diverted = True
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            if sy >= 1e-6:
                head_yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
                if abs(head_yaw) < 25:
                    is_diverted = False

        # --- Blink & Drowsiness Detection ---
        LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        left_eye_pts = [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in LEFT_EYE_INDICES]
        right_eye_pts = [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in RIGHT_EYE_INDICES]
        avg_ear = (self._calculate_ear(left_eye_pts) + self._calculate_ear(right_eye_pts)) / 2.0
        
        EAR_THRESHOLD = 0.20
        is_drowsy = False
        
        if avg_ear < EAR_THRESHOLD:
            self.ear_counter += 1
            if self.ear_counter >= self.ear_consecutive_frames * 5: # If eyes closed for ~0.5s, consider drowsy
                is_drowsy = True
        else:
            if self.ear_counter >= self.ear_consecutive_frames:
                self.total_blinks += 1
                current_time = time.time()
                # Simple blink rate over the last minute (can be improved)
                if current_time - self.last_blink_time > 60:
                    self.blink_rate = self.total_blinks
                    self.total_blinks = 0
                    self.last_blink_time = current_time
            self.ear_counter = 0

        # Drowsiness can also be a very low blink rate over time
        if time.time() - self.last_blink_time > 60 and self.blink_rate < 5:
             is_drowsy = True

        return {'face_detected': True, 'is_diverted': is_diverted, 'is_drowsy': is_drowsy, 'blink_count': self.total_blinks, 'success': True}
