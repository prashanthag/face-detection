"""RetinaFace Detector (ResNet50 backbone)"""

import cv2
import numpy as np
from typing import List
from .base import BaseDetector, Face
from .gpu_utils import get_gpu_info

try:
    from retinaface import RetinaFace as RF
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False


class RetinaFaceDetector(BaseDetector):
    """RetinaFace detector with ResNet50 backbone - high accuracy."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10, **kwargs):
        """
        Args:
            confidence: Min detection confidence
            max_faces: Max faces to detect
        """
        super().__init__(confidence, max_faces)
        self.name = "retinaface"

        if not RETINAFACE_AVAILABLE:
            raise ImportError("RetinaFace not installed. Run: pip install retinaface")

        # RetinaFace uses TensorFlow/Keras, check GPU
        gpu_info = get_gpu_info()
        self.device = gpu_info['type']

        print(f"[RetinaFace] Loaded detector (conf={confidence}, backend={self.device})")

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]

        # RetinaFace expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        try:
            results = RF.detect_faces(rgb, threshold=self.confidence)
        except Exception as e:
            return []

        faces = []
        if results:
            # Sort by confidence
            sorted_faces = sorted(results.items(),
                                  key=lambda x: x[1].get('score', 0),
                                  reverse=True)

            for face_id, face_data in sorted_faces[:self.max_faces]:
                score = face_data.get('score', 0)
                if score < self.confidence:
                    continue

                box = face_data['facial_area']
                x1, y1, x2, y2 = box

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Get landmarks (5 points)
                landmarks = None
                if 'landmarks' in face_data:
                    lm = face_data['landmarks']
                    landmarks = np.array([
                        lm['right_eye'],
                        lm['left_eye'],
                        lm['nose'],
                        lm['mouth_right'],
                        lm['mouth_left']
                    ])

                if x2 > x1 and y2 > y1:
                    faces.append(Face(x1, y1, x2, y2, float(score), landmarks))

        return faces
