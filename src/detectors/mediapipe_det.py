"""MediaPipe Face Detector"""

import cv2
import numpy as np
from typing import List
from .base import BaseDetector, Face

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class MediaPipeDetector(BaseDetector):
    """MediaPipe face detector - good balance of speed and accuracy."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10, model: int = 0, **kwargs):
        """
        Args:
            confidence: Min detection confidence
            max_faces: Max faces to detect
            model: 0=short range (2m), 1=full range (5m)
        """
        super().__init__(confidence, max_faces)
        self.name = "mediapipe"

        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed. Run: pip install mediapipe")

        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=model,
            min_detection_confidence=confidence
        )
        print(f"[MediaPipe] Loaded face detector (conf={confidence}, model={model})")

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        faces = []
        if results.detections:
            for det in results.detections[:self.max_faces]:
                bbox = det.location_data.relative_bounding_box
                conf = det.score[0]

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    faces.append(Face(x1, y1, x2, y2, conf))

        return faces

    def close(self):
        self.detector.close()
