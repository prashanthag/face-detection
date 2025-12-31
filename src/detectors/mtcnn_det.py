"""MTCNN Face Detector"""

import cv2
import numpy as np
from typing import List
from .base import BaseDetector, Face
from .gpu_utils import get_gpu_info

try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class MTCNNDetector(BaseDetector):
    """MTCNN detector - better for small faces and varied angles."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10,
                 min_face_size: int = 20, device: str = None, **kwargs):
        """
        Args:
            confidence: Min detection confidence
            max_faces: Max faces to detect
            min_face_size: Minimum face size in pixels
            device: 'cuda', 'mps', or 'cpu' (auto-detect if None)
        """
        super().__init__(confidence, max_faces)
        self.name = "mtcnn"

        if not MTCNN_AVAILABLE:
            raise ImportError("MTCNN not installed. Run: pip install facenet-pytorch")

        # Auto-detect GPU
        if device is None:
            gpu_info = get_gpu_info()
            device = gpu_info['device']
            # MTCNN supports cuda and mps
            if device not in ('cuda', 'mps'):
                device = 'cpu'

        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # Detection thresholds for 3 stages
            post_process=False
        )
        print(f"[MTCNN] Loaded detector (conf={confidence}, device={device}, min_size={min_face_size})")

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]

        # MTCNN expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)

        faces = []
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None or prob < self.confidence:
                    continue
                if len(faces) >= self.max_faces:
                    break

                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                lm = landmarks[i] if landmarks is not None else None

                if x2 > x1 and y2 > y1:
                    faces.append(Face(x1, y1, x2, y2, float(prob), lm))

        return faces
