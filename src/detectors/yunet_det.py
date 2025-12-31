"""YuNet Face Detector"""

import cv2
import numpy as np
import os
from typing import List
from .base import BaseDetector, Face


class YuNetDetector(BaseDetector):
    """YuNet detector - fast and lightweight, built into OpenCV."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10,
                 input_size: tuple = (320, 320), **kwargs):
        """
        Args:
            confidence: Min detection confidence
            max_faces: Max faces to detect
            input_size: Model input size (width, height)
        """
        super().__init__(confidence, max_faces)
        self.name = "yunet"
        self.input_size = input_size

        # Check OpenCV version
        cv_version = tuple(map(int, cv2.__version__.split('.')[:2]))
        if cv_version < (4, 5):
            raise RuntimeError(f"YuNet requires OpenCV 4.5.4+, got {cv2.__version__}")

        # Find model
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        model_path = os.path.join(model_dir, 'face_detection_yunet_2023mar.onnx')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}\n"
                "Download: wget -P models/ https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )

        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            score_threshold=confidence,
            nms_threshold=0.3,
            top_k=max_faces
        )

        print(f"[YuNet] Loaded detector (conf={confidence}, size={input_size})")

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]

        # Resize input if needed
        self.detector.setInputSize((w, h))

        # Detect faces
        _, results = self.detector.detect(frame)

        faces = []
        if results is not None:
            for det in results[:self.max_faces]:
                x1, y1, bw, bh = det[:4].astype(int)
                conf = float(det[14])  # confidence is at index 14

                x2, y2 = x1 + bw, y1 + bh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Landmarks (5 points: right eye, left eye, nose, right mouth, left mouth)
                landmarks = det[4:14].reshape(5, 2) if len(det) > 4 else None

                # Filter small boxes (likely false positives)
                if bw > 40 and bh > 40:
                    faces.append(Face(x1, y1, x2, y2, conf, landmarks))

        return faces
