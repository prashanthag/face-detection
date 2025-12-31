"""OpenCV DNN Face Detector"""

import cv2
import numpy as np
import os
from typing import List
from .base import BaseDetector, Face


class OpenCVDetector(BaseDetector):
    """OpenCV DNN SSD face detector - fast and reliable."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10, **kwargs):
        super().__init__(confidence, max_faces)
        self.name = "opencv"

        # Load model
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        prototxt = os.path.join(model_dir, 'deploy.prototxt')
        caffemodel = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

        if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
            raise FileNotFoundError(f"OpenCV model not found in {model_dir}")

        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        print(f"[OpenCV] Loaded DNN face detector (conf={confidence})")

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > self.confidence and len(faces) < self.max_faces:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 + 10 and y2 > y1 + 10:
                    faces.append(Face(x1, y1, x2, y2, conf))

        return faces
