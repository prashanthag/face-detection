"""RetinaFace Detector - ONNX based (no TensorFlow dependency)"""

import cv2
import numpy as np
import os
from typing import List
from .base import BaseDetector, Face
from .gpu_utils import get_gpu_info

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class RetinaFaceDetector(BaseDetector):
    """RetinaFace-style detector using SCRFD (more compatible, similar accuracy)."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10,
                 input_size: tuple = (640, 640), **kwargs):
        """
        Args:
            confidence: Min detection confidence
            max_faces: Max faces to detect
            input_size: Model input size
        """
        super().__init__(confidence, max_faces)
        self.name = "retinaface"

        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime")

        # Use SCRFD model as RetinaFace alternative (similar architecture, ONNX based)
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        model_path = os.path.join(model_dir, 'scrfd_10g_320_batch.onnx')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.input_size = (320, 320)  # Match SCRFD model
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._center_cache = {}

        # Get GPU providers
        gpu_info = get_gpu_info()
        providers = gpu_info['onnx_providers']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        active_provider = self.session.get_providers()[0]
        print(f"[RetinaFace] Loaded SCRFD backend (conf={confidence}, provider={active_provider})")

    def _preprocess(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        input_h, input_w = self.input_size[1], self.input_size[0]

        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        pad_w = (input_w - new_w) // 2
        pad_h = (input_h - new_h) // 2

        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        return blob, scale, (pad_w, pad_h)

    def _generate_anchors(self, height: int, width: int, stride: int):
        key = (height, width, stride)
        if key in self._center_cache:
            return self._center_cache[key]

        anchors = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchors = (anchors * stride).reshape(-1, 2)
        anchors = np.stack([anchors] * self._num_anchors, axis=1).reshape(-1, 2)
        anchors += stride // 2

        self._center_cache[key] = anchors
        return anchors

    def detect(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]

        blob, scale, (pad_w, pad_h) = self._preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        output_dict = {name: out for name, out in zip(self.output_names, outputs)}

        scores_all, boxes_all = [], []

        for stride in self._feat_stride_fpn:
            score_key = f'score_{stride}'
            bbox_key = f'bbox_{stride}'

            if score_key not in output_dict:
                continue

            scores = output_dict[score_key][0].reshape(-1)
            bbox_preds = output_dict[bbox_key][0].reshape(-1, 4)

            feat_h = self.input_size[1] // stride
            feat_w = self.input_size[0] // stride
            anchors = self._generate_anchors(feat_h, feat_w, stride)

            pos_inds = np.where(scores >= self.confidence)[0]
            if len(pos_inds) == 0:
                continue

            pos_scores = scores[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_bbox = bbox_preds[pos_inds] * stride

            x1 = pos_anchors[:, 0] - pos_bbox[:, 0]
            y1 = pos_anchors[:, 1] - pos_bbox[:, 1]
            x2 = pos_anchors[:, 0] + pos_bbox[:, 2]
            y2 = pos_anchors[:, 1] + pos_bbox[:, 3]

            boxes = np.stack([x1, y1, x2, y2], axis=1)

            scores_all.append(pos_scores)
            boxes_all.append(boxes)

        if not scores_all:
            return []

        scores_all = np.concatenate(scores_all)
        boxes_all = np.concatenate(boxes_all)

        indices = cv2.dnn.NMSBoxes(boxes_all.tolist(), scores_all.tolist(),
                                    self.confidence, 0.4)
        if len(indices) == 0:
            return []

        faces = []
        for i in indices.flatten()[:self.max_faces]:
            box = boxes_all[i]
            conf = scores_all[i]

            x1 = int((box[0] - pad_w) / scale)
            y1 = int((box[1] - pad_h) / scale)
            x2 = int((box[2] - pad_w) / scale)
            y2 = int((box[3] - pad_h) / scale)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Filter small boxes (likely false positives)
            box_w, box_h = x2 - x1, y2 - y1
            if box_w > 40 and box_h > 40:
                faces.append(Face(x1, y1, x2, y2, float(conf)))

        return faces
