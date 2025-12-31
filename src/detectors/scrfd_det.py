"""SCRFD Face Detector"""

import cv2
import numpy as np
import os
from typing import List
from .base import BaseDetector, Face

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class SCRFDDetector(BaseDetector):
    """SCRFD detector - efficient and accurate, various model sizes."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10,
                 model_path: str = None, input_size: tuple = (320, 320), **kwargs):
        """
        Args:
            confidence: Min detection confidence
            max_faces: Max faces to detect
            model_path: Path to ONNX model
            input_size: Model input size (width, height)
        """
        super().__init__(confidence, max_faces)
        self.name = "scrfd"

        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime")

        # Find model
        if model_path is None:
            model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
            model_path = os.path.join(model_dir, 'scrfd_10g_320_batch.onnx')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        self.input_size = input_size
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._center_cache = {}

        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"[SCRFD] Loaded {os.path.basename(model_path)} (conf={confidence})")

    def _preprocess(self, frame: np.ndarray):
        """Preprocess frame for SCRFD."""
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
        """Generate anchor centers."""
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

        faces = []
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

            # Distance to bbox
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

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes_all.tolist(), scores_all.tolist(),
                                    self.confidence, 0.4)
        if len(indices) == 0:
            return []

        for i in indices.flatten()[:self.max_faces]:
            box = boxes_all[i]
            conf = scores_all[i]

            # Scale back to original
            x1 = int((box[0] - pad_w) / scale)
            y1 = int((box[1] - pad_h) / scale)
            x2 = int((box[2] - pad_w) / scale)
            y2 = int((box[3] - pad_h) / scale)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                faces.append(Face(x1, y1, x2, y2, float(conf)))

        return faces
