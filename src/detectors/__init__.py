"""Face Detection Methods"""

from .base import BaseDetector
from .opencv_dnn import OpenCVDetector
from .mediapipe_det import MediaPipeDetector
from .mtcnn_det import MTCNNDetector
from .yunet_det import YuNetDetector
from .gpu_utils import get_gpu_info, print_gpu_info

# Working detectors only
DETECTORS = {
    'opencv': OpenCVDetector,
    'mediapipe': MediaPipeDetector,
    'mtcnn': MTCNNDetector,
    'yunet': YuNetDetector,
}

def get_detector(method: str, **kwargs):
    """Get detector by method name."""
    method = method.lower()
    if method not in DETECTORS:
        available = ', '.join(DETECTORS.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")
    return DETECTORS[method](**kwargs)

def list_methods():
    """List available detection methods."""
    return list(DETECTORS.keys())
