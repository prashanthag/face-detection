"""Base Detector Interface"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Face:
    """Detected face."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    landmarks: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class BaseDetector(ABC):
    """Abstract base class for face detectors."""

    def __init__(self, confidence: float = 0.5, max_faces: int = 10):
        self.confidence = confidence
        self.max_faces = max_faces
        self.name = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Face]:
        """Detect faces in frame. Returns list of Face objects."""
        pass

    def close(self):
        """Clean up resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
