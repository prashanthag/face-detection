"""GPU Detection Utilities"""

import os
import sys


def get_available_gpu():
    """
    Detect available GPU and return info.

    Returns:
        dict with keys: type, device, onnx_providers
    """
    gpu_info = {
        'type': 'cpu',
        'device': 'cpu',
        'onnx_providers': ['CPUExecutionProvider'],
        'opencv_backend': None,
        'opencv_target': None,
    }

    # Check CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['type'] = 'cuda'
            gpu_info['device'] = 'cuda'
            gpu_info['onnx_providers'] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            gpu_info['opencv_backend'] = 'CUDA'
            gpu_info['opencv_target'] = 'CUDA'
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            return gpu_info
    except ImportError:
        pass

    # Check Apple Metal (MPS)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info['type'] = 'mps'
            gpu_info['device'] = 'mps'
            gpu_info['onnx_providers'] = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            gpu_info['gpu_name'] = 'Apple Metal'
            return gpu_info
    except ImportError:
        pass

    # Check ROCm (AMD Linux)
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            gpu_info['type'] = 'rocm'
            gpu_info['device'] = 'cuda'  # ROCm uses cuda device in PyTorch
            gpu_info['onnx_providers'] = ['ROCMExecutionProvider', 'CPUExecutionProvider']
            gpu_info['gpu_name'] = 'AMD ROCm'
            return gpu_info
    except ImportError:
        pass

    # Check DirectML (Windows - AMD/Intel/NVIDIA)
    if sys.platform == 'win32':
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                gpu_info['type'] = 'directml'
                gpu_info['device'] = 'cpu'  # PyTorch still uses CPU
                gpu_info['onnx_providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
                gpu_info['gpu_name'] = 'DirectML GPU'
                return gpu_info
        except ImportError:
            pass

    # Check OpenVINO (Intel)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'OpenVINOExecutionProvider' in providers:
            gpu_info['type'] = 'openvino'
            gpu_info['device'] = 'cpu'
            gpu_info['onnx_providers'] = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
            gpu_info['gpu_name'] = 'Intel OpenVINO'
            return gpu_info
    except ImportError:
        pass

    # Fallback: check ONNX Runtime available providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            gpu_info['type'] = 'cuda'
            gpu_info['device'] = 'cuda'
            gpu_info['onnx_providers'] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            gpu_info['gpu_name'] = 'CUDA (via ONNX)'
            return gpu_info
    except ImportError:
        pass

    return gpu_info


def get_opencv_gpu_backend():
    """Get OpenCV DNN backend and target for GPU."""
    import cv2

    # Try CUDA
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    except:
        pass

    # Try OpenVINO
    try:
        # Check if OpenVINO backend is available
        return cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv2.dnn.DNN_TARGET_CPU
    except:
        pass

    # Fallback to default
    return cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU


# Cache GPU info
_gpu_info = None

def get_gpu_info():
    """Get cached GPU info."""
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = get_available_gpu()
    return _gpu_info


def print_gpu_info():
    """Print detected GPU information."""
    info = get_gpu_info()
    if info['type'] == 'cpu':
        print("[GPU] No GPU detected, using CPU")
    else:
        name = info.get('gpu_name', info['type'].upper())
        print(f"[GPU] Detected: {name} ({info['type']})")
