"""GPU Detection Utilities"""

import os
import sys


def _get_onnx_providers():
    """Get actually available ONNX Runtime providers."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        # Prefer GPU providers in order
        preferred = ['CUDAExecutionProvider', 'TensorrtExecutionProvider',
                     'ROCMExecutionProvider', 'DmlExecutionProvider',
                     'CoreMLExecutionProvider', 'OpenVINOExecutionProvider']
        providers = [p for p in preferred if p in available]
        providers.append('CPUExecutionProvider')
        return providers
    except ImportError:
        return ['CPUExecutionProvider']


def get_available_gpu():
    """
    Detect available GPU and return info.

    Returns:
        dict with keys: type, device, onnx_providers
    """
    # Get actual ONNX providers
    onnx_providers = _get_onnx_providers()

    gpu_info = {
        'type': 'cpu',
        'device': 'cpu',
        'onnx_providers': onnx_providers,
        'opencv_backend': None,
        'opencv_target': None,
    }

    # Check CUDA (NVIDIA) via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['type'] = 'cuda'
            gpu_info['device'] = 'cuda'
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
            gpu_info['gpu_name'] = 'AMD ROCm'
            return gpu_info
    except ImportError:
        pass

    # Check ONNX providers for GPU
    if 'CUDAExecutionProvider' in onnx_providers:
        gpu_info['type'] = 'cuda'
        gpu_info['gpu_name'] = 'CUDA (ONNX)'
    elif 'DmlExecutionProvider' in onnx_providers:
        gpu_info['type'] = 'directml'
        gpu_info['gpu_name'] = 'DirectML'
    elif 'CoreMLExecutionProvider' in onnx_providers:
        gpu_info['type'] = 'coreml'
        gpu_info['gpu_name'] = 'CoreML'
    elif 'OpenVINOExecutionProvider' in onnx_providers:
        gpu_info['type'] = 'openvino'
        gpu_info['gpu_name'] = 'OpenVINO'

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
        # Show ONNX provider
        onnx_gpu = info['onnx_providers'][0]
        if onnx_gpu != 'CPUExecutionProvider':
            print(f"[ONNX] Using: {onnx_gpu}")
        else:
            print("[ONNX] GPU not available, using CPU")
