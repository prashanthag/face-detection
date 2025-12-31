# Face Blur

Multi-method face detection and masking for video streams with automatic GPU detection.

## GPU Support

Automatically detects and uses:
- NVIDIA CUDA
- Apple Metal (MPS)
- AMD ROCm
- Intel OpenVINO
- DirectML (Windows)

## Install

```bash
# Standard install
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# For Jetson/aarch64 with CUDA (use system PyTorch)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install mediapipe facenet-pytorch onnxruntime
pip uninstall torch torchvision -y  # Use system CUDA torch
```

## Models

Download models to `models/` folder:

```bash
# OpenCV DNN (required for opencv method)
wget -P models/ https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget -P models/ https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# SCRFD (required for scrfd method)
wget -P models/ https://huggingface.co/MonsterMMORPG/SCRFD/resolve/main/scrfd_10g_320_batch.onnx
```

## Usage

```bash
python src/face_blur.py [options]
```

| Option | Description |
|--------|-------------|
| `-m METHOD` | Detection: `opencv`, `mediapipe` (default), `mtcnn`, `scrfd` |
| `-k MASK` | Mask type: `black` (default), `blur`, `pixelate`, `color` |
| `-s SOURCE` | Input: webcam index (0), RTSP URL, or video file |
| `-c FLOAT` | Confidence threshold (default: 0.5) |
| `-n INT` | Max faces (default: 10) |
| `-e FLOAT` | Mask expansion factor (default: 1.3) |
| `-o FILE` | Save output to file |
| `--no-stats` | Hide FPS overlay |
| `--no-display` | Headless mode |
| `-l` | List available methods |

## Examples

```bash
python src/face_blur.py                          # Webcam, mediapipe, black mask
python src/face_blur.py -m opencv -k blur        # OpenCV with blur
python src/face_blur.py -s "rtsp://ip/stream"    # RTSP input
python src/face_blur.py -o out.mp4 --no-display  # Save without display
```

## Controls

- `q` - Quit
