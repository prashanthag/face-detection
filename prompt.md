# Face Detection and Masking Application

Create a face detection and masking application with these requirements:

## Detection Backends

- OpenCV DNN (SSD face detector)
- MediaPipe face detection
- MTCNN (facenet-pytorch)
- SCRFD (ONNX model)

## Architecture

- Modular design with base detector class and common interface
- Each detector in separate file under `detectors/`
- Factory function to get detector by name

## Masking Options

- Black solid rectangle (fastest)
- Gaussian blur
- Pixelation
- Solid color

## Video Input Support

- USB webcam (device index)
- RTSP stream URL
- Video file

## CLI Arguments

- Detection method selection
- Mask type selection
- Confidence threshold
- Max faces limit
- Mask expansion factor
- Resolution settings
- Output file saving
- Headless mode
- Toggle FPS/stats overlay (on by default)

## Features

- Real-time FPS display (toggleable, on by default)
- Detection time overlay
- Face count display
- Auto-reconnect for streams
- Graceful cleanup on exit

## Target

Python with OpenCV, runs on Linux (including ARM/aarch64)
