#!/usr/bin/env python3
"""
Face Blur - Multi-method face detection and blurring

Supported methods:
  - opencv    : OpenCV DNN (fast, reliable)
  - mediapipe : MediaPipe (good balance)
  - mtcnn     : MTCNN (better for small faces)
  - scrfd     : SCRFD ONNX (efficient, accurate)
"""

import cv2
import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detectors import get_detector, list_methods, print_gpu_info


def mask_faces(frame, faces, expansion=1.3, method='black', blur_strength=51, color=(0,0,0)):
    """
    Apply mask to detected faces.

    Methods:
        black    - Solid black rectangle (fastest)
        color    - Solid color rectangle
        blur     - Gaussian blur
        pixelate - Pixelation effect
    """
    result = frame.copy()
    h, w = frame.shape[:2]

    for face in faces:
        # Expand box
        cx = (face.x1 + face.x2) // 2
        cy = (face.y1 + face.y2) // 2
        bw = int(face.width * expansion)
        bh = int(face.height * expansion)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)

        if x2 <= x1 or y2 <= y1:
            continue

        if method == 'black':
            # Fastest - solid black
            result[y1:y2, x1:x2] = 0
        elif method == 'color':
            # Solid color
            result[y1:y2, x1:x2] = color
        elif method == 'blur':
            # Gaussian blur (slower)
            ksize = blur_strength | 1
            result[y1:y2, x1:x2] = cv2.GaussianBlur(result[y1:y2, x1:x2], (ksize, ksize), 0)
        elif method == 'pixelate':
            # Pixelation
            roi = result[y1:y2, x1:x2]
            small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_LINEAR)
            result[y1:y2, x1:x2] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

    return result


def main():
    methods = list_methods()

    parser = argparse.ArgumentParser(
        description='Face Blur - Multiple Detection Methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Methods:
  opencv     - OpenCV DNN SSD (fast, good for most cases)
  mediapipe  - Google MediaPipe (balanced speed/accuracy)
  mtcnn      - MTCNN (better for small/angled faces, slower)
  scrfd      - SCRFD ONNX (efficient, high accuracy)

Examples:
  python face_blur.py                        # Default (mediapipe)
  python face_blur.py -m opencv              # Use OpenCV
  python face_blur.py -m mtcnn -c 0.3        # MTCNN with lower threshold
  python face_blur.py -s rtsp://ip/stream    # RTSP stream
  python face_blur.py -o output.mp4          # Save to file
'''
    )

    parser.add_argument('--method', '-m', choices=methods, default='mediapipe',
                        help=f'Detection method: {", ".join(methods)}')
    parser.add_argument('--source', '-s', default=0,
                        help='Video source (0=webcam, or RTSP URL, or file)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--max-faces', '-n', type=int, default=10,
                        help='Maximum faces to detect')
    parser.add_argument('--mask', '-k', choices=['black', 'blur', 'pixelate', 'color'],
                        default='black', help='Mask type: black, blur, pixelate, color')
    parser.add_argument('--blur', '-b', type=int, default=51,
                        help='Blur strength (for blur mask)')
    parser.add_argument('--expansion', '-e', type=float, default=1.3,
                        help='Expand mask region by factor')
    parser.add_argument('--width', type=int, default=1280,
                        help='Video width')
    parser.add_argument('--height', type=int, default=720,
                        help='Video height')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save output video to file')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without display window')
    parser.add_argument('--no-stats', action='store_true',
                        help='Hide FPS/stats overlay (default: shown)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available methods and exit')

    args = parser.parse_args()

    if args.list:
        print("Available detection methods:")
        for m in methods:
            print(f"  - {m}")
        return

    # Print GPU info
    print_gpu_info()

    # Parse source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Initialize detector
    try:
        detector = get_detector(
            args.method,
            confidence=args.confidence,
            max_faces=args.max_faces
        )
    except Exception as e:
        print(f"Error loading {args.method} detector: {e}")
        return 1

    # Open video
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Source: {source}")
    print(f"Resolution: {width}x{height}")
    print(f"Method: {args.method}")

    # Video writer
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving to: {args.output}")

    frame_count = 0
    start_time = time.time()
    last_faces = []

    print("\nPress 'q' to quit, 'm' to cycle methods")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    print("Reconnecting...")
                    time.sleep(1)
                    cap = cv2.VideoCapture(source)
                    continue
                break

            frame_count += 1

            # Detect faces
            t0 = time.time()
            faces = detector.detect(frame)
            det_time = (time.time() - t0) * 1000

            if faces:
                last_faces = faces
            elif last_faces:
                faces = last_faces  # Use last detection briefly

            # Mask faces
            result = mask_faces(frame, faces, args.expansion, args.mask, args.blur)

            # Stats overlay (optional)
            if not args.no_stats:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(result, f'{args.method.upper()} | FPS: {fps_actual:.1f} | Faces: {len(faces)} | Det: {det_time:.0f}ms',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(result)

            if not args.no_display:
                cv2.imshow('Face Blur', result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        detector.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        fps_avg = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({fps_avg:.1f} FPS)")


if __name__ == '__main__':
    sys.exit(main() or 0)
