# YOLO Vehicle Detection Benchmark

A Python project that detects vehicles in a video using YOLOv8 and benchmarks CPU vs CUDA/GPU inference performance.

---

## Features

- Vehicle detection for cars, motorcycles, buses, and trucks using YOLOv8
- CPU vs GPU (CUDA) inference benchmarking
- Per-frame inference time measurement (milliseconds)
- Average FPS, min/max latency, and total detection counts
- Annotated output video with bounding boxes
- JSON benchmark reports per run
- Automatic comparison report when both CPU and CUDA results are available
- Graceful CUDA fallback to CPU if GPU is unavailable

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection model |
| OpenCV (`opencv-python`) | Video I/O, frame drawing |
| PyTorch | Backend for YOLO, CUDA support |
| Python 3.9+ | Language |

---

## Folder Structure

```
yolo-vehicle-benchmark/
├── main.py               # CLI entry point
├── detector.py           # YOLO model wrapper
├── video_utils.py        # Video I/O and annotation helpers
├── benchmark.py          # Timing, metrics, and report generation
├── config.py             # Default settings (model, paths, classes)
├── requirements.txt      # Python dependencies
├── README.md
├── CarData.mp4           # Local input video (ignored by Git)
└── outputs/              # Generated benchmark files (ignored by Git)
    ├── videos/
    │   ├── detected_cuda.mp4
    │   └── detected_cpu.mp4
    ├── reports/
    │   ├── benchmark_cuda.json
    │   ├── benchmark_cpu.json
    │   └── comparison_report.json
    └── screenshots/
```

---

## Installation

**1. Clone the repo (or copy the files) and enter the directory:**

```bash
cd yolo-vehicle-benchmark
```

**2. Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

> YOLOv8 will automatically download `yolov8n.pt` on first run.

**4. Add an input video:**

Place your own video in the project root as `CarData.mp4`, or pass a custom path with `--source`.

**CUDA users:** Make sure you install the PyTorch version that matches your CUDA toolkit.  
Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the correct install command.

---

## How to Run

### CUDA / GPU Benchmark

```bash
python main.py --source CarData.mp4 --device cuda --model yolov8n.pt --conf 0.35 --save-video
```

### CPU Benchmark

```bash
python main.py --source CarData.mp4 --device cpu --model yolov8n.pt --conf 0.35 --save-video
```

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `CarData.mp4` | Path to the input video |
| `--device` | `cuda` | Inference device: `cpu` or `cuda` |
| `--model` | `yolov8n.pt` | YOLO model weights file |
| `--conf` | `0.35` | Detection confidence threshold |
| `--save-video` | `True` | Save annotated output video |

---

## Example Output

Console output after a run:

```
Selected device: cuda
Input : CarData.mp4  |  Frames: 450  |  FPS: 30.0  |  Resolution: 1280x720
Output video will be saved to: outputs/videos/detected_cuda.mp4

Processing frames...
  [50/450] 11.1%
  [100/450] 22.2%
  ...
Benchmark report saved: outputs/reports/benchmark_cuda.json

--- Benchmark Results ---
  model                        yolov8n.pt
  source                       CarData.mp4
  device                       cuda
  total_frames                 450
  total_detections             1832
  avg_inference_ms             8.214
  min_inference_ms             6.102
  max_inference_ms             22.417
  avg_fps                      121.75
  total_processing_time_sec    42.331
```

**benchmark_cuda.json:**

```json
{
    "model": "yolov8n.pt",
    "source": "CarData.mp4",
    "device": "cuda",
    "total_frames": 450,
    "total_detections": 1832,
    "avg_inference_ms": 8.214,
    "min_inference_ms": 6.102,
    "max_inference_ms": 22.417,
    "avg_fps": 121.75,
    "total_processing_time_sec": 42.331
}
```

**comparison_report.json** (generated after running both CPU and CUDA):

```json
{
    "cpu_avg_inference_ms": 48.5,
    "cuda_avg_inference_ms": 8.2,
    "cpu_avg_fps": 20.6,
    "cuda_avg_fps": 121.7,
    "speedup": 5.91
}
```

---

## Limitations

- Only detects the four COCO vehicle classes: car, motorcycle, bus, truck.
- No object tracking — each frame is detected independently.
- `yolov8n` is the nano model; accuracy is lower than larger variants.
- Video must be a local file; live streams are not currently supported.
- First frame on CUDA may be slower due to GPU warm-up.

---
