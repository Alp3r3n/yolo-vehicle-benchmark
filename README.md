# YOLO Vehicle Detection Benchmark

Benchmark YOLOv8 vehicle detection on a local traffic video and compare CPU vs CUDA inference performance.

The project reads `CarData.mp4`, runs YOLOv8 vehicle-class detection, writes annotated videos, and saves benchmark summaries as JSON reports.

## Current Results

These are the real results currently saved in `outputs/reports/`. They were generated on May 6, 2026 using `yolov8n.pt`, `CarData.mp4`, and a confidence threshold of `0.35`.

Input video:

| Property | Value |
|----------|-------|
| File | `CarData.mp4` |
| Resolution | 3840x2160 |
| Frames | 484 |
| Source FPS | 28.57 |

Benchmark comparison:

| Metric | CPU | CUDA |
|--------|-----|------|
| Average inference time | 33.037 ms/frame | 14.432 ms/frame |
| Average inference FPS | 30.27 FPS | 69.29 FPS |
| Total processing time | 44.618 sec | 35.620 sec |
| Total vehicle detections | 9,926 | 9,925 |
| Minimum inference time | 23.599 ms | 6.492 ms |
| Maximum inference time | 1539.456 ms | 3656.784 ms |

CUDA was **2.29x faster** than CPU by average inference latency in the saved comparison report.

The maximum latency values include large one-off outliers, so the average inference time is the better headline metric for comparing CPU and CUDA. The one-detection difference between CPU and CUDA is also normal for floating-point inference across devices.

## Output Files

The current run produced:

| File | Description |
|------|-------------|
| `outputs/videos/detected_cpu.mp4` | Annotated CPU output video |
| `outputs/videos/detected_cuda.mp4` | Annotated CUDA output video |
| `outputs/reports/benchmark_cpu.json` | CPU benchmark metrics |
| `outputs/reports/benchmark_cuda.json` | CUDA benchmark metrics |
| `outputs/reports/comparison_report.json` | CPU vs CUDA comparison |

Current `comparison_report.json`:

```json
{
    "cpu_avg_inference_ms": 33.037,
    "cuda_avg_inference_ms": 14.432,
    "cpu_avg_fps": 30.27,
    "cuda_avg_fps": 69.29,
    "speedup": 2.29
}
```

## Project Structure

```text
yolo-vehicle-benchmark/
├── main.py
├── detector.py
├── video_utils.py
├── benchmark.py
├── config.py
├── requirements.txt
├── README.md
├── CarData.mp4
├── yolov8n.pt
└── outputs/
    ├── videos/
    │   ├── detected_cpu.mp4
    │   └── detected_cuda.mp4
    └── reports/
        ├── benchmark_cpu.json
        ├── benchmark_cuda.json
        └── comparison_report.json
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA benchmarking, install a PyTorch build that matches your GPU driver and CUDA runtime. If CUDA is not available, the detector prints a warning and uses CPU instead.

## Run The Benchmark

CUDA run:

```bash
python main.py --source CarData.mp4 --device cuda --model yolov8n.pt --conf 0.35 --save-video
```

CPU run:

```bash
python main.py --source CarData.mp4 --device cpu --model yolov8n.pt --conf 0.35 --save-video
```

After both runs complete, `outputs/reports/comparison_report.json` is regenerated automatically.

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `CarData.mp4` | Input video path |
| `--device` | `cuda` | Inference device: `cpu` or `cuda` |
| `--model` | `yolov8n.pt` | YOLO weights file |
| `--conf` | `0.35` | Detection confidence threshold |
| `--save-video` | enabled | Save annotated output video |

## Notes

- The saved benchmark reports measure YOLO inference time per frame plus total processing wall time.
- Hardware details are not stored in the JSON reports, so rerun the benchmark on your machine before comparing against other systems.
- `yolov8n.pt` is the nano YOLOv8 model; larger YOLO models may improve accuracy but will usually reduce FPS.
- The app processes local video files. It does not currently support live streams or object tracking.
