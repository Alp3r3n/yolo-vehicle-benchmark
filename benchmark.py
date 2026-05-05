import json
import os
import time


class BenchmarkRecorder:
    """Records per-frame inference times and computes aggregate metrics."""

    def __init__(self, model: str, source: str, device: str):
        self.model = model
        self.source = source
        self.device = device
        self.frame_times: list[float] = []
        self.total_detections = 0
        self._wall_start: float | None = None

    def start(self):
        """Mark the start of the processing session."""
        self._wall_start = time.perf_counter()

    def record_frame(self, inference_ms: float, num_detections: int):
        """Accumulate one frame's stats."""
        self.frame_times.append(inference_ms)
        self.total_detections += num_detections

    def get_summary(self) -> dict:
        """Return a dict with all benchmark metrics."""
        n = len(self.frame_times)
        if n == 0:
            return {}

        avg_ms = sum(self.frame_times) / n
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        wall_time = (time.perf_counter() - self._wall_start) if self._wall_start else 0.0

        return {
            "model": self.model,
            "source": self.source,
            "device": self.device,
            "total_frames": n,
            "total_detections": self.total_detections,
            "avg_inference_ms": round(avg_ms, 3),
            "min_inference_ms": round(min(self.frame_times), 3),
            "max_inference_ms": round(max(self.frame_times), 3),
            "avg_fps": round(avg_fps, 2),
            "total_processing_time_sec": round(wall_time, 3),
        }

    def save_report(self, output_path: str) -> dict:
        """Write the benchmark summary to a JSON file and return it."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary = self.get_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Benchmark report saved: {output_path}")
        return summary


def generate_comparison_report(cpu_path: str, cuda_path: str, output_path: str):
    """
    Load the CPU and CUDA benchmark reports and write a side-by-side comparison.
    Safe to call even if one report is missing (it will skip quietly).
    """
    if not os.path.exists(cpu_path) or not os.path.exists(cuda_path):
        return  # Nothing to compare yet

    with open(cpu_path) as f:
        cpu = json.load(f)
    with open(cuda_path) as f:
        gpu = json.load(f)

    cuda_ms = gpu.get("avg_inference_ms", 0)
    speedup = round(cpu["avg_inference_ms"] / cuda_ms, 2) if cuda_ms > 0 else 0.0

    comparison = {
        "cpu_avg_inference_ms": cpu["avg_inference_ms"],
        "cuda_avg_inference_ms": gpu["avg_inference_ms"],
        "cpu_avg_fps": cpu["avg_fps"],
        "cuda_avg_fps": gpu["avg_fps"],
        "speedup": speedup,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=4)
    print(f"Comparison report saved: {output_path}")
