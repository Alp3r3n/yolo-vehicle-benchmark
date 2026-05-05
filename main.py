"""
YOLO Vehicle Detection Benchmark
Entry point: parse CLI args, run the detection pipeline, save outputs.
"""

import argparse
import os

from config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    INPUT_VIDEO,
    MODEL_NAME,
    OUTPUT_REPORT_DIR,
    OUTPUT_VIDEO_DIR,
)
from benchmark import BenchmarkRecorder, generate_comparison_report
from detector import VehicleDetector
from video_utils import create_video_writer, draw_detections, open_video


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Vehicle Detection Benchmark")
    parser.add_argument("--source", default=INPUT_VIDEO, help="Path to the input video.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device for YOLO inference (cpu or cuda).",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="YOLO model weights file.")
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Detection confidence threshold (0.0 – 1.0).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save annotated output video.",
    )
    return parser.parse_args()


def run(args):
    # Derive output paths from the selected device tag
    tag = args.device
    output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"detected_{tag}.mp4")
    report_path = os.path.join(OUTPUT_REPORT_DIR, f"benchmark_{tag}.json")

    # --- Detector ---
    detector = VehicleDetector(args.model, args.device)

    # --- Video ---
    cap, width, height, src_fps, total_frames = open_video(args.source)
    print(
        f"Input : {args.source}  |  Frames: {total_frames}  |  "
        f"FPS: {src_fps:.1f}  |  Resolution: {width}x{height}"
    )

    writer = None
    if args.save_video:
        writer = create_video_writer(output_video_path, width, height, src_fps)
        print(f"Output video will be saved to: {output_video_path}")

    # --- Benchmark ---
    recorder = BenchmarkRecorder(args.model, args.source, detector.device)
    recorder.start()

    frame_idx = 0
    print("\nProcessing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, inference_ms = detector.detect(frame, args.conf)
        recorder.record_frame(inference_ms, len(detections))

        if writer is not None:
            annotated = draw_detections(frame, detections)
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  [{frame_idx}/{total_frames}] {pct:.1f}%")

    cap.release()
    if writer:
        writer.release()

    # --- Results ---
    summary = recorder.save_report(report_path)

    print("\n--- Benchmark Results ---")
    for key, val in summary.items():
        print(f"  {key:<28} {val}")

    if args.save_video:
        print(f"\nAnnotated video saved: {output_video_path}")

    # Generate comparison report whenever both CPU and CUDA reports exist
    cpu_report = os.path.join(OUTPUT_REPORT_DIR, "benchmark_cpu.json")
    cuda_report = os.path.join(OUTPUT_REPORT_DIR, "benchmark_cuda.json")
    comparison_out = os.path.join(OUTPUT_REPORT_DIR, "comparison_report.json")
    generate_comparison_report(cpu_report, cuda_report, comparison_out)


if __name__ == "__main__":
    run(parse_args())
