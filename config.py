# Default configuration for YOLO Vehicle Detection Benchmark

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35
DEFAULT_DEVICE = "cuda"
INPUT_VIDEO = "CarData.mp4"

# COCO class IDs that correspond to vehicle types
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Output directory paths (relative to project root)
OUTPUT_VIDEO_DIR = "outputs/videos"
OUTPUT_REPORT_DIR = "outputs/reports"
OUTPUT_SCREENSHOT_DIR = "outputs/screenshots"
