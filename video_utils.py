import os
import cv2


def open_video(source: str):
    """
    Open a video file with OpenCV.

    Returns:
        cap, width, height, fps, total_frames
    Raises:
        FileNotFoundError: if the file does not exist.
        IOError: if OpenCV cannot open the file.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"Input video not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, width, height, fps, total_frames


def create_video_writer(output_path: str, width: int, height: int, fps: float):
    """Create a VideoWriter that saves to output_path as MP4."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


def draw_detections(frame, detections: list):
    """
    Draw bounding boxes and labels on a copy of frame.

    Returns the annotated frame (modifies in-place and returns it).
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        # Green rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text above the box
        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 2, y1), (0, 255, 0), -1)
        cv2.putText(
            frame, text, (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return frame
