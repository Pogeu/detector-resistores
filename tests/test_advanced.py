import cv2
import numpy as np

from resistor_detector.advanced import analyze_frame_auto, segment_resistor_mask
from resistor_detector.colors import classify_bgr_lab_pca
from resistor_detector.tracking import IoUTracker


def test_lab_pca_recognizes_primary_band_colors():
    assert classify_bgr_lab_pca((0, 0, 220)) == "red"
    assert classify_bgr_lab_pca((180, 80, 0)) == "blue"


def test_segmentation_returns_binary_mask():
    frame = np.full((100, 160, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (20, 40), (140, 60), (40, 40, 100), -1)
    mask = segment_resistor_mask(frame)

    assert mask.shape == frame.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})


def test_auto_decoder_accepts_four_or_five_bands():
    frame = np.full((100, 160, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (20, 35), (140, 65), (70, 70, 120), -1)
    detections = analyze_frame_auto(frame, min_area=50)

    assert all(len(detection.bands) in (4, 5) for detection in detections)


def test_iou_tracker_keeps_id_for_overlapping_boxes():
    tracker = IoUTracker(iou_threshold=0.2)
    first = tracker.update([(10, 10, 30, 20)])
    second = tracker.update([(12, 11, 30, 20)])

    assert first == second
