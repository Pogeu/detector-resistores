import numpy as np

from resistor_detector.matlab_compat import analyze_matlab_compatible


def test_matlab_compat_returns_detector_schema_on_blank_frame():
    result = analyze_matlab_compatible(np.full((96, 128, 3), 255, dtype=np.uint8))
    assert result["candidate_count"] >= 0
    assert result["accepted_count"] == len(result["detections"])
    assert result["values"] == [item["value_ohms"] for item in result["detections"]]
