"""Reproduce the ResistorPrediction repository's original SVM pipeline.

This is a source-controlled, headless extraction of the executable cells in
``production_resistor_prediction.ipynb``.  It intentionally preserves the
original preprocessing assumptions, first-row profile, extrema consensus,
Cb/Cr/S/Y SVM, heuristics, and 4-band resistance rules.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2 as cv
import joblib
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/sources/github-resistorprediction"
MODEL_ROOT = SOURCE_ROOT / "models"


def _load_datacleaner():
    module_path = SOURCE_ROOT / "scripts/datacleaner.py"
    spec = importlib.util.spec_from_file_location("resistorprediction_datacleaner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smooth_data(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def find_local_maxima(data: np.ndarray, window_size: int) -> np.ndarray:
    return argrelextrema(data, np.greater, order=window_size)[0]


def find_local_minima(data: np.ndarray, window_size: int) -> np.ndarray:
    return argrelextrema(data, np.less, order=window_size)[0]


def remove_overlapping_points(values: np.ndarray, window_threshold: int = 20) -> list[float]:
    if len(values) <= 0:
        return values.tolist()
    filtered: list[float] = []
    current_group = [float(values[0]), float(values[0])]
    for value in values:
        value = float(value)
        if abs(value - current_group[0]) <= window_threshold:
            current_group[-1] = max(value, current_group[0])
        else:
            filtered.append(float(np.mean(current_group)))
            current_group = [value, value]
    filtered.append(float(np.mean(current_group)))
    return filtered


def get_critical_points(
    image: np.ndarray,
    smoothing_window: int = 41,
    window_size_for_extrema: int = 28,
) -> tuple[list[float], list[float], list[float], list[float]]:
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv_row = hsv_image[0, :, :]
    h_values, s_values, v_values = hsv_row[:, 0], hsv_row[:, 1], hsv_row[:, 2]
    smoothed_h = smooth_data(h_values, smoothing_window)
    smoothed_s = smooth_data(s_values, smoothing_window)
    smoothed_v = smooth_data(v_values, smoothing_window)
    h_maxima = find_local_maxima(smoothed_h, window_size_for_extrema)
    s_maxima = find_local_maxima(smoothed_s, window_size_for_extrema)
    s_minima = find_local_minima(smoothed_s, window_size_for_extrema)
    v_maxima = find_local_maxima(smoothed_v, window_size_for_extrema)
    v_minima = find_local_minima(smoothed_v, window_size_for_extrema)
    red_channel = image[0, :, 2]
    smoothed_red = smooth_data(red_channel, smoothing_window)
    red_minima = find_local_minima(smoothed_red, window_size_for_extrema)
    return (
        remove_overlapping_points(h_maxima),
        remove_overlapping_points(np.concatenate((s_maxima, s_minima))),
        remove_overlapping_points(np.concatenate((v_maxima, v_minima))),
        remove_overlapping_points(red_minima),
    )


def get_consensus_roi(
    set_a: list[float],
    set_b: list[float],
    set_c: list[float],
    consensus_width: int = 41,
    roi_padding: int = 35,
) -> list[list[int]]:
    data = np.concatenate([set_a, set_b, set_c]) if (set_a or set_b or set_c) else np.array([])
    if len(data) == 0:
        return []
    sorted_data = np.sort(data)
    rois: list[list[float]] = []
    roi: list[float] = []
    prev = sorted_data[0] - consensus_width - 1
    for point in sorted_data:
        if point - prev > consensus_width:
            if len(roi) > 1:
                rois.append(roi)
            roi = [float(point)]
            prev = point
        else:
            roi.append(float(point))
    if len(roi) > 1:
        rois.append(roi)
    return [[int(pair[0]), int(pair[0] + roi_padding)] for pair in rois]


def get_regions_of_interest(
    image: np.ndarray,
    smoothing_window: int = 41,
    window_size_for_extrema: int = 28,
    consensus_width: int = 41,
    padding: int = 35,
) -> list[list[int]]:
    points = get_critical_points(image, smoothing_window, window_size_for_extrema)
    return get_consensus_roi(points[0], points[1], np.concatenate((points[2], points[3])).tolist(), consensus_width, padding)


def get_1d_slice(image: np.ndarray) -> pd.DataFrame:
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    _, a, b = cv.split(lab_image)
    ycrcb_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb_image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_image)
    return pd.DataFrame(
        {
            "col": list(range(image.shape[1])),
            "L": lab_image[0, :, 0].flatten(),
            "a": a[0, :].flatten(),
            "b": b[0, :].flatten(),
            "R": image[0, :, 2].flatten(),
            "G": image[0, :, 1].flatten(),
            "B": image[0, :, 0].flatten(),
            "Y": y[0, :].flatten(),
            "Cr": cr[0, :].flatten(),
            "Cb": cb[0, :].flatten(),
            "H": h[0, :].flatten(),
            "S": s[0, :].flatten(),
            "V": v[0, :].flatten(),
        }
    )


def _predict_colors(image: np.ndarray, classifier) -> np.ndarray:
    features = get_1d_slice(image).loc[:, ["Cb", "Cr", "S", "Y"]]
    return classifier.predict_proba(features)


def _predict_colors_in_roi(roi, image, classifier, label_encoder, threshold: float = 0.5) -> str:
    start_col, end_col = int(roi[0]), int(roi[1])
    roi_image = image[:, start_col:end_col, :]
    probabilities = _predict_colors(roi_image, classifier)
    unknown = int(label_encoder.transform(["_Unknown"])[0])
    labels = [int(np.argmax(row)) if float(np.max(row)) >= threshold else unknown for row in probabilities]
    label_counts = np.bincount(labels)
    most_common = int(np.argmax(label_counts))
    if most_common == unknown:
        sorted_counts = np.argsort(label_counts)[::-1]
        if len(sorted_counts) > 1 and label_counts[sorted_counts[1]] > len(labels) / 5:
            most_common = int(sorted_counts[1])
    return str(label_encoder.inverse_transform([most_common])[0])


def predict_colors_for_rois(rois, image, classifier, label_encoder, threshold: float = 0.5):
    return [
        [_predict_colors_in_roi(roi, image, classifier, label_encoder, threshold), (roi[0] + roi[1]) // 2]
        for roi in rois
    ]


def correct_grays(image: np.ndarray, predictions: list[list], gray_threshold: int = 50):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    silvers = []
    highest_value = 0
    for index, (color, column) in enumerate(predictions):
        if color == "Silver":
            value = int(hsv_image[0, int(column), 2])
            highest_value = max(value, highest_value)
            silvers.append((index, value))
    for index, value in silvers:
        if highest_value - value >= gray_threshold:
            predictions[index][0] = "Grey"
    return predictions


def process_predictions(image: np.ndarray, predictions: list[list], bands: int = 4, min_distance: int = 150):
    res = []
    chosen_indexes = []
    for index, (prediction, _) in enumerate(predictions):
        if prediction in ("_Unknown", "Noise"):
            continue
        res.append(index)
    prev_centroid = -2000
    prev_color = ""
    for index in res:
        # Preserve the original notebook's logical expression (always true for
        # any single color), because this is a literal reproduction benchmark.
        if predictions[index][0] != "Silver" or predictions[index][0] != "Gold":
            chosen_indexes.append(index)
            prev_color, prev_centroid = predictions[index]
        elif predictions[index][1] - prev_centroid >= min_distance:
            chosen_indexes.append(index)
            prev_color, prev_centroid = predictions[index]
        if predictions[index][1] - prev_centroid >= min_distance:
            chosen_indexes.append(index)
            prev_color, prev_centroid = predictions[index]
    selected = [predictions[index] for index in chosen_indexes]
    selected = correct_grays(image, selected)
    index = 0
    while index < 2 and len(selected) > 2:
        if selected[index][0] in ("Silver", "Gold"):
            selected.pop(index)
            index -= 1
        index += 1
    final_prediction: list[str] = []
    if len(selected) > bands:
        prev_color = ""
        for color, _ in selected:
            if prev_color == color and prev_color in ("Silver", "Gold"):
                continue
            prev_color = color
            final_prediction.append(color)
        if len(final_prediction) > bands:
            final_prediction = final_prediction[:bands]
    else:
        final_prediction = [color for color, _ in selected]
    return final_prediction


BAND_ONE_MAP = {"Brown": 1, "Red": 2, "Orange": 3, "Yellow": 4, "Green": 5, "Blue": 6, "Purple": 7, "Grey": 8, "White": 9}
BAND_TWO_MAP = {"Black": 0, **BAND_ONE_MAP}
BAND_THREE_MAP = {"Black": 1, "Brown": 1e1, "Red": 1e2, "Orange": 1e3, "Yellow": 1e4, "Green": 1e5, "Blue": 1e6, "Purple": 1e7, "Grey": 1e8, "White": 1e9, "Gold": 0.1, "Silver": 0.01}
BAND_FOUR_MAP = {"Brown": 1, "Red": 2, "Orange": 0.05, "Yellow": 0.02, "Green": 0.5, "Blue": 0.25, "Purple": 0.1, "Grey": 0.01, "Gold": 5, "Silver": 10}


def get_resistance_tolerance(colors: list[str]):
    if len(colors) < 3 or BAND_ONE_MAP.get(colors[0]) is None:
        return "NaN", "NaN"
    resistance = (BAND_ONE_MAP[colors[0]] * 10 + BAND_TWO_MAP[colors[1]]) * BAND_THREE_MAP[colors[2]]
    tolerance = BAND_FOUR_MAP.get(colors[3], 0) if len(colors) == 4 else 0
    return resistance, tolerance


def _answer(filename: str) -> list[str]:
    return filename.split("-")


def _num_correct(answer: list[str], prediction: list[str]) -> int:
    return sum(
        index < len(answer) and answer[index].lower() == color.lower()
        for index, color in enumerate(prediction)
    )


def run(image_dir: Path, model_dir: Path, output_dir: Path, original_csv: Path | None) -> dict:
    classifier = joblib.load(model_dir / "clf_CbCrSY_20230724.pkl")
    label_encoder = joblib.load(model_dir / "label_encoder.pkl")
    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        answer = _answer(image_path.name)
        if len(answer) != 5:
            continue
        image = cv.imread(str(image_path))
        if image is None:
            continue
        rois = get_regions_of_interest(image)
        predictions = predict_colors_for_rois(rois, image, classifier, label_encoder)
        band_prediction = process_predictions(image, predictions, min_distance=150)
        resistance, tolerance = get_resistance_tolerance(band_prediction)
        rows.append(
            {
                "filename": image_path.name,
                "answer": answer[:4],
                "color_predict": "-".join(color.lower() for color in band_prediction),
                "predicted_colors": band_prediction,
                "correct_colors": _num_correct(answer, band_prediction),
                "resistance_predict": resistance,
                "tolerance_predict": tolerance,
                "roi_count": len(rois),
                "rois": rois,
            }
        )
    accepted = sum(row["correct_colors"] in (3, 4) for row in rows)
    perfect = sum(row["correct_colors"] == 4 for row in rows)
    partial = sum(row["correct_colors"] == 3 for row in rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "final_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "color_predict", "correct_colors", "resistance_predict", "tolerance_predict"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    summary = {
        "source": "nichyjt/ResistorPrediction",
        "model": "models/clf_CbCrSY_20230724.pkl",
        "images": len(rows),
        "accepted_3_or_4": accepted,
        "accepted_rate": accepted / len(rows) if rows else 0.0,
        "perfect_4": perfect,
        "perfect_rate": perfect / len(rows) if rows else 0.0,
        "partial_3": partial,
        "partial_rate": partial / len(rows) if rows else 0.0,
        "mean_correct_colors": float(np.mean([row["correct_colors"] for row in rows])) if rows else 0.0,
        "original_csv_comparable": bool(original_csv and original_csv.exists()),
    }
    if original_csv and original_csv.exists():
        original_rows = list(csv.DictReader(original_csv.open(encoding="utf-8")))
        original_by_name = {row.get("filename", ""): row for row in original_rows}
        summary["rows_matching_original_color_prediction"] = sum(
            row["color_predict"] == original_by_name.get(row["filename"], {}).get("color_predict")
            for row in rows
            if row["filename"] in original_by_name
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=SOURCE_ROOT / "data/labelling")
    parser.add_argument("--model-dir", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/results/resistorprediction-original")
    parser.add_argument("--original-csv", type=Path, default=SOURCE_ROOT / "results/final_evaluation.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.image_dir, args.model_dir, args.output_dir, args.original_csv), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

