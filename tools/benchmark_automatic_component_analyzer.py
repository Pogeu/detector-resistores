"""Benchmark the Automatic Component Analyzer against the MATLAB batch.

The upstream application only accepts a live camera and GUI trackbars.  The
headless adapter preserves its default trackbar values and algorithmic path;
this script runs it on exactly the same MathWorks sample images used by the
MATLAB comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2

# Make the benchmark runnable directly from a source checkout, without an
# editable install.  This keeps the Windows and Linux command identical.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resistor_detector.automatic_component_analyzer import analyze_frame
from resistor_detector.automatic_component_analyzer import AnalyzerConfig


def _as_values(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def _same_value(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, 1e-9 * max(abs(left), abs(right)))


def _matched_values(expected: list[float], predicted: list[float]) -> int:
    remaining = list(predicted)
    matched = 0
    for target in expected:
        for index, candidate in enumerate(remaining):
            if _same_value(target, candidate):
                matched += 1
                remaining.pop(index)
                break
    return matched


def _evaluate(
    image_dir: Path,
    matlab_rows: list[dict[str, Any]],
    config: AnalyzerConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for reference in sorted(matlab_rows, key=lambda row: row["image"].lower()):
        image_name = reference["image"]
        image_path = image_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        result = analyze_frame(image, config=config)
        expected = _as_values(reference["values"])
        predicted = [item["value"] for item in result["decoded_results"]]
        matched = _matched_values(expected, predicted)
        rows.append(
            {
                "image": image_name,
                "expected_matlab_values": expected,
                "predicted_values": predicted,
                "expected_count": len(expected),
                "predicted_count": len(predicted),
                "matched_value_count": matched,
                "all_expected_values_matched": matched == len(expected),
                "exact_value_sequence": len(expected) == len(predicted)
                and all(_same_value(a, b) for a, b in zip(expected, predicted)),
                "band_count": result["band_count"],
                "colors": result["colors"],
                "bands": result["bands"],
                "rejected": result["rejected"],
            }
        )
    expected_total = sum(row["expected_count"] for row in rows)
    predicted_total = sum(row["predicted_count"] for row in rows)
    matched_total = sum(row["matched_value_count"] for row in rows)
    metrics = {
        "gaussian_kernel_radius": config.gaussian_kernel_radius,
        "dark_threshold": config.dark_threshold,
        "images": len(rows),
        "image_count_agreement": sum(
            row["expected_count"] == row["predicted_count"] for row in rows
        ) / len(rows)
        if rows
        else 0.0,
        "exact_value_sequence_agreement": sum(
            row["exact_value_sequence"] for row in rows
        ) / len(rows)
        if rows
        else 0.0,
        "expected_value_recall": matched_total / expected_total
        if expected_total
        else 0.0,
        "expected_values": expected_total,
        "predicted_values": predicted_total,
        "matched_values": matched_total,
    }
    return metrics, rows


def run(image_dir: Path, matlab_json: Path, output_dir: Path) -> dict[str, Any]:
    matlab_rows = json.loads(matlab_json.read_text(encoding="utf-8"))
    default_metrics, rows = _evaluate(
        image_dir, matlab_rows, AnalyzerConfig()
    )
    sweep_metrics = []
    for radius in (0, 1, 3, 5, 7, 10, 15, 20):
        for dark_threshold in (20, 30, 40, 50, 60, 80, 100):
            metrics, _ = _evaluate(
                image_dir,
                matlab_rows,
                AnalyzerConfig(radius, dark_threshold),
            )
            sweep_metrics.append(metrics)
    sweep_metrics.sort(
        key=lambda item: (
            item["expected_value_recall"],
            item["exact_value_sequence_agreement"],
            item["image_count_agreement"],
        ),
        reverse=True,
    )
    summary = {
        "source": "bytesizedengineering/automatic-component-analyzer",
        "mode": "headless-static-frame-adapter",
        "reference": "MathWorks sample images + MATLAB batch output",
        "default": default_metrics,
        "tunable_parameter_sweep": {
            "radii": [0, 1, 3, 5, 7, 10, 15, 20],
            "dark_thresholds": [20, 30, 40, 50, 60, 80, 100],
            "tested_configurations": len(sweep_metrics),
            "best": sweep_metrics[0],
            "top_10": sweep_metrics[:10],
        },
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_image.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/sources/mathworks/extracted/ResistorColorCode/Resistor Images"),
    )
    parser.add_argument(
        "--matlab-json",
        type=Path,
        default=Path("data/results/matlab-comparison/matlab_original.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/automatic-component-analyzer-comparison"),
    )
    args = parser.parse_args()
    summary = run(args.image_dir, args.matlab_json, args.output_dir)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
