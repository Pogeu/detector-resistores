"""Compare the original MATLAB implementation with its Python port."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

from resistor_detector.matlab_compat import analyze_matlab_compatible


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _matlab_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _run_matlab(image_dir: Path, output_json: Path, matlab_bin: str) -> None:
    tools_dir = Path(__file__).resolve().parent
    expression = (
        f"addpath('{_matlab_path(tools_dir)}'); "
        f"matlab_original_batch('{_matlab_path(image_dir)}','{_matlab_path(output_json)}');"
    )
    completed = subprocess.run(
        [matlab_bin, "-batch", expression],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MATLAB falhou (codigo %d):\n%s\n%s"
            % (completed.returncode, completed.stdout[-4000:], completed.stderr[-4000:])
        )


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    return payload


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def compare(image_dir: Path, output_dir: Path, matlab_bin: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matlab_json = output_dir / "matlab_original.json"
    _run_matlab(image_dir, matlab_json, matlab_bin)
    matlab_records = {record["image"]: record for record in _load_records(matlab_json)}

    comparisons: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"nao foi possivel abrir {image_path}")
        python_result = analyze_matlab_compatible(image)
        matlab_result = matlab_records.get(image_path.name, {})
        py_values = python_result["values"]
        matlab_values = [int(value) for value in _as_list(matlab_result.get("values"))]
        matched_value_count = sum(
            (Counter(py_values) & Counter(matlab_values)).values()
        )
        comparisons.append(
            {
                "image": image_path.name,
                "python_candidate_count": python_result["candidate_count"],
                "matlab_candidate_count": matlab_result.get("candidate_count"),
                "python_count": python_result["accepted_count"],
                "matlab_count": matlab_result.get("accepted_count"),
                "count_equal": python_result["accepted_count"] == matlab_result.get("accepted_count"),
                "python_values": py_values,
                "matlab_values": matlab_values,
                "python_detections": python_result["detections"],
                "matlab_bands_digits": matlab_result.get("bands_digits", []),
                "values_equal": py_values == matlab_values,
                "values_multiset_equal": sorted(py_values) == sorted(matlab_values),
                "matched_value_count": matched_value_count,
                "matlab_error": matlab_result.get("error", ""),
            }
        )

    count_equal = sum(item["count_equal"] for item in comparisons)
    values_equal = sum(item["values_equal"] for item in comparisons)
    values_multiset_equal = sum(item["values_multiset_equal"] for item in comparisons)
    matched_values = sum(item["matched_value_count"] for item in comparisons)
    matlab_values_total = sum(len(item["matlab_values"]) for item in comparisons)
    matlab_errors = sum(bool(item["matlab_error"]) for item in comparisons)
    summary = {
        "image_dir": str(image_dir),
        "images": len(comparisons),
        "matlab_engine": next(iter(matlab_records.values())).get("engine", "unknown") if matlab_records else "unknown",
        "count_agreement": count_equal / len(comparisons) if comparisons else 0.0,
        "value_sequence_agreement": values_equal / len(comparisons) if comparisons else 0.0,
        "value_multiset_agreement": values_multiset_equal / len(comparisons) if comparisons else 0.0,
        "matched_value_recall": matched_values / matlab_values_total if matlab_values_total else 0.0,
        "matlab_errors": matlab_errors,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/sources/mathworks/extracted/ResistorColorCode/Resistor Images"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/matlab-comparison"))
    parser.add_argument("--matlab", default=shutil.which("matlab") or "matlab")
    args = parser.parse_args()
    summary = compare(args.image_dir, args.output_dir, args.matlab)
    print(json.dumps({key: summary[key] for key in ("images", "count_agreement", "value_sequence_agreement", "value_multiset_agreement", "matched_value_recall", "matlab_errors")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
