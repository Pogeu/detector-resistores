"""Evaluate the HiRes pipeline's results.txt against filename-derived values."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

UNIT = {"Ω": 1.0, "kΩ": 1_000.0, "MΩ": 1_000_000.0}


def parse_detected(text: str) -> float | None:
    match = re.search(r"(?:resistor\s+\d+:\s*)?([0-9]+(?:\.[0-9]+)?)\s*(Ω|kΩ|MΩ)\s*±", text)
    if not match:
        return None
    return float(match.group(1)) * UNIT[match.group(2)]


def evaluate(results_path: Path, manifest_path: Path, output_path: Path) -> dict:
    truth = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        truth[Path(record["path"]).name] = float(record["objects"][0]["ohms"])

    predictions = {}
    current_filename = None
    for line in results_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.lower().endswith((".jpg", ".jpeg", ".png")) and not stripped.startswith("resistor"):
            current_filename = stripped
            continue
        match = re.match(r"resistor\s+0:\s+(.+)$", stripped)
        if match and current_filename:
            predictions[current_filename] = parse_detected(match.group(1))

    exact = 0
    detected = 0
    errors = []
    for filename, expected in truth.items():
        predicted = predictions.get(filename)
        if predicted is None:
            continue
        detected += 1
        if abs(predicted - expected) < 1e-9:
            exact += 1
        else:
            errors.append({"image": filename, "expected_ohms": expected, "predicted_ohms": predicted})

    summary = {
        "results": str(results_path),
        "manifest": str(manifest_path),
        "images": len(truth),
        "detected_first_resistor": detected,
        "detection_rate": detected / len(truth) if truth else 0.0,
        "nominal_exact": exact / len(truth) if truth else 0.0,
        "nominal_exact_when_detected": exact / detected if detected else 0.0,
        "mismatches": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "mismatches"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.results, args.manifest, args.output)


if __name__ == "__main__":
    main()
