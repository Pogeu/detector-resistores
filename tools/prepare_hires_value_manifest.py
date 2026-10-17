"""Create a value-labeled manifest from the public HiRes test filenames."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "data" / "sources" / "github-hires" / "datasets" / "test"
OUTPUT = ROOT / "data" / "manifests" / "hires-test-value.jsonl"


def parse_value(token: str) -> float | None:
    token = token.upper().replace("Ω", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([RKM]?)([0-9]*)", token)
    if not match:
        return None
    first, unit, tail = match.groups()
    number = float(first + (f".{tail}" if tail else ""))
    return number * {"": 1.0, "R": 1.0, "K": 1_000.0, "M": 1_000_000.0}[unit]


def main() -> None:
    records = []
    for image_path in sorted(IMAGE_ROOT.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = re.match(r"(?:4B|5B)-(.+?)-T([0-9]+)-", image_path.stem, flags=re.IGNORECASE)
        if not match:
            continue
        ohms = parse_value(match.group(1))
        if ohms is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            {
                "id": f"hires-test-{image_path.stem}",
                "source_id": "github-hires",
                "path": image_path.relative_to(ROOT).as_posix(),
                "media_type": "image",
                "task": "crop_value",
                "objects": [
                    {
                        "bbox": [0, 0, width, height],
                        "bands": [],
                        "ohms": ohms,
                        "tolerance_percent": float(match.group(2)),
                    }
                ],
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(OUTPUT.relative_to(ROOT)), "samples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
