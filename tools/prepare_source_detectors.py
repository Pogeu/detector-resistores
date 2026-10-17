"""Materialize one-class YOLO datasets per source from the aggregate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value.lower())


def prepare(manifest_path: Path, output_root: Path, sources: list[str]) -> None:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for source in sources:
        selected = [record for record in records if record.get("source_id") == source and record.get("task") == "object_detection"]
        if not selected:
            print(f"{source}: 0")
            continue
        output = output_root / _slug(source)
        for split in ("train", "val", "test"):
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

        has_val = any(record.get("split") == "val" for record in selected)
        has_test = any(record.get("split") == "test" for record in selected)
        counts = Counter()
        normalized = []
        for index, record in enumerate(selected):
            split = record.get("split", "train")
            if split not in {"train", "val", "test"}:
                split = "val" if has_val else "train"
            if split == "train" and not has_val:
                digest = int(hashlib.sha1(record["id"].encode()).hexdigest()[:2], 16)
                split = "val" if digest < 51 else "train"
            if split == "val" and not has_test:
                # Validation is reused as a test split only in the YAML; the
                # original manifest remains explicit and is not duplicated.
                pass
            counts[split] += 1
            image_path = ROOT / record["path"]
            stem = f"{_slug(source)}__{index:07d}"
            destination = output / "images" / split / f"{stem}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination)
            lines = []
            for obj in record.get("objects", []):
                bbox = obj.get("bbox_yolo")
                if bbox and len(bbox) == 4:
                    lines.append("0 " + " ".join(f"{float(value):.8f}" for value in bbox))
            (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            normalized.append({**record, "aggregate_split": split, "normalized_image": str(destination.relative_to(ROOT))})

        yaml_path = output / "data.yaml"
        yaml_path.write_text(
            f"path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/{'test' if has_test else 'val'}\nnc: 1\nnames: ['resistor']\n",
            encoding="utf-8",
        )
        with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for record in normalized:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"source": source, "samples": len(selected), "splits": dict(counts), "yaml": str(yaml_path.relative_to(ROOT))}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/derived/general-dataset/object-detection-manifest.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/derived/source-detectors")
    parser.add_argument("--sources", nargs="+", required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output_root, args.sources)


if __name__ == "__main__":
    main()
