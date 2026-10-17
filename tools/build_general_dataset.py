"""Build a provenance-preserving aggregate dataset from all available sources.

The sources in this project do not share one label space: some contain physical
resistor values, others contain component boxes, circuit symbols, or generic
component classes.  This tool therefore creates two related artifacts:

* ``manifest.jsonl``: one index containing every discovered image/task and its
  original path, label and source provenance;
* ``yolo/``: a normalized one-class ``resistor`` detector dataset made only
  from compatible annotations.  Original files are copied, not moved.

The aggregate is intentionally conservative: a generic component image is not
pretended to have a resistor-value label, and a circuit diagram is not mixed
with a photograph in the value benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources"
DERIVED = ROOT / "data" / "derived"


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda item: int(item))]
    return []


def _resolve(value: str | Path, base: Path, data_root: Path | None = None) -> Path:
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((base / path, (data_root or base) / path))
        if str(path).startswith("../"):
            candidates.append(base / str(path)[3:])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


def _safe_split(value: str) -> str:
    value = value.lower()
    return "val" if value in {"valid", "validation", "dev"} else value


def _source_from(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
        if relative.parts and relative.parts[0] == "data" and len(relative.parts) > 2:
            return relative.parts[2]
    except ValueError:
        pass
    return path.name


def _is_resistor_name(name: str, source_id: str) -> bool:
    token = _norm(name)
    source = _norm(source_id)
    if "resistor" in token or token in {"r", "r1", "r2", "res"}:
        return True
    # These sources are documented as single-purpose resistor datasets even
    # when their labels use r1/r2 or class 0 rather than the word resistor.
    return source in {
        "kaggle nyahmet",
        "github vinit5893",
        "github hires",
        "github share2code99",
    }


def _iter_images(directory: Path) -> Iterable[Path]:
    if directory.is_file() and directory.suffix.lower() in IMAGE_EXTS:
        yield directory
        return
    if directory.exists():
        yield from sorted(
            path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )


def _read_yolo_records(
    image_dir: Path,
    label_dir: Path,
    source_id: str,
    split: str,
    names: list[str],
    seen: set[Path],
) -> list[dict[str, Any]]:
    records = []
    from PIL import Image

    for image_path in _iter_images(image_dir):
        image_path = image_path.resolve()
        if image_path in seen:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except OSError:
            continue
        objects = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                class_id = int(float(fields[0]))
                bbox = [float(item) for item in fields[1:5]]
            except ValueError:
                continue
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            if names and not _is_resistor_name(class_name, source_id):
                continue
            if not names and not _is_resistor_name("resistor", source_id):
                continue
            center_x, center_y, box_width, box_height = bbox
            objects.append(
                {
                    "class_name": "resistor",
                    "bbox_yolo": bbox,
                    "bbox": [
                        (center_x - box_width / 2) * image_width,
                        (center_y - box_height / 2) * image_height,
                        box_width * image_width,
                        box_height * image_height,
                    ],
                }
            )
        if not objects:
            continue
        seen.add(image_path)
        records.append(
            {
                "id": f"general-{source_id}-{len(seen):07d}",
                "source_id": source_id,
                "path": image_path.relative_to(ROOT).as_posix(),
                "split": _safe_split(split),
                "task": "object_detection",
                "annotation_format": "yolo",
                "annotation_path": label_path.relative_to(ROOT).as_posix(),
                "objects": objects,
            }
        )
    return records


def _yaml_specs() -> Iterable[tuple[str, Path, Path, Path, str, list[str]]]:
    """Yield (source, yaml, images, labels, split, class names)."""
    for yaml_path in sorted(SOURCES.rglob("*.yaml")):
        if ".git" in yaml_path.parts:
            continue
        try:
            import yaml

            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = _names(config.get("names"))
        data_root = _resolve(config.get("path", "."), yaml_path.parent)
        source_id = _source_from(yaml_path)
        for split in ("train", "val", "valid", "test"):
            value = config.get(split)
            if value is None or not isinstance(value, (str, Path)):
                continue
            image_dir = _resolve(value, yaml_path.parent, data_root)
            if image_dir.name != "images" and (image_dir.parent / "labels").exists():
                label_dir = image_dir.parent / "labels"
            else:
                label_dir = image_dir.parent / "labels"
            yield source_id, yaml_path, image_dir, label_dir, split, names


def _manual_specs() -> Iterable[tuple[str, Path, Path, str, list[str]]]:
    """Yield source, images, labels, split, names for non-YAML exports."""
    # Kaggle's circuit-drawing export stores all images/labels in obj/.
    nyahmet = SOURCES / "kaggle-nyahmet" / "obj"
    if nyahmet.exists():
        yield "kaggle-nyahmet", nyahmet, nyahmet, "train", ["resistor"]

    vinit = SOURCES / "github-vinit5893"
    for split in ("train", "valid", "test"):
        images = vinit / split / "images"
        labels = vinit / split / "labels"
        if images.exists() and labels.exists():
            yield "github-vinit5893", images, labels, split, ["r1", "r2"]

    aitee = SOURCES / "github-aitee" / "extracted" / "dataset" / "split"
    for split in ("train", "val"):
        images = aitee / "images" / split
        labels = aitee / "labels" / split
        if images.exists() and labels.exists():
            yield "github-aitee", images, labels, split, ["E", "GND", "I", "P", "R", "V"]

    # The derived segmentation export has polygon labels.  Its bounding boxes
    # are already available through data/manifests, so it is not duplicated here.


def _voc_records(source_id: str, root: Path, seen: set[Path]) -> list[dict[str, Any]]:
    records = []
    for xml_path in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        image_name = tree.findtext("filename")
        if not image_name:
            continue
        image_candidates = [xml_path.parent / image_name, xml_path.parent.parent / "JPEGImages" / image_name]
        image_path = next((path for path in image_candidates if path.exists()), None)
        if image_path is None or image_path.resolve() in seen:
            continue
        objects = []
        for obj in tree.findall("object"):
            name = obj.findtext("name", "")
            if not _is_resistor_name(name, source_id):
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                width = float(tree.findtext("size/width", "1"))
                height = float(tree.findtext("size/height", "1"))
                bbox = [((xmin + xmax) / 2) / width, ((ymin + ymax) / 2) / height,
                        (xmax - xmin) / width, (ymax - ymin) / height]
            except ValueError:
                continue
            objects.append(
                {
                    "class_name": "resistor",
                    "bbox_yolo": bbox,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                }
            )
        if not objects:
            continue
        seen.add(image_path.resolve())
        records.append({
            "id": f"general-{source_id}-{len(seen):07d}",
            "source_id": source_id,
            "path": image_path.resolve().relative_to(ROOT).as_posix(),
            "split": "train" if int(hashlib.sha1(str(image_path).encode()).hexdigest()[:2], 16) < 205 else "val",
            "task": "object_detection",
            "annotation_format": "voc",
            "annotation_path": xml_path.relative_to(ROOT).as_posix(),
            "objects": objects,
        })
    return records


def _classification_records() -> list[dict[str, Any]]:
    records = []
    roots = {
        "kaggle-electronic-components": SOURCES / "kaggle-electronic-components",
        "kaggle-electronic-component-recognizer": SOURCES / "kaggle-electronic-component-recognizer",
        "github-resistorprediction": SOURCES / "github-resistorprediction" / "data" / "labelling",
    }
    for source_id, root in roots.items():
        if not root.exists():
            continue
        for image_path in _iter_images(root):
            relative = image_path.relative_to(root)
            label = relative.parts[-2] if len(relative.parts) > 1 else "unknown"
            records.append({
                "id": f"general-{source_id}-{len(records):07d}",
                "source_id": source_id,
                "path": image_path.relative_to(ROOT).as_posix(),
                "task": "component_classification" if source_id.startswith("kaggle-") else "value_classification",
                "label": label,
                "media_type": "image",
            })
    return records


def _load_existing_manifests() -> list[dict[str, Any]]:
    records = []
    manifest_dir = ROOT / "data" / "manifests"
    for path in sorted(manifest_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = dict(record)
            record["aggregate_manifest"] = path.name
            records.append(record)
    return records


def _unlabeled_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Index images that have no compatible annotation without inventing labels."""
    known = {record["path"] for record in records if record.get("path")}
    extra = []
    for image_path in sorted(SOURCES.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if ".git" in image_path.parts:
            continue
        relative = image_path.relative_to(ROOT).as_posix()
        if relative in known:
            continue
        source_id = _source_from(image_path)
        extra.append(
            {
                "id": f"general-{source_id}-unlabeled-{len(extra):07d}",
                "source_id": source_id,
                "path": relative,
                "task": "unlabeled_image",
                "label_status": "not_annotated_or_not_compatible",
                "media_type": "image",
            }
        )
        known.add(relative)
    return extra


def _copy_yolo(records: list[dict[str, Any]], output: Path) -> None:
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    counters = Counter()
    for record in records:
        split = _safe_split(record.get("split", "train"))
        if split not in {"train", "val", "test"}:
            split = "train"
        source = _norm(record["source_id"]).replace(" ", "_")
        counters[source] += 1
        image_path = ROOT / record["path"]
        stem = f"{source}__{counters[source]:07d}"
        destination = output / "images" / split / f"{stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)
        lines = []
        for obj in record.get("objects", []):
            bbox = obj.get("bbox_yolo")
            if bbox and len(bbox) == 4:
                lines.append("0 " + " ".join(f"{float(value):.8f}" for value in bbox))
        (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_text = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames: ['resistor']\n"
    )
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")


def build(copy_files: bool = True) -> dict[str, Any]:
    output = DERIVED / "general-dataset"
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for source_id, _yaml, images, labels, split, names in _yaml_specs():
        found = _read_yolo_records(images, labels, source_id, split, names, seen)
        records.extend(found)
        object_records.extend(found)
    for source_id, images, labels, split, names in _manual_specs():
        found = _read_yolo_records(images, labels, source_id, split, names, seen)
        records.extend(found)
        object_records.extend(found)

    power = SOURCES / "github-power-equipment" / "substation object detection"
    if power.exists():
        found = _voc_records("github-power-equipment", power, seen)
        records.extend(found)
        object_records.extend(found)

    records.extend(_classification_records())
    records.extend(_load_existing_manifests())
    records.extend(_unlabeled_records(records))

    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    object_manifest_path = output / "object-detection-manifest.jsonl"
    with object_manifest_path.open("w", encoding="utf-8") as handle:
        for record in object_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    value_records = [
        record for record in records
        if record.get("task") in {"crop_value", "value_classification"}
    ]
    value_manifest_path = output / "value-manifest.jsonl"
    with value_manifest_path.open("w", encoding="utf-8") as handle:
        for record in value_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if copy_files:
        _copy_yolo(object_records, output / "yolo")

    source_dirs = sorted(path.name for path in SOURCES.iterdir() if path.is_dir())
    summary = {
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "object_manifest": object_manifest_path.relative_to(ROOT).as_posix(),
        "value_manifest": value_manifest_path.relative_to(ROOT).as_posix(),
        "yolo_data_yaml": (output / "yolo" / "data.yaml").relative_to(ROOT).as_posix() if copy_files else None,
        "record_count": len(records),
        "object_detection_count": len(object_records),
        "task_counts": dict(Counter(record.get("task", "unknown") for record in records)),
        "source_counts": dict(Counter(record.get("source_id", "unknown") for record in records)),
        "object_source_counts": dict(Counter(record.get("source_id", "unknown") for record in object_records)),
        "source_directories_seen": source_dirs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true", help="do not copy the unified YOLO files")
    args = parser.parse_args()
    build(copy_files=not args.manifest_only)


if __name__ == "__main__":
    main()
