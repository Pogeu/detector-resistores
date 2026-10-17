"""Benchmark reproduzível para detecção e leitura de resistores.

O benchmark separa as métricas por etapa. Assim, um detector que encontra o
resistor, mas erra uma faixa, não fica artificialmente misturado com um modelo
que nem localizou o objeto.

O formato JSONL aceita previsões produzidas por qualquer abordagem. O comando
``run`` também executa os dois baselines clássicos disponíveis no protótipo.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

from .advanced import analyze_frame_auto, analyze_segmented_frame
from .colors import classify_bgr, classify_bgr_lab_pca, decode_color_bands
from .detection import analyze_frame, sample_band_colors


Box = tuple[float, float, float, float]

BASELINE_STRATEGIES = {
    "classical-4": 4,
    "classical-5": 5,
    "crop-classical-4": 4,
    "crop-classical-5": 5,
    "crop-lab-pca-4": 4,
    "crop-lab-pca-5": 5,
    "classical-lab-pca": 4,
    "segmentation-rule": 4,
    "hybrid-4": 4,
    "hybrid-5": 5,
    "hybrid-auto": 4,
}


@dataclass(frozen=True)
class ObjectRecord:
    bbox: Box
    bands: tuple[str, ...] = ()
    ohms: float | None = None
    tolerance_percent: float | None = None
    confidence: float = 1.0
    track_id: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ObjectRecord":
        raw_box = data.get("bbox")
        if isinstance(raw_box, dict):
            box = (
                float(raw_box["x"]),
                float(raw_box["y"]),
                float(raw_box["width"]),
                float(raw_box["height"]),
            )
        elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            box = (
                float(raw_box[0]),
                float(raw_box[1]),
                float(raw_box[2]),
                float(raw_box[3]),
            )
        else:
            raise ValueError("cada objeto precisa de bbox=[x,y,width,height]")

        value = data.get("value")
        if isinstance(value, dict):
            raw_ohms = data.get("ohms", data.get("value_ohms", value.get("ohms")))
            raw_tolerance = data.get(
                "tolerance_percent",
                value.get("tolerance_percent"),
            )
        else:
            raw_ohms = data.get("ohms", data.get("value_ohms"))
            raw_tolerance = data.get("tolerance_percent")

        return cls(
            bbox=box,
            bands=tuple(str(band).lower() for band in data.get("bands", []) if band),
            ohms=None if raw_ohms is None else float(raw_ohms),
            tolerance_percent=(
                None if raw_tolerance is None else float(raw_tolerance)
            ),
            confidence=float(data.get("confidence", 1.0)),
            track_id=None if data.get("track_id") is None else str(data["track_id"]),
        )


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    objects: tuple[ObjectRecord, ...]
    source_id: str | None = None
    frame: int | None = None
    path: str | None = None
    media_type: str = "image"
    latency_ms: float | None = None
    strategy: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SampleRecord":
        return cls(
            sample_id=str(data.get("id", data.get("sample_id", ""))),
            objects=tuple(
                ObjectRecord.from_mapping(obj) for obj in data.get("objects", [])
            ),
            source_id=(
                None if data.get("source_id") is None else str(data["source_id"])
            ),
            frame=None if data.get("frame") is None else int(data["frame"]),
            path=None if data.get("path") is None else str(data["path"]),
            media_type=str(data.get("media_type", "image")),
            latency_ms=(
                None
                if data.get("latency_ms") is None
                else float(data["latency_ms"])
            ),
            strategy=(
                None if data.get("strategy") is None else str(data["strategy"])
            ),
        )


def load_jsonl(path: str | Path) -> list[SampleRecord]:
    """Carrega JSONL ou um JSON contendo uma lista de amostras."""

    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        raw_records = json.loads(text)
    else:
        raw_records = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not isinstance(raw_records, list):
        raise ValueError("o arquivo precisa conter uma lista JSON ou linhas JSONL")
    return [SampleRecord.from_mapping(record) for record in raw_records]


def _intersection_over_union(first: Box, second: Box) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def _match_objects(
    expected: Iterable[ObjectRecord],
    predicted: Iterable[ObjectRecord],
    iou_threshold: float,
) -> tuple[list[tuple[ObjectRecord, ObjectRecord, float]], int, int]:
    expected_items = list(expected)
    predicted_items = sorted(
        predicted,
        key=lambda item: item.confidence,
        reverse=True,
    )
    used_expected: set[int] = set()
    matches: list[tuple[ObjectRecord, ObjectRecord, float]] = []
    false_positives = 0

    for prediction in predicted_items:
        candidates = [
            (index, _intersection_over_union(prediction.bbox, target.bbox))
            for index, target in enumerate(expected_items)
            if index not in used_expected
        ]
        if not candidates:
            false_positives += 1
            continue
        index, overlap = max(candidates, key=lambda item: item[1])
        if overlap < iou_threshold:
            false_positives += 1
            continue
        used_expected.add(index)
        matches.append((expected_items[index], prediction, overlap))

    false_negatives = len(expected_items) - len(matches)
    return matches, false_positives, false_negatives


def _same_value(expected: float | None, predicted: float | None) -> bool:
    if expected is None or predicted is None:
        return expected is predicted
    return math.isclose(expected, predicted, rel_tol=1e-9, abs_tol=1e-6)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _object_label(obj: ObjectRecord) -> str | None:
    if obj.ohms is not None:
        return f"ohms:{obj.ohms:g}"
    if obj.bands:
        return "bands:" + ",".join(obj.bands)
    return None


def evaluate(
    ground_truth: Iterable[SampleRecord],
    predictions: Iterable[SampleRecord],
    *,
    strategy: str | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Calcula métricas de localização, bandas, valor e vídeo."""

    expected_by_id = {sample.sample_id: sample for sample in ground_truth}
    predicted_by_id = {
        sample.sample_id: sample
        for sample in predictions
        if strategy is None or sample.strategy in (None, strategy)
    }
    sample_ids = list(expected_by_id)

    total_expected = 0
    total_predicted = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    band_slots = 0
    correct_band_slots = 0
    band_objects = 0
    complete_band_sequences = 0
    value_exact = 0
    tolerance_exact = 0
    decode_success = 0
    exact_objects = 0
    sample_exact_count = 0
    count_errors: list[float] = []
    latencies: list[float] = []
    per_sample: list[dict[str, Any]] = []
    temporal_tracks: dict[tuple[str, str], list[tuple[int, str | None]]] = defaultdict(list)

    for sample_id in sample_ids:
        expected = expected_by_id[sample_id]
        predicted = predicted_by_id.get(
            sample_id,
            SampleRecord(sample_id=sample_id, objects=()),
        )
        matches, fp, fn = _match_objects(
            expected.objects,
            predicted.objects,
            iou_threshold,
        )
        total_expected += len(expected.objects)
        total_predicted += len(predicted.objects)
        true_positive += len(matches)
        false_positive += fp
        false_negative += fn
        count_errors.append(abs(len(expected.objects) - len(predicted.objects)))
        if predicted.latency_ms is not None:
            latencies.append(predicted.latency_ms)

        sample_band_slots = 0
        sample_correct_band_slots = 0
        sample_value_exact = 0
        sample_exact_objects = 0
        for expected_obj, predicted_obj, overlap in matches:
            if expected_obj.bands:
                band_objects += 1
                slots = max(len(expected_obj.bands), len(predicted_obj.bands))
                sample_band_slots += slots
                sample_correct_band_slots += sum(
                    index < len(predicted_obj.bands)
                    and expected_obj.bands[index] == predicted_obj.bands[index]
                    for index in range(len(expected_obj.bands))
                )
                if tuple(expected_obj.bands) == tuple(predicted_obj.bands):
                    complete_band_sequences += 1

            if predicted_obj.ohms is not None:
                decode_success += 1
            if _same_value(expected_obj.ohms, predicted_obj.ohms):
                value_exact += 1
                sample_value_exact += 1
            if expected_obj.tolerance_percent == predicted_obj.tolerance_percent:
                tolerance_exact += 1

            object_is_exact = (
                _same_value(expected_obj.ohms, predicted_obj.ohms)
                and expected_obj.tolerance_percent == predicted_obj.tolerance_percent
                and (
                    not expected_obj.bands
                    or expected_obj.bands == predicted_obj.bands
                )
            )
            if object_is_exact:
                exact_objects += 1
                sample_exact_objects += 1

            if predicted_obj.track_id is not None and predicted.frame is not None:
                track_key = (predicted.source_id or predicted.sample_id, predicted_obj.track_id)
                temporal_tracks[track_key].append(
                    (predicted.frame, _object_label(predicted_obj))
                )

            _ = overlap

        band_slots += sample_band_slots
        correct_band_slots += sample_correct_band_slots
        sample_exact = (
            len(expected.objects) == len(predicted.objects)
            and sample_exact_objects == len(expected.objects)
            and len(expected.objects) > 0
        )
        if sample_exact:
            sample_exact_count += 1
        per_sample.append(
            {
                "id": sample_id,
                "expected_count": len(expected.objects),
                "predicted_count": len(predicted.objects),
                "true_positive": len(matches),
                "false_positive": fp,
                "false_negative": fn,
                "band_accuracy": _safe_ratio(
                    sample_correct_band_slots,
                    sample_band_slots,
                ),
                "value_exact": sample_value_exact == len(expected.objects)
                and len(expected.objects) > 0,
                "object_exact": sample_exact_objects == len(expected.objects)
                and len(expected.objects) > 0,
                "sample_exact": sample_exact,
            }
        )

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    track_consistency: list[float] = []
    switch_count = 0
    transition_count = 0
    for observations in temporal_tracks.values():
        observations.sort(key=lambda item: item[0])
        labels = [label for _, label in observations if label is not None]
        if len(labels) >= 2:
            counts = Counter(labels)
            track_consistency.append(max(counts.values()) / len(labels))
            transition_count += len(labels) - 1
            switch_count += sum(
                first != second for first, second in zip(labels, labels[1:])
            )

    return {
        "strategy": strategy,
        "config": {"iou_threshold": iou_threshold},
        "samples": len(sample_ids),
        "objects": {
            "expected": total_expected,
            "predicted": total_predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count_mae": statistics.mean(count_errors) if count_errors else 0.0,
        },
        "bands": {
            "slot_accuracy": _safe_ratio(correct_band_slots, band_slots),
            "sequence_exact": _safe_ratio(complete_band_sequences, band_objects),
            "slots": band_slots,
            "objects": band_objects,
        },
        "value": {
            "decode_success": _safe_ratio(decode_success, true_positive),
            "nominal_exact": _safe_ratio(value_exact, true_positive),
            "tolerance_exact": _safe_ratio(tolerance_exact, true_positive),
            "object_exact": _safe_ratio(exact_objects, total_expected),
            "sample_exact": _safe_ratio(sample_exact_count, len(sample_ids)),
        },
        "video": {
            "tracks_evaluated": len(track_consistency),
            "temporal_consistency": (
                statistics.mean(track_consistency) if track_consistency else None
            ),
            "value_switch_rate": _safe_ratio(switch_count, transition_count),
        },
        "performance": {
            "latency_ms_mean": statistics.mean(latencies) if latencies else None,
            "latency_ms_p95": _percentile(latencies, 0.95),
            "fps_estimate": (
                1000.0 / statistics.mean(latencies)
                if latencies and statistics.mean(latencies) > 0
                else None
            ),
        },
        "per_sample": per_sample,
    }


def _detection_to_record(detection: Any) -> dict[str, Any]:
    value = detection.value
    return {
        "bbox": list(detection.bbox),
        "bands": list(detection.bands),
        "ohms": value.ohms if value else None,
        "tolerance_percent": value.tolerance_percent if value else None,
        "confidence": detection.confidence,
    }


def _load_frame(sample: SampleRecord, root: Path) -> Any:
    if sample.path is None:
        raise ValueError(f"amostra {sample.sample_id!r} não possui path")
    path = Path(sample.path)
    if not path.is_absolute():
        path = root / path
    if sample.media_type != "video":
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"não foi possível abrir {path}")
        return frame

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"não foi possível abrir {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame or 0)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(
                f"não foi possível ler frame {sample.frame or 0} de {path}"
            )
        return frame
    finally:
        capture.release()


def run_baseline(
    manifest: Iterable[SampleRecord],
    *,
    strategy: str,
    root: str | Path = ".",
) -> list[SampleRecord]:
    """Executa um baseline clássico sobre as imagens do manifesto."""

    if strategy not in BASELINE_STRATEGIES:
        available = ", ".join(sorted(BASELINE_STRATEGIES))
        raise ValueError(f"estratégia {strategy!r} não disponível; use: {available}")
    band_count = BASELINE_STRATEGIES[strategy]
    root_path = Path(root)
    predictions: list[SampleRecord] = []
    for sample in manifest:
        started = time.perf_counter()
        frame = _load_frame(sample, root_path)
        classifier = classify_bgr_lab_pca if "lab-pca" in strategy else None
        if strategy.startswith("crop-"):
            height, width = frame.shape[:2]
            bands = sample_band_colors(
                frame,
                (0, 0, width, height),
                band_count=band_count,
                classifier=classifier or classify_bgr,
            )
            value = decode_color_bands(bands)

            class CropDetection:
                bbox = (0, 0, width, height)
                confidence = sum(band is not None for band in bands) / band_count

                def __init__(self) -> None:
                    self.bands = tuple(band or "unknown" for band in bands)
                    self.value = value

            detections = [CropDetection()]
        elif strategy == "segmentation-rule":
            detections = analyze_segmented_frame(
                frame,
                band_count=band_count,
                classifier=classifier or classify_bgr,
            )
        elif strategy == "hybrid-auto":
            detections = analyze_frame_auto(
                frame,
                classifier=classify_bgr_lab_pca,
            )
        elif strategy in {"hybrid-4", "hybrid-5"}:
            detections = analyze_segmented_frame(
                frame,
                band_count=band_count,
                classifier=classify_bgr_lab_pca,
            )
        else:
            detections = analyze_frame(
                frame,
                band_count=band_count,
                classifier=classifier or classify_bgr,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        predictions.append(
            SampleRecord(
                sample_id=sample.sample_id,
                source_id=sample.source_id,
                frame=sample.frame,
                objects=tuple(
                    ObjectRecord.from_mapping(_detection_to_record(detection))
                    for detection in detections
                ),
                latency_ms=latency_ms,
                strategy=strategy,
            )
        )
    return predictions


def write_jsonl(path: str | Path, records: Iterable[SampleRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "id": record.sample_id,
                        "source_id": record.source_id,
                        "frame": record.frame,
                        "latency_ms": record.latency_ms,
                        "strategy": record.strategy,
                        "objects": [
                            {
                                "bbox": list(obj.bbox),
                                "bands": list(obj.bands),
                                "ohms": obj.ohms,
                                "tolerance_percent": obj.tolerance_percent,
                                "confidence": obj.confidence,
                                "track_id": obj.track_id,
                            }
                            for obj in record.objects
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark do detector de resistores")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="executa um baseline e salva previsões")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--strategy", choices=sorted(BASELINE_STRATEGIES), required=True)
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--output", type=Path, required=True)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="avalia previsões produzidas por qualquer abordagem",
    )
    evaluate_parser.add_argument("--ground-truth", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--strategy")
    evaluate_parser.add_argument("--iou", type=float, default=0.5)
    evaluate_parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        records = load_jsonl(args.manifest)
        write_jsonl(
            args.output,
            run_baseline(records, strategy=args.strategy, root=args.root),
        )
        return

    report = evaluate(
        load_jsonl(args.ground_truth),
        load_jsonl(args.predictions),
        strategy=args.strategy,
        iou_threshold=args.iou,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
