from __future__ import annotations

import json
import sys
from pathlib import Path


for argument in sys.argv[1:]:
    report = json.loads(Path(argument).read_text(encoding="utf-8"))
    objects = report["objects"]
    bands = report["bands"]
    value = report["value"]
    performance = report["performance"]
    print(
        json.dumps(
            {
                "strategy": report.get("strategy"),
                "samples": report["samples"],
                "detection_f1": objects["f1"],
                "precision": objects["precision"],
                "recall": objects["recall"],
                "band_slot_accuracy": bands["slot_accuracy"],
                "band_sequence_exact": bands["sequence_exact"],
                "nominal_exact": value["nominal_exact"],
                "sample_exact": value["sample_exact"],
                "latency_ms_mean": performance["latency_ms_mean"],
                "latency_ms_p95": performance["latency_ms_p95"],
                "fps_estimate": performance["fps_estimate"],
            },
            ensure_ascii=False,
        )
    )
