"""Run the classical/segmentation benchmark matrix with resumable outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from resistor_detector.benchmark import (
    BASELINE_STRATEGIES,
    evaluate,
    load_jsonl,
    run_baseline,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL manifest")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(BASELINE_STRATEGIES),
        choices=list(BASELINE_STRATEGIES),
    )
    parser.add_argument("--root", default=".", help="Project/data root")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    ground_truth = load_jsonl(manifest_path)
    if args.limit is not None:
        ground_truth = ground_truth[: args.limit]

    stem = manifest_path.stem
    output_root = Path(args.output_dir)
    print(f"manifest={manifest_path} samples={len(ground_truth)}", flush=True)

    for strategy in args.strategies:
        output_dir = output_root / f"benchmark-{stem}-{strategy}"
        summary_path = output_dir / "summary.json"
        if args.resume and summary_path.exists():
            print(f"SKIP {strategy}: {summary_path}", flush=True)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"START {strategy}", flush=True)
        predictions = run_baseline(ground_truth, strategy=strategy, root=args.root)
        write_jsonl(output_dir / "predictions.jsonl", predictions)
        summary = evaluate(ground_truth, predictions, strategy=strategy)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "DONE "
            f"{strategy} f1={summary['objects']['f1']:.4f} "
            f"nominal_exact={summary['value']['nominal_exact']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
