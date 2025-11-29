#!/usr/bin/env python3
"""Plot concept confidence and belief divergence over time.

The utility expects a log file or directory containing JSON/JSONL records.
Each record may embed metrics directly or inside a ``metrics`` field. The
script is tolerant to minimal inputs: if only concept confidence dictionaries
are found, it plots their average; if multiple divergence metrics exist, you
can pick one via ``--divergence-key``.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


def _iter_json_records(paths: Iterable[Path]) -> Iterable[Dict]:
    """Yield JSON objects from a collection of files."""

    for path in paths:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Try loading the whole file if line parsing fails
                    try:
                        f.seek(0)
                        payload = json.load(f)
                        if isinstance(payload, list):
                            for item in payload:
                                if isinstance(item, dict):
                                    yield item
                        elif isinstance(payload, dict):
                            yield payload
                    except json.JSONDecodeError:
                        continue
                    break


def _collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files: List[Path] = []
    for ext in ("*.json", "*.jsonl", "*.log"):
        files.extend(sorted(input_path.glob(ext)))
    return files


def _extract_metric(record: Dict, key: str) -> Optional[float]:
    """Find a metric either at the top level or under ``metrics``."""

    if key in record and isinstance(record[key], (int, float)):
        return float(record[key])
    metrics = record.get("metrics") or {}
    if isinstance(metrics, dict) and key in metrics and isinstance(metrics[key], (int, float)):
        return float(metrics[key])
    return None


def _average_concept_confidence(record: Dict) -> Optional[float]:
    """Compute the mean confidence when a dict of concept scores is present."""

    concepts = record.get("concept_confidence")
    if isinstance(concepts, dict) and concepts:
        return float(sum(concepts.values()) / len(concepts))
    return _extract_metric(record, "avg_concept_confidence")


def _prepare_series(records: List[Dict], divergence_key: str) -> Tuple[List[int], List[float], List[float]]:
    steps: List[int] = []
    confidences: List[float] = []
    divergences: List[float] = []

    for idx, record in enumerate(records):
        step = record.get("step") or record.get("timestep") or record.get("episode_step")
        steps.append(int(step) if step is not None else idx)

        confidence = _average_concept_confidence(record)
        divergence = _extract_metric(record, divergence_key)

        # Fallback to aggregated divergence dicts
        if divergence is None:
            divergence_dict = record.get("divergence") or record.get("belief_divergence")
            if isinstance(divergence_dict, dict) and divergence_key in divergence_dict:
                divergence = float(divergence_dict[divergence_key])

        confidences.append(confidence if confidence is not None else 1.0)
        divergences.append(divergence if divergence is not None else 0.0)

    return steps, confidences, divergences


def plot_metrics(
    records: List[Dict],
    divergence_key: str,
    output_path: Path,
    title: str = "Belief divergence over time",
) -> None:
    steps, confidences, divergences = _prepare_series(records, divergence_key)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, confidences, label="Avg concept confidence", color="#1f77b4")
    axes[0].set_ylabel("Confidence")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, divergences, label=f"Divergence: {divergence_key}", color="#d62728")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Divergence")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize belief divergence and concept confidence.")
    parser.add_argument("input", type=Path, help="Path to a log file or directory with JSON/JSONL records.")
    parser.add_argument(
        "--divergence-key",
        default="belief_divergence",
        help="Metric key to visualize (e.g., belief_divergence, concept_js_divergence).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/belief_divergence.png"),
        help="Where to write the resulting plot.",
    )
    parser.add_argument(
        "--title",
        default="Belief divergence over time",
        help="Custom title for the plot.",
    )

    args = parser.parse_args()

    files = _collect_files(args.input)
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL logs found under {args.input}")

    records = list(_iter_json_records(files))
    if not records:
        raise ValueError(f"No readable JSON records found in {len(files)} files")

    plot_metrics(records, args.divergence_key, args.output, title=args.title)


if __name__ == "__main__":
    main()
