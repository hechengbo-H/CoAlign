"""Utility helpers to map detection outputs to concept-space annotations."""

from typing import List, Optional, Tuple

from habitat_llm.sims.metadata_interface import MetadataInterface


def map_detection_to_concepts(
    detected_type: Optional[str],
    metadata: Optional[MetadataInterface] = None,
) -> Tuple[List[str], List[float]]:
    """Map a detected semantic type to concept labels and confidences.

    This lightweight mapper currently mirrors the detected semantic category
    into the concept space with a high-confidence label while providing a
    best-effort fallback for unknown classes.
    """

    if detected_type is None:
        return [], []

    labels: List[str] = [detected_type]
    confidences: List[float] = [1.0]

    if metadata is not None and detected_type not in metadata.lexicon:
        labels.append("unknown")
        confidences.append(0.1)

    return labels, confidences
