#!/usr/bin/env python3

"""
Lightweight helpers for routing planner decisions based on world model
confidence and belief divergence.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class BeliefMetrics:
    avg_concept_confidence: float = 1.0
    belief_divergence: float = 0.0
    divergence_metrics: Optional[Dict[str, float]] = None
    note: str = ""

    def get_divergence(self, metric_type: str) -> float:
        """Return the divergence score requested by the planner.

        Fallback to the scalar ``belief_divergence`` cached on the planner
        when the requested metric is unavailable. This keeps the hook robust
        when only coarse divergence is recorded.
        """

        if metric_type == "belief_divergence":
            return self.belief_divergence
        if self.divergence_metrics and metric_type in self.divergence_metrics:
            return float(self.divergence_metrics[metric_type])
        return self.belief_divergence


def choose_belief_action(decision_conf, metrics: BeliefMetrics) -> Tuple[Optional[str], str]:
    """Return the tool name and a short reason when a hook should run.

    The decision hierarchy is:
    1) If divergence is above the correction threshold, prefer correction.
    2) If average concept confidence is below the configured threshold, add
       more observations.
    3) If divergence is above the warning threshold, ask the human for help.
    """

    if decision_conf is None:
        return None, ""

    if not decision_conf.get("cbwm_enabled", True):
        return None, "CBWM hooks disabled."

    divergence_metric_type = decision_conf.get("divergence_metric_type", "belief_divergence")
    active_divergence = metrics.get_divergence(divergence_metric_type)

    div_threshold = decision_conf.get("divergence_threshold", 0.3)
    correction_threshold = decision_conf.get("correction_divergence_threshold", div_threshold * 1.5)
    confidence_threshold = decision_conf.get("concept_confidence_threshold", 0.5)
    l2d_threshold = decision_conf.get("l2d_divergence_threshold", div_threshold * 0.5)
    l2d_enabled = decision_conf.get("l2d_action_enabled", False)

    if active_divergence >= correction_threshold:
        action = decision_conf.get("correction_action", "CorrectHuman")
        reason = (
            f"Belief divergence ({divergence_metric_type}) {active_divergence:.2f} exceeds correction "
            f"threshold {correction_threshold:.2f}."
        )
        return action, reason

    if metrics.avg_concept_confidence < confidence_threshold:
        action = decision_conf.get("low_confidence_action", "AppendObservation")
        reason = (
            f"Average concept confidence {metrics.avg_concept_confidence:.2f} is below "
            f"threshold {confidence_threshold:.2f}."
        )
        return action, reason

    if l2d_enabled and active_divergence >= l2d_threshold:
        action = decision_conf.get("l2d_action", "LookToDisambiguate")
        reason = (
            f"Divergence ({divergence_metric_type}) {active_divergence:.2f} exceeds "
            f"L2D threshold {l2d_threshold:.2f}."
        )
        return action, reason

    if active_divergence >= div_threshold:
        action = decision_conf.get("high_divergence_action", "AskHuman")
        reason = (
            f"Belief divergence ({divergence_metric_type}) {active_divergence:.2f} exceeds "
            f"threshold {div_threshold:.2f}."
        )
        return action, reason

    return None, metrics.note
