"""Validate the five-function continuation proposed by the decision model."""
from .program import validate


def _validate_proposal(proposal, profile=None):
    fields = {"operation", "hypothesis", "expected_result", "falsifier", "program"}
    if not isinstance(proposal, dict) or set(proposal) - fields - ({"steady_channel_dropout"} if profile == "A" else set()) or not fields <= set(proposal):
        raise ValueError("Expected operation, hypothesis, expected_result, falsifier and program")
    if proposal["operation"] != "core_code":
        raise ValueError("Expected a core_code continuation")
    for field in ("hypothesis", "expected_result", "falsifier"):
        if not isinstance(proposal[field], str) or not 1 <= len(proposal[field].strip()) <= 3000:
            raise ValueError("Expected bounded text for " + field)
    if "steady_channel_dropout" in proposal and type(proposal["steady_channel_dropout"]) is not bool:
        raise ValueError("steady_channel_dropout must be boolean")
    validate(proposal["program"])
