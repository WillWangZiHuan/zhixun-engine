"""Validate the five-function continuation proposed by the decision model."""
from .program import validate


def _validate_proposal(proposal, profile=None):
    fields = {"operation", "hypothesis", "expected_result", "falsifier", "program"}
    optional = {"steady_channel_dropout", "update_mode"} if profile == "A" else {"update_mode", "batch_limit"}
    if not isinstance(proposal, dict) or set(proposal) - fields - optional or not fields <= set(proposal):
        raise ValueError("Expected operation, hypothesis, expected_result, falsifier and program")
    if proposal["operation"] != "core_code":
        raise ValueError("Expected a core_code continuation")
    for field in ("hypothesis", "expected_result", "falsifier"):
        if not isinstance(proposal[field], str) or not 1 <= len(proposal[field].strip()) <= 3000:
            raise ValueError("Expected bounded text for " + field)
    if "steady_channel_dropout" in proposal and type(proposal["steady_channel_dropout"]) is not bool:
        raise ValueError("steady_channel_dropout must be boolean")
    if profile == "A" and "update_mode" in proposal and proposal["update_mode"] not in ("reference", "adam_only"):
        raise ValueError("Choose the gas update_mode reference or adam_only")
    if "batch_limit" in proposal:
        limit = proposal["batch_limit"]
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("batch_limit must be a positive integer up to 256")
    validate(proposal["program"])
