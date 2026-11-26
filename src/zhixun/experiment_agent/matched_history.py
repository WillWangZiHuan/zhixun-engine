"""B experiment history for the same stage, parent and observed input files."""
import ast
from copy import deepcopy
from hashlib import sha256


def open_history(previous, dataset, stage, hashes):
    condition = {"profile": dataset, "stage": stage, "parent": hashes["parent"],
                 "train": hashes["train"], "validation": hashes["validation"]}
    if previous and previous["condition"] == condition:
        return deepcopy(previous)
    return {"condition": condition, "experiments": []}


def append_result(history, round_index, program, report, role, reason):
    program_hash = sha256(program.encode()).hexdigest()
    if report["program_sha256"] != program_hash or any(
        history["condition"][key] != report["source_input_hashes"][key] for key in ("parent","train","validation")
    ):
        raise ValueError("Program, parent or measured inputs differ from the history condition")
    key = (program_hash, report["checkpoint_sha256"])
    for experiment in history["experiments"]:
        if (experiment["program_sha"], experiment["candidate_sha"]) == key:
            break
    else:
        functions = {f.name: "\n".join(ast.unparse(s) for s in f.body)
                     for f in ast.parse(program).body if isinstance(f, ast.FunctionDef)}
        experiment = {"program_sha":program_hash, "executed_plan":report["numerical"]["plan"],
                      "functions":functions, "candidate_sha":report["checkpoint_sha256"],
                      "before":report["local_validation_before"], "after":report["local_validation_after"],
                      "choices":[]}
        history["experiments"].append(experiment)
    experiment["choices"].append({"round":round_index, "selected":role, "reason":reason})


def prompt_history(history, current_report=None):
    result = deepcopy(history)
    if current_report is None:
        result["use"] = "Compare the previously executed numerical functions and measurements under this same parent and data. Propose a materially different numerical experiment or give a concrete reason in the existing reflection for a repeat. A changed parent, stage, train or validation set needs fresh measurements."
    else:
        result["current_program_sha"] = current_report["program_sha256"]
        result["current_candidate_sha"] = current_report["checkpoint_sha256"]
        result["use"] = "Compare this measured tradeoff with prior choices under the matching conditions. State why the evidence supports retaining or changing the choice, especially if this exact program and candidate were previously measured. Choose the artifact yourself according to the task contract."
    return result
