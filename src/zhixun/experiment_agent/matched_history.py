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
    functions = []
    experiments = []
    same_program = []
    same_artifact = []
    parent = None
    metrics = ("rmse", "mae", "short_life_rmse")
    for item in history["experiments"]:
        if parent is None:
            parent = item["before"]
        if parent != item["before"]:
            raise ValueError("A matched history contains two different parent measurements")
        if any(item["after"][k] != parent[k] for k in ("short_life_rows", "short_life_cutoff_minutes")):
            raise ValueError("A matched history contains incompatible short-life measurements")
        used = {}
        for name, body in item["functions"].items():
            if body not in functions:
                functions.append(body)
            used[name] = functions.index(body)
        choices = [[c["round"], c["selected"]] for c in item["choices"]]
        experiments.append({"functions":used, "executed_plan":item["executed_plan"],
                            "errors":[item["after"][k] for k in metrics], "choices":choices})
        if item["program_sha"] == (current_report or {}).get("program_sha256"):
            same_program.extend(c[0] for c in choices)
        if item["candidate_sha"] == (current_report or {}).get("checkpoint_sha256"):
            same_artifact.extend(c[0] for c in choices)
    result = {"parent_measurement":parent,
              "error_columns":list(metrics),"function_bodies":functions,"experiments":experiments}
    if current_report is None:
        result["use"] = "Compare the previously executed numerical functions and measurements under this same parent and data. Propose a materially different numerical experiment or give a concrete reason in the existing reflection for a repeat. A changed parent, stage, train or validation set needs fresh measurements."
    else:
        result["use"] = "Compare this measured tradeoff with prior choices under the matching conditions. State why the evidence supports retaining or changing the choice, especially if this exact program and candidate were previously measured. Choose the artifact yourself according to the task contract."
        result.update(previous_rounds_with_identical_program=same_program,
                      previous_rounds_with_identical_candidate=same_artifact)
    return result
