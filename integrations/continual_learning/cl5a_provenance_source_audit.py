"""CL-5A static generator/protocol provenance audit; never generates samples."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from integrations.continual_learning.cl3y_readiness_audit import scientific_readiness_audit

ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT / "cl2_shared_rule_candidate.py"
DATA_CANDIDATE = ROOT / "cl2_data_candidate.py"
CL3X = ROOT / "cl3x_h1_preregistration.json"
CL2_PLAN = ROOT / "cl2_paired_run_plan_candidate.json"


def _literal_assignment(module: ast.Module, name: str):
    matches = [node.value for node in module.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(matches) != 1:
        raise ValueError(f"missing or duplicated source constant {name}")
    return ast.literal_eval(matches[0])


def static_provenance_audit() -> dict:
    """Fingerprint source and verify unapproved protocol invariants; no imports of generators."""
    source_bytes = GENERATOR.read_bytes()
    source = ast.parse(source_bytes, filename=str(GENERATOR))
    data_source = ast.parse(DATA_CANDIDATE.read_bytes(), filename=str(DATA_CANDIDATE))
    cl3x = json.loads(CL3X.read_text(encoding="utf-8"))
    cl2 = json.loads(CL2_PLAN.read_text(encoding="utf-8"))
    audit = scientific_readiness_audit()
    required_functions = {"generate_shared_rule_candidate", "global_label", "shared_rule_dataset_sha256"}
    names = {node.name for node in source.body if isinstance(node, ast.FunctionDef)}
    if not required_functions.issubset(names):
        raise ValueError("shared-rule generator functions changed")
    label_functions = [node for node in source.body
                       if isinstance(node, ast.FunctionDef) and node.name == "global_label"]
    if len(label_functions) != 1:
        raise ValueError("global label function missing or ambiguous")
    returns = [node for node in ast.walk(label_functions[0]) if isinstance(node, ast.Return)]
    expected_label = "features[GLOBAL_LABEL_PAIR[0]] ^ features[GLOBAL_LABEL_PAIR[1]]"
    if len(returns) != 1 or ast.unparse(returns[0].value) != expected_label:
        raise ValueError("global label XOR operation or operands changed")
    version = _literal_assignment(source, "VERSION")
    label_pair = _literal_assignment(source, "GLOBAL_LABEL_PAIR")
    activity = _literal_assignment(data_source, "ACTIVITY_RATES")
    if version != "cl2-shared-rule-candidate-v0" or label_pair != (0, 1):
        raise ValueError("generator version/label definition changed")
    expected_activity = {"T1": 0.125, "T2": 0.25, "T3": 0.375, "T4": 0.5}
    if type(activity) is not dict or list(activity) != list(expected_activity) or activity != expected_activity:
        raise ValueError("task activity rates or order changed")
    if (
        cl3x["decision"] != "NO_GO"
        or cl3x["study_design"]["dataset"] != version
        or cl3x["study_design"]["dataset_source_sha256"] is not None
        or cl3x["study_design"]["dataset_digests_pinned"] is not False
        or cl3x["study_design"]["paired_seeds"] != [101, 102, 103, 104, 105]
        or cl3x["study_design"]["train_per_task"] != 128
        or cl3x["study_design"]["test_per_task"] != 64
        or cl2["dataset"]["candidate"] != version
        or cl2["dataset"]["source_sha256"] is not None
        or cl2["dataset"]["split_policy"] != "globally_disjoint_within_seed"
        or audit["decision"] != "NO_GO"
        or audit["open_count"] != 9
    ):
        raise ValueError("CL-5A cannot silently approve or drift from current protocols")
    return {
        "kind": "CL5A_STATIC_SOURCE_PROVENANCE_ONLY",
        "decision": "NO_GO",
        "generator_path": GENERATOR.name,
        "generator_source_sha256_observed": hashlib.sha256(source_bytes).hexdigest(),
        "related_data_candidate_source_sha256_observed": hashlib.sha256(DATA_CANDIDATE.read_bytes()).hexdigest(),
        "generator_version": version,
        "global_label_pair": list(label_pair),
        "task_order": list(activity),
        "planned_seeds": cl3x["study_design"]["paired_seeds"],
        "dataset_digests_pinned": False,
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "samples_examined": 0,
        "scientific_source_pin_approved": False,
        "energy_measured": False,
        "open_scientific_gates": 9,
    }
