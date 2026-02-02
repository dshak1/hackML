from __future__ import annotations

import json
import os
from typing import Dict, List


def _format_examples(examples: List[Dict[str, object]]) -> str:
    if not examples:
        return "None"
    return json.dumps(examples, indent=2, sort_keys=True)


def _summarize_checks(results: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    checks = results.get("checks", [])
    failed = [c for c in checks if c.get("status") == "fail"]
    warned = [c for c in checks if c.get("status") == "warn"]
    passed = [c for c in checks if c.get("status") == "pass"]
    return {"failed": failed, "warned": warned, "passed": passed}


def _build_validation_plan(results: Dict[str, object]) -> List[str]:
    checks = results.get("checks", [])
    lookup = {c["name"]: c for c in checks}
    plan = []
    schema = results.get("schema", {})
    exclude_fields = schema.get("exclude_fields", [])

    for key in ["train_missing_values", "test_missing_values"]:
        if key in lookup:
            plan.append(
                f"Missing/NaN checks ({key}: {lookup[key].get('count', 0)} missing)."
            )

    for key in ["train_inf_values", "test_inf_values"]:
        if key in lookup:
            plan.append(
                f"Inf checks ({key}: {lookup[key].get('count', 0)} inf values)."
            )

    for key in ["train_duplicate_rows", "train_duplicate_ids"]:
        if key in lookup:
            plan.append(
                f"Duplicate checks ({key}: {lookup[key].get('count', 0)} issues)."
            )

    for key in ["type_invalid_categories", "amount_negative", "amount_outliers"]:
        if key in lookup:
            plan.append(
                f"Domain checks ({key}: {lookup[key].get('count', 0)} flagged)."
            )

    for key in ["balance_error_orig", "balance_error_dest"]:
        if key in lookup:
            details = lookup[key].get("details", {})
            sample_size = details.get("sample_size", "unknown")
            plan.append(
                f"Cross-field consistency ({key}: sample_size={sample_size})."
            )
        elif any(field in exclude_fields for field in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]):
            plan.append("Cross-field consistency (balance checks skipped - fields excluded from model).")

    plan.append("Drift checks: numeric KS-stat + categorical PSI.")
    plan.append("Target class distribution report for imbalance tracking.")

    return plan


def _model_impact_actions(results: Dict[str, object]) -> List[Dict[str, str]]:
    actions = []
    checks = {c["name"]: c for c in results.get("checks", [])}
    class_dist = results.get("class_distribution", {}).get("counts", {})
    drift = results.get("drift", {})
    schema = results.get("schema", {})
    exclude_fields = schema.get("exclude_fields", [])

    if "amount_spikes" in checks:
        spike_values = checks["amount_spikes"].get("details", {}).get("spike_values", [])
        actions.append(
            {
                "issue": "Amount spikes can bias decision boundaries.",
                "impact": "Macro F1 can suffer if rare classes correlate with repeated amounts.",
                "action": f"Add `amount_spike_flag` for values: {spike_values[:5]}.",
            }
        )

    if "amount_outliers" in checks:
        threshold = checks["amount_outliers"].get("details", {}).get("threshold", None)
        actions.append(
            {
                "issue": "Extreme amount outliers compress model scale.",
                "impact": "Minority classes with large amounts can be underfit.",
                "action": f"Use `log1p_amount` and `amount_outlier_flag` (threshold={threshold}).",
            }
        )

    for key in ["balance_error_orig", "balance_error_dest"]:
        if key in checks and not any(field in exclude_fields for field in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]):
            count = checks[key].get("count", 0)
            actions.append(
                {
                    "issue": f"{key} shows balance inconsistencies (sample count={count}).",
                    "impact": "Balance errors are strong fraud signals but can add noise.",
                    "action": "Use `balance_error_*` features and threshold flags.",
                }
            )

    if "train_nameDest_pattern" in checks or "train_nameOrig_pattern" in checks:
        actions.append(
            {
                "issue": "Name prefixes encode account types (customer vs merchant).",
                "impact": "Ignoring prefixes can hide high-risk merchant patterns.",
                "action": "Add `is_merchant_dest` and prefix features.",
            }
        )

    if class_dist:
        actions.append(
            {
                "issue": "Target class imbalance is severe.",
                "impact": "Macro F1 will drop if rare classes are ignored.",
                "action": "Use class weights or focal loss; stratified CV by `urgency_level`.",
            }
        )

    if drift.get("top_numeric"):
        top_num = drift["top_numeric"][0]
        actions.append(
            {
                "issue": f"Numeric drift in {top_num['column']} (KS={top_num['ks_stat']:.3f}).",
                "impact": "Train/test mismatch can lower recall on rare classes.",
                "action": "Consider robust scaling or monotonic binning for drifting features.",
            }
        )

    if drift.get("top_categorical"):
        top_cat = drift["top_categorical"][0]
        actions.append(
            {
                "issue": f"Categorical drift in {top_cat['column']} (PSI={top_cat['psi']:.3f}).",
                "impact": "Category frequency shifts can bias calibration.",
                "action": "Use target encoding with smoothing or frequency-aware encoding.",
            }
        )

    if (checks.get("amount_negative", {}).get("count") or 0) > 0:
        actions.append(
            {
                "issue": "Negative amounts indicate data errors or reversals.",
                "impact": "Model may misinterpret sign flips, hurting recall.",
                "action": "Flag negative amounts and consider absolute value features.",
            }
        )

    if len(actions) < 5:
        actions.append(
            {
                "issue": "Transaction type is a strong driver of fraud risk.",
                "impact": "Poor encoding can underfit rare transaction types.",
                "action": "Use one-hot encoding for `type` and interaction with amount.",
            }
        )

    return actions[:10]


def write_report(results: Dict[str, object], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    summary = _summarize_checks(results)
    plan = _build_validation_plan(results)
    actions = _model_impact_actions(results)

    md_path = os.path.join(out_dir, "validation_report.md")
    json_path = os.path.join(out_dir, "validation_report.json")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Data Validation Report\n\n")
        f.write("## Dataset Summary\n")
        f.write(f"- Train shape: {results['train']['shape']}\n")
        f.write(f"- Test shape: {results['test']['shape']}\n")
        f.write(f"- Train columns: {results['train']['columns']}\n")
        f.write(f"- Test columns: {results['test']['columns']}\n\n")

        f.write("## Inferred Schema\n")
        schema = results.get("schema", {})
        f.write(f"- Allowed types: {schema.get('allowed_type_values')}\n")
        f.write(f"- Target values: {schema.get('target_values')}\n")
        exclude_fields = schema.get('exclude_fields', [])
        if exclude_fields:
            f.write(f"- Excluded fields (not used in model): {exclude_fields}\n")
        thresholds = schema.get("thresholds", {})
        f.write(f"- Thresholds: {thresholds}\n\n")

        f.write("## Validation Plan (Evidence-Based)\n")
        for item in plan:
            f.write(f"- {item}\n")
        f.write("\n")

        f.write("## Failed Checks\n")
        if summary["failed"]:
            for check in summary["failed"]:
                f.write(
                    f"- {check['name']} (count={check.get('count', 0)}, severity={check.get('severity')})\n"
                )
                if check.get("examples"):
                    f.write("```\n")
                    f.write(_format_examples(check["examples"]))
                    f.write("\n```\n")
        else:
            f.write("- None\n")
        f.write("\n")

        f.write("## Warnings\n")
        if summary["warned"]:
            for check in summary["warned"]:
                f.write(
                    f"- {check['name']} (count={check.get('count', 0)}, severity={check.get('severity')})\n"
                )
                if check.get("examples"):
                    f.write("```\n")
                    f.write(_format_examples(check["examples"]))
                    f.write("\n```\n")
        else:
            f.write("- None\n")
        f.write("\n")

        f.write("## Drift Summary\n")
        drift = results.get("drift", {})
        f.write(f"- Top numeric drift: {drift.get('top_numeric', [])}\n")
        f.write(f"- Top categorical drift: {drift.get('top_categorical', [])}\n\n")

        f.write("## Class Distribution (Train)\n")
        f.write(f"- Counts: {results.get('class_distribution', {}).get('counts', {})}\n")
        f.write(
            f"- Percentages: {results.get('class_distribution', {}).get('percentages', {})}\n\n"
        )

        f.write("## Model Impact\n")
        for action in actions:
            f.write(
                f"- Issue: {action['issue']} Impact: {action['impact']} Action: {action['action']}\n"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        payload = {
            "results": results,
            "summary": summary,
            "validation_plan": plan,
            "model_impact": actions,
        }
        json.dump(payload, f, indent=2)

    return {"md": md_path, "json": json_path}
