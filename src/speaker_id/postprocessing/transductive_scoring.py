"""Leakage-controlled S014 powered-ratio alignment and evaluation helpers.

The fitting entry point accepts only probabilities and validity. It cannot
consume labels, folds, groups, audio features or embeddings.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from speaker_id.inference.batch_decoding import PoweredRatioAlignmentConfig, powered_ratio_alignment
from speaker_id.models.campp import file_sha256

KNOWN_CLASSES = 446
TOTAL_CLASSES = 447
DESIGN_UNKNOWN_PRIOR = 0.5
DESIGN_KNOWN_PRIOR = DESIGN_UNKNOWN_PRIOR / KNOWN_CLASSES
KNOWN_STRENGTH = 0.5
UNKNOWN_STRENGTH = 0.0
EPSILON = 1e-12
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_LOWER_QUANTILE = 0.025

FIXED_ALIGNMENT_CONFIG = PoweredRatioAlignmentConfig(
    schema_version=1,
    design_unknown_prior=DESIGN_UNKNOWN_PRIOR,
    design_known_prior=DESIGN_KNOWN_PRIOR,
    unknown_strength=UNKNOWN_STRENGTH,
    known_strength=KNOWN_STRENGTH,
    epsilon=EPSILON,
)
FIXED_POLICY = {
    "id": "powered_ratio_fixed_design_prior_sqrt",
    "kind": "powered_ratio_class_prior_alignment",
    "assumed_unknown_mass": DESIGN_UNKNOWN_PRIOR,
    "known_classes": KNOWN_CLASSES,
    "assumed_per_known_mass": DESIGN_KNOWN_PRIOR,
    "unknown_strength": UNKNOWN_STRENGTH,
    "known_strength": KNOWN_STRENGTH,
    "epsilon": EPSILON,
    "formula": "known_factor_c=((0.5/446)/max(observed_absolute_column_c,1e-12))**0.5;unknown_factor=1;then_row_normalize",
    "organizer_guarantees_this_prior": False,
}
PROMOTION_RULE = {
    "minimum_pooled_macro_f1_gain": 0.0015,
    "minimum_each_fold_macro_f1_gain": 0.0,
    "minimum_accuracy_gain": 0.0,
    "maximum_unknown_to_known_error_increase": 5,
    "maximum_known_to_other_known_error_increase": 2,
    "minimum_bootstrap_lower_gain": -0.001,
    "require_cpu_cuda_prediction_parity": True,
    "otherwise": "exact_historical_S008c_baseline",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    descriptor = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(descriptor + b"\0" + array.tobytes()).hexdigest()


def _strings_sha256(values):
    return hashlib.sha256(_canonical_json(list(values))).hexdigest()


def _write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False)
        handle.write("\n")


def align_probabilities(probabilities, valid):
    """Apply the one fixed S014 policy without labels or group metadata."""
    source, mask = np.asarray(probabilities), np.asarray(valid)
    _require(
        source.ndim == 2 and source.shape[1] == TOTAL_CLASSES,
        "S014 requires exactly 447 probability columns")
    _require(
        mask.dtype == np.bool_ and mask.shape == (len(source),),
        "S014 validity must be an aligned boolean vector")
    result = powered_ratio_alignment(source, mask, FIXED_ALIGNMENT_CONFIG)
    _require(
        result.unknown_factor == 1.0
        and result.design_prior[0] == DESIGN_UNKNOWN_PRIOR
        and np.all(result.design_prior[1:] == DESIGN_KNOWN_PRIOR),
        "The fixed S014 design prior or unknown factor changed")
    _require(
        np.all(result.probabilities[~mask, 0] == 1.0)
        and np.all(result.probabilities[~mask, 1:] == 0.0),
        "Invalid rows must remain exact unknown")
    arrays = {
        "base_probabilities": np.array(source, copy=True, order="C"),
        "adjusted_probabilities": np.array(
            result.probabilities, copy=True, order="C"),
        "valid": np.array(mask, copy=True, order="C"),
        "observed_prior": np.array(result.observed_prior, copy=True),
        "design_prior": np.array(result.design_prior, copy=True),
        "prior_ratios": np.array(result.prior_ratios, copy=True),
        "factors": np.array(result.factors, copy=True),
        "adjusted_prior": np.array(result.adjusted_prior, copy=True),
    }
    for array in arrays.values():
        array.flags.writeable = False
    return {
        **arrays,
        "policy": dict(FIXED_POLICY),
        "valid_rows": int(result.valid_rows),
        "invalid_rows": int((~mask).sum()),
        "unknown_factor": float(result.unknown_factor),
        "unknown_probability_may_change_via_row_normalization": bool(
            result.unknown_probability_may_change_via_row_normalization),
        "alignment_reads_truth": False,
        "alignment_reads_groups": False,
    }


def seal_fold(
        directory, outer_fold, probabilities, valid, audio_files, class_labels):
    """Compute and exclusively seal one label-free outer-fold alignment."""
    _require(
        type(outer_fold) is int and outer_fold in (0, 1),
        "S014 outer fold must be 0 or 1")
    files, labels = tuple(audio_files), tuple(class_labels)
    _require(
        len(files) == len(probabilities) and len(set(files)) == len(files)
        and all(isinstance(value, str) and value for value in files),
        "Sealed rows require unique nonempty filenames")
    _require(
        len(labels) == TOTAL_CLASSES and labels[0] == "unknown"
        and len(set(labels)) == TOTAL_CLASSES
        and all(isinstance(value, str) and value for value in labels),
        "Sealed columns require the exact 447-label order")
    result = align_probabilities(probabilities, valid)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays_path = directory / "probabilities_and_alignment.npz"
    array_keys = (
        "base_probabilities", "adjusted_probabilities", "valid",
        "observed_prior", "design_prior", "prior_ratios", "factors",
        "adjusted_prior")
    np.savez_compressed(
        arrays_path, **{key: result[key] for key in array_keys},
        audio_files=np.asarray(files), class_labels=np.asarray(labels))
    core = {
        "schema_version": 1,
        "status": "sealed_before_outer_scoring",
        "outer_fold": outer_fold,
        "rows": len(files),
        "classes": len(labels),
        "valid_rows": result["valid_rows"],
        "invalid_rows": result["invalid_rows"],
        "audio_files_sha256": _strings_sha256(files),
        "class_labels_sha256": _strings_sha256(labels),
        "policy": result["policy"],
        "alignment_reads_truth": False,
        "alignment_reads_groups": False,
        "alignment_received_row_truth": False,
        "array_hashes": {
            key: _array_sha256(result[key]) for key in array_keys},
        "arrays_file": arrays_path.name,
        "arrays_file_sha256": file_sha256(arrays_path),
        "unknown_factor_exactly_one": result["unknown_factor"] == 1.0,
        "invalid_rows_exact_unknown": True,
    }
    manifest = {
        **core,
        "seal_sha256": hashlib.sha256(_canonical_json(core)).hexdigest(),
    }
    _write_json_exclusive(directory / "seal.json", manifest)
    return manifest


def load_sealed_fold(directory, expected_manifest_sha256=None):
    """Verify every sealed byte identity and return immutable arrays."""
    directory = Path(directory)
    manifest_path = directory / "seal.json"
    arrays_path = directory / "probabilities_and_alignment.npz"
    _require(
        manifest_path.is_file() and arrays_path.is_file()
        and not manifest_path.is_symlink() and not arrays_path.is_symlink(),
        "A complete regular-file S014 fold seal is required")
    if expected_manifest_sha256 is not None:
        _require(
            file_sha256(manifest_path) == expected_manifest_sha256,
            "Fold seal manifest differs from the selection receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = manifest.pop("seal_sha256", None)
    _require(
        seal == hashlib.sha256(_canonical_json(manifest)).hexdigest(),
        "Fold seal metadata was modified")
    manifest["seal_sha256"] = seal
    _require(
        manifest.get("status") == "sealed_before_outer_scoring"
        and manifest.get("outer_fold") in (0, 1)
        and manifest.get("classes") == TOTAL_CLASSES
        and manifest.get("policy") == FIXED_POLICY
        and manifest.get("alignment_reads_truth") is False
        and manifest.get("alignment_reads_groups") is False
        and manifest.get("alignment_received_row_truth") is False,
        "Fold seal does not represent the fixed pre-scoring S014 method")
    _require(
        file_sha256(arrays_path) == manifest["arrays_file_sha256"],
        "Sealed alignment NPZ bytes changed")
    with np.load(arrays_path, allow_pickle=False) as saved:
        required = {
            "base_probabilities", "adjusted_probabilities", "valid",
            "observed_prior", "design_prior", "prior_ratios", "factors",
            "adjusted_prior", "audio_files", "class_labels"}
        _require(
            set(saved.files) == required,
            "Sealed alignment archive fields changed")
        arrays = {key: np.array(saved[key], copy=True) for key in saved.files}
    files = tuple(arrays.pop("audio_files").tolist())
    labels = tuple(arrays.pop("class_labels").tolist())
    _require(
        len(files) == manifest["rows"]
        and _strings_sha256(files) == manifest["audio_files_sha256"]
        and len(labels) == TOTAL_CLASSES and labels[0] == "unknown"
        and _strings_sha256(labels) == manifest["class_labels_sha256"],
        "Sealed row or class identity changed")
    _require(
        set(arrays) == set(manifest["array_hashes"])
        and all(
            _array_sha256(array) == manifest["array_hashes"][key]
            for key, array in arrays.items()),
        "A sealed array changed")
    recomputed = align_probabilities(
        arrays["base_probabilities"], arrays["valid"])
    _require(
        all(np.array_equal(arrays[key], recomputed[key]) for key in arrays),
        "Sealed probabilities do not reproduce from the fixed policy")
    for array in arrays.values():
        array.flags.writeable = False
    return {
        **arrays,
        "audio_files": files,
        "class_labels": labels,
        "manifest": manifest,
    }


def seal_selection(seals_root):
    """Freeze the single policy after both folds, before outer scoring."""
    seals_root = Path(seals_root)
    folds = {}
    for outer in (0, 1):
        directory = seals_root / f"fold_{outer}"
        loaded = load_sealed_fold(directory)
        manifest_path = directory / "seal.json"
        _require(
            loaded["manifest"]["outer_fold"] == outer,
            "Fold seal directory/identity mismatch")
        folds[str(outer)] = {
            "manifest": f"fold_{outer}/seal.json",
            "manifest_sha256": file_sha256(manifest_path),
            "seal_sha256": loaded["manifest"]["seal_sha256"],
            "arrays_file_sha256":
                loaded["manifest"]["arrays_file_sha256"],
            "rows": loaded["manifest"]["rows"],
        }
    core = {
        "schema_version": 1,
        "status": "fixed_selection_sealed_before_outer_scoring",
        "selected_policy": dict(FIXED_POLICY),
        "candidate_count": 1,
        "hyperparameter_selection": "none_fixed_before_execution",
        "query_expansion_executed": False,
        "alignment_received_row_truth": False,
        "folds": folds,
    }
    receipt = {
        **core,
        "selection_sha256":
            hashlib.sha256(_canonical_json(core)).hexdigest(),
    }
    _write_json_exclusive(
        seals_root / "selection_receipt.json", receipt)
    return receipt


def load_selection(seals_root):
    """Verify the selection receipt and both referenced fold seals."""
    seals_root = Path(seals_root)
    path = seals_root / "selection_receipt.json"
    _require(
        path.is_file() and not path.is_symlink(),
        "S014 selection receipt is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    digest = receipt.pop("selection_sha256", None)
    _require(
        digest == hashlib.sha256(_canonical_json(receipt)).hexdigest(),
        "S014 selection receipt was modified")
    receipt["selection_sha256"] = digest
    _require(
        receipt.get("status")
        == "fixed_selection_sealed_before_outer_scoring"
        and receipt.get("selected_policy") == FIXED_POLICY
        and receipt.get("candidate_count") == 1
        and receipt.get("query_expansion_executed") is False
        and receipt.get("alignment_received_row_truth") is False
        and set(receipt.get("folds", {})) == {"0", "1"},
        "S014 selection receipt is not the fixed pre-scoring policy")
    folds = {}
    for outer in (0, 1):
        entry = receipt["folds"][str(outer)]
        _require(
            entry["manifest"] == f"fold_{outer}/seal.json",
            "Unsafe or unexpected fold-seal path")
        loaded = load_sealed_fold(
            seals_root / f"fold_{outer}", entry["manifest_sha256"])
        _require(
            loaded["manifest"]["seal_sha256"] == entry["seal_sha256"]
            and loaded["manifest"]["arrays_file_sha256"]
            == entry["arrays_file_sha256"],
            "Fold seal differs from the frozen selection")
        folds[outer] = loaded
    return receipt, folds


def _macro_f1_indices(truth, prediction, classes=TOTAL_CLASSES):
    support = np.bincount(truth, minlength=classes)
    predicted = np.bincount(prediction, minlength=classes)
    tp = np.bincount(truth[truth == prediction], minlength=classes)
    denominator = support + predicted
    return float(np.divide(
        2.0 * tp, denominator, out=np.zeros(classes, dtype=np.float64),
        where=denominator > 0).mean())


def speaker_group_bootstrap(
        truth, baseline_prediction, adjusted_prediction, group_ids, *,
        samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
        lower_quantile=BOOTSTRAP_LOWER_QUANTILE):
    """Paired true-class-stratified, within-label group bootstrap."""
    actual, before, after, groups = [
        np.asarray(item) for item in
        (truth, baseline_prediction, adjusted_prediction, group_ids)]
    _require(
        all(item.ndim == 1 and len(item) == len(actual)
            for item in (actual, before, after, groups)) and len(actual),
        "Bootstrap inputs must be aligned nonempty vectors")
    _require(
        all(item.dtype.kind in "iu" for item in (actual, before, after))
        and all(np.all((0 <= item) & (item < TOTAL_CLASSES))
                for item in (actual, before, after)),
        "Bootstrap labels must be 447-class integer indices")
    _require(
        type(samples) is int and samples >= 100
        and type(seed) is int and seed >= 0
        and type(lower_quantile) is float and 0 < lower_quantile < .5,
        "Invalid deterministic bootstrap configuration")
    group_speakers = {}
    for index, group in enumerate(groups):
        group_speakers.setdefault(str(group), set()).add(int(actual[index]))
    cross_label_groups = {
        group for group, speakers in group_speakers.items() if len(speakers) > 1}
    cross_label_rows = sum(str(group) in cross_label_groups for group in groups)

    by_speaker, grouped_units = [], 0
    for speaker in range(TOTAL_CLASSES):
        indices = np.flatnonzero(actual == speaker)
        _require(
            len(indices), "All 447 scoring classes need bootstrap support")
        keyed = {}
        for index in indices:
            key = str(groups[index])
            _require(key, "Every bootstrap row needs a content group")
            keyed.setdefault(key, []).append(int(index))
        grouped = [np.asarray(keyed[key], dtype=np.int64) for key in sorted(keyed)]
        grouped_units += len(grouped)
        by_speaker.append(grouped)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for replicate in range(samples):
        sampled = []
        for grouped in by_speaker:
            draws = rng.integers(0, len(grouped), size=len(grouped))
            sampled.extend(grouped[int(index)] for index in draws)
        indices = np.concatenate(sampled)
        deltas[replicate] = (
            _macro_f1_indices(actual[indices], after[indices])
            - _macro_f1_indices(actual[indices], before[indices]))
    lower = float(np.quantile(deltas, lower_quantile, method="linear"))
    upper = float(np.quantile(
        deltas, 1.0 - lower_quantile, method="linear"))
    deltas.flags.writeable = False
    return {
        "deltas": deltas,
        "samples": samples,
        "seed": seed,
        "stratification":
            "true_class_then_within_label_content_group_resampled_with_replacement",
        "speaker_strata": TOTAL_CLASSES,
        "speaker_content_group_units": grouped_units,
        "cross_label_content_groups_are_resampled_within_each_true_speaker": True,
        "cross_label_content_groups": len(cross_label_groups),
        "rows_in_cross_label_content_groups": int(cross_label_rows),
        "lower_quantile": lower_quantile,
        "lower": lower,
        "upper": upper,
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "deltas_sha256": _array_sha256(deltas),
    }


def promotion_decision(
        baseline_metrics, adjusted_metrics, baseline_fold_metrics,
        adjusted_fold_metrics, bootstrap_summary,
        cpu_cuda_prediction_parity):
    """Apply every preregistered S014 promotion gate."""
    _require(
        len(baseline_fold_metrics) == len(adjusted_fold_metrics) == 2,
        "Promotion requires both fixed outer folds")
    _require(
        type(cpu_cuda_prediction_parity) is bool,
        "CPU/CUDA prediction parity must be an explicit boolean")
    pooled_gain = adjusted_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    accuracy_gain = adjusted_metrics["accuracy"] - baseline_metrics["accuracy"]
    fold_gains = [
        adjusted_fold_metrics[index]["macro_f1"]
        - baseline_fold_metrics[index]["macro_f1"]
        for index in range(2)]
    unknown_increase = (
        adjusted_metrics["errors"]["unknown_to_known"]
        - baseline_metrics["errors"]["unknown_to_known"])
    other_increase = (
        adjusted_metrics["errors"]["known_to_other_known"]
        - baseline_metrics["errors"]["known_to_other_known"])
    gates = {
        "pooled_macro_f1_gain":
            pooled_gain >=
            PROMOTION_RULE["minimum_pooled_macro_f1_gain"],
        "both_fold_macro_f1_nonnegative":
            min(fold_gains) >=
            PROMOTION_RULE["minimum_each_fold_macro_f1_gain"],
        "accuracy_nonnegative":
            accuracy_gain >= PROMOTION_RULE["minimum_accuracy_gain"],
        "unknown_to_known_increase_bounded":
            unknown_increase <=
            PROMOTION_RULE["maximum_unknown_to_known_error_increase"],
        "known_to_other_known_increase_bounded":
            other_increase <=
            PROMOTION_RULE["maximum_known_to_other_known_error_increase"],
        "bootstrap_lower_bounded":
            bootstrap_summary["lower"] >=
            PROMOTION_RULE["minimum_bootstrap_lower_gain"],
        "cpu_cuda_prediction_parity": (
            cpu_cuda_prediction_parity
            is PROMOTION_RULE["require_cpu_cuda_prediction_parity"]),
    }
    promoted = all(gates.values())
    return {
        "promoted": promoted,
        "selected_for_future_packaging":
            FIXED_POLICY["id"] if promoted else PROMOTION_RULE["otherwise"],
        "gates": gates,
        "rule": dict(PROMOTION_RULE),
        "observed": {
            "pooled_macro_f1_gain": pooled_gain,
            "fold_macro_f1_gains": fold_gains,
            "accuracy_gain": accuracy_gain,
            "unknown_to_known_error_increase": int(unknown_increase),
            "known_to_other_known_error_increase": int(other_increase),
            "bootstrap_lower_gain": bootstrap_summary["lower"],
        },
        "statistical_significance_claimed": False,
    }
