"""Build provisional file-group folds without separating verified duplicates.

These folds establish content-group separation, not independent recording sessions
or unknown speaker identities. They must be revised when new grouping evidence is
available. Invalid and conflicting-label files remain in evaluation accounting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


class Groups:
    def __init__(self, names: list[str]):
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        self.parent[max(a, b)] = min(a, b)


def construct_folds(
    rows: list[dict], verified_pairs: list[tuple[str, str]], requested_folds: int = 5,
    seed: int = 20260907,
) -> tuple[list[dict], dict]:
    if requested_folds < 2:
        raise ValueError("At least two validation folds are required")
    rows = sorted(rows, key=lambda row: row["audio_file"])
    names = [row["audio_file"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Manifest must contain exactly one row per audio file")
    known = sorted({r["speaker_id"] for r in rows} - {"unknown"})
    groups = Groups(names)
    for field in ("input_sha256", "pcm_sha256"):
        first_by_hash: dict[str, str] = {}
        for row in rows:
            digest = row.get(field)
            if digest:
                first = first_by_hash.setdefault(str(digest), row["audio_file"])
                groups.union(first, row["audio_file"])
    for first, second in verified_pairs:
        if first not in groups.parent or second not in groups.parent:
            raise ValueError("Duplicate pair references a file outside the manifest")
        groups.union(first, second)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[groups.find(row["audio_file"])].append(row)
    info = []
    for members in grouped.values():
        labels = {row["speaker_id"] for row in members}
        conflict = len(labels) > 1
        eligible = [r for r in members if truth(r.get("usable_for_training")) and not conflict]
        group_id = "g_" + hashlib.sha256("\n".join(r["audio_file"] for r in members).encode()).hexdigest()[:20]
        info.append({"id": group_id, "members": members, "labels": labels, "conflict": conflict,
                     "eligible": eligible, "duration": sum(float(r.get("duration_seconds") or 0) for r in members)})
    support = Counter()
    for group in info:
        for label in {r["speaker_id"] for r in group["eligible"]}:
            support[label] += 1
    minimum = min((support[label] for label in known), default=0)
    n_folds = min(requested_folds, minimum)
    base_summary = {
        "status": "provisional_content_group_folds" if n_folds >= 2 else "infeasible",
        "requested_folds": requested_folds, "actual_folds": n_folds if n_folds >= 2 else 0,
        "seed": seed, "known_class_count": len(known), "file_count": len(rows),
        "group_count": len(info), "verified_pair_count": len(verified_pairs),
        "cross_label_group_count": sum(g["conflict"] for g in info),
        "cross_label_file_count": sum(len(g["members"]) for g in info if g["conflict"]),
        "eligible_groups_per_known_class": {label: support[label] for label in known},
        "minimum_eligible_known_groups": minimum,
        "scope": "File/content groups using exact hashes and verified acoustic duplicates or overlaps",
        "caveats": [
            "Independent recording sessions are unverified; no source/session metadata is available",
            "Unknown-person identities are not provided; unknown-speaker separation is not established",
            "Unverified duplicate candidates are not automatically merged",
            "These folds are provisional pending acoustic/manual/embedding review",
            "Do not tune rejection thresholds and report their optimum on the same validation predictions",
            "VAD and quality diagnostics on all data are descriptive; learned transformations fit training only",
        ],
        "training_exclusion_policy": "Unusable signal or confirmed duplicate/overlap component with conflicting labels",
        "evaluation_policy": "Every original file is assigned once and remains scoreable, including invalid signal and conflicts",
    }
    if n_folds < 2:
        base_summary["unsupported_known_classes"] = [label for label in known if support[label] < 2]
        base_summary["known_classes_without_eligible_signal"] = [label for label in known if support[label] == 0]
        base_summary["training_excluded_files"] = sum(len(g["members"]) - len(g["eligible"]) for g in info)
        return [], base_summary

    rng = random.Random(seed)
    durations = [0.0] * n_folds
    counts = [0] * n_folds
    assignments = {}
    label_groups = defaultdict(list)
    remainder = []
    for group in info:
        if len(group["labels"]) == 1 and group["eligible"]:
            label_groups[next(iter(group["labels"]))].append(group)
        else:
            remainder.append(group)
    for label in known + ["unknown"]:
        selected = list(label_groups[label])
        rng.shuffle(selected)
        selected.sort(key=lambda g: -g["duration"])
        per_label = [0] * n_folds
        label_duration = [0.0] * n_folds
        for group in selected:
            fold_order = list(range(n_folds))
            rng.shuffle(fold_order)
            fold = min(fold_order, key=lambda f: (per_label[f], label_duration[f], durations[f], counts[f]))
            assignments[group["id"]] = fold
            per_label[fold] += 1
            label_duration[fold] += group["duration"]
            durations[fold] += group["duration"]
            counts[fold] += len(group["members"])
    rng.shuffle(remainder)
    for group in sorted(remainder, key=lambda g: -len(g["members"])):
        fold = min(range(n_folds), key=lambda f: (counts[f], durations[f]))
        assignments[group["id"]] = fold
        counts[fold] += len(group["members"])
        durations[fold] += group["duration"]
    output = []
    for group in info:
        for row in group["members"]:
            eligible = truth(row.get("usable_for_training")) and not group["conflict"]
            reasons = []
            if not truth(row.get("usable_for_training")):
                reasons.append("unusable_signal")
            if group["conflict"]:
                reasons.append("cross_label_content_conflict")
            output.append({"audio_file": row["audio_file"], "speaker_id": row["speaker_id"],
                           "group_id": group["id"], "fold": assignments[group["id"]],
                           "train_eligible": eligible, "evaluation_included": True,
                           "exclusion_reasons": "|".join(reasons),
                           "duration_seconds": row.get("duration_seconds", "")})
    output.sort(key=lambda r: r["audio_file"])
    fold_summary = []
    for fold in range(n_folds):
        evaluation = [r for r in output if r["fold"] == fold]
        training = [r for r in output if r["fold"] != fold and r["train_eligible"]]
        train_labels = {r["speaker_id"] for r in training}
        val_labels = {r["speaker_id"] for r in evaluation}
        missing_train = sorted(set(known) - train_labels)
        missing_val = sorted(set(known) - val_labels)
        if missing_train or missing_val:
            raise AssertionError("Fold assignment failed known-label coverage")
        fold_summary.append({"fold": fold, "validation_files": len(evaluation),
                             "training_eligible_files": len(training),
                             "validation_unknown_files": sum(r["speaker_id"] == "unknown" for r in evaluation),
                             "validation_invalid_or_conflicting_files": sum(not r["train_eligible"] for r in evaluation),
                             "validation_known_classes": len(set(known) & val_labels),
                             "training_known_classes": len(set(known) & train_labels)})
    base_summary["folds"] = fold_summary
    base_summary["all_original_files_assigned_once"] = len(output) == len(rows)
    base_summary["training_excluded_files"] = sum(not r["train_eligible"] for r in output)
    return output, base_summary


def write_splits(manifest: Path, verified_pairs: list[tuple[str, str]], output_dir: Path,
                 summary_path: Path, requested_folds: int = 5, seed: int = 20260907) -> dict:
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    folds, summary = construct_folds(rows, verified_pairs, requested_folds, seed)
    summary["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["unknown"] + sorted({r["speaker_id"] for r in rows} - {"unknown"})
    (output_dir / "label_map.json").write_text(json.dumps({"labels": labels, "unknown_index": 0}, indent=2) + "\n", encoding="utf-8")
    support_path = summary_path.parent / "split_support.csv"
    with support_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["speaker_id", "file_count", "nonzero_integrity_passed_files", "eligible_content_groups", "supports_two_way_validation", "supports_requested_folds"])
        writer.writeheader()
        for label, group_count in summary["eligible_groups_per_known_class"].items():
            members = [r for r in rows if r["speaker_id"] == label]
            writer.writerow({"speaker_id": label, "file_count": len(members),
                             "nonzero_integrity_passed_files": sum(truth(r.get("usable_for_training")) for r in members),
                             "eligible_content_groups": group_count, "supports_two_way_validation": group_count >= 2,
                             "supports_requested_folds": group_count >= requested_folds})
    if folds:
        path = output_dir / "folds.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(folds[0]))
            writer.writeheader()
            writer.writerows(folds)
        summary["folds_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif (output_dir / "folds.csv").exists():
        # Invalidate the previous generated assignment if new evidence makes it infeasible.
        (output_dir / "folds.csv").unlink()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
