"""Validate final automatic EDA provenance, coverage and calibration separation."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from check_eda import ROOT, digest, read_csv, run as check_initial


def main():
    check_initial()
    reports, processed = ROOT/"reports/eda", ROOT/"data/processed/eda_v1"
    def read(name):
        return json.loads((reports/name).read_text(encoding="utf-8"))
    manifest = read_csv(processed/"audio_manifest.csv")
    mh = digest(processed/"audio_manifest.csv")
    names = {r["audio_file"] for r in manifest}
    nonzero = {r["audio_file"] for r in manifest if r["has_nonzero_signal"] == "True"}
    embedding = read("embedding_summary.json")
    geometry = read("embedding_geometry_summary.json")
    semantic = read("semantic_summary.json")
    forensics = read("forensics_summary.json")
    quality = read("quality_review_summary.json")
    for summary in (embedding, geometry, semantic, quality):
        assert summary["manifest_sha256"] == mh
    assert forensics["input_manifest_sha256"] == mh
    assert embedding["status"] == "complete" and embedding["source_files"] == len(names)
    embedded = read_csv(reports/"embedding_files.csv")
    assert len(embedded) == len(names) and {r["audio_file"] for r in embedded} == names
    assert {r["audio_file"] for r in embedded if r["status"] == "ok"} == nonzero
    assert embedding["embedded_files"] == len(nonzero) == geometry["embedded_files"]
    assert embedding["embedding_npz_sha256"] == digest(ROOT/"artifacts/eda/ecapa/embeddings.npz")
    assert geometry["embedding_summary_sha256"] == digest(reports/"embedding_summary.json")
    assert geometry["folds_sha256"] == digest(processed/"folds.csv")
    for summary, field in ((embedding,"artifact_sha256"),(geometry,"artifact_sha256"),(semantic,"output_sha256"),(quality,"input_sha256")):
        for filename, expected in summary[field].items():
            assert digest(reports/filename) == expected, filename
    for summary, source in ((embedding,"embeddings.py"),(geometry,"geometry.py"),(semantic,"semantic.py"),(quality,"triage.py")):
        assert summary["code_sha256"] == digest(ROOT/"src/speaker_id/eda"/source), source
    assert semantic["listening_sha256"] == digest(reports/"listening_queue.csv")
    assert semantic["sensitivity_sha256"] == digest(reports/"vad_sensitivity.csv")
    assert semantic["human_listening_completed"] is False
    assert quality["human_listening_completed"] is False
    assert forensics["output_csv_sha256"] == digest(reports/"forensics.csv")
    assert forensics["near_empty_csv_sha256"] == digest(reports/"forensics_near_empty.csv")
    assert forensics["histogram_sha256"] == digest(reports/"forensics_code_histograms.json")
    for name, expected in forensics["figure_sha256"].items():
        assert digest(reports/"figures"/name) == expected
    for name, expected in forensics["code_sha256"].items():
        folder = "audio" if name == "io.py" else "eda"
        assert digest(ROOT/"src/speaker_id"/folder/name) == expected
    semantic_rows = read_csv(reports/"semantic_files.csv")
    assert {r["audio_file"] for r in semantic_rows} <= names
    assert len(semantic_rows) == semantic["coverage"]["selected_files"]
    review = read_csv(reports/"quality_review.csv")
    assert len(review) == len(names) and {r["audio_file"] for r in review} == names
    assert quality["quality_review_sha256"] == digest(reports/"quality_review.csv")
    roles = read_csv(processed/"calibration_roles.csv")
    calibration = read("calibration_summary.json")
    assert calibration["calibration_roles_sha256"] == digest(processed/"calibration_roles.csv")
    for name in ("calibration_summary.json", "metric_contract.json"):
        s = read(name)
        for filename, expected in s["source_sha256"].items():
            assert digest(processed/filename) == expected
        for filename, expected in s["code_sha256"].items():
            assert digest(ROOT/filename) == expected
    known = {r["speaker_id"] for r in manifest} - {"unknown"}
    outer_counts = defaultdict(int)
    for fold in {r["outer_fold"] for r in roles}:
        rows = [r for r in roles if r["outer_fold"] == fold]
        assert len(rows) == len(names) and {r["audio_file"] for r in rows} == names
        groups = {}
        for role, field in (("fit","encoder_fit_allowed"),("enroll","enrollment_allowed"),("query","calibration_query"),("outer","outer_evaluation_included")):
            selected = [r for r in rows if r[field] == "True"]
            groups[role] = {r["group_id"] for r in selected}
            if role == "enroll":
                assert {r["speaker_id"] for r in selected} == known
            if role == "outer":
                for row in selected:
                    outer_counts[row["audio_file"]] += 1
        assert not groups["query"] & (groups["fit"] | groups["enroll"] | groups["outer"])
        assert not groups["outer"] & (groups["fit"] | groups["enroll"])
    assert dict(outer_counts) == dict.fromkeys(names, 1)
    overlap = read("embedding_overlap_summary.json")
    for name, expected in overlap["input_sha256"].items():
        folder = processed if name == "audio_manifest.csv" else reports
        assert digest(folder/name) == expected
    assert overlap["output_csv_sha256"] == digest(reports/"embedding_overlap_pairs.csv")
    for name, expected in overlap["code_sha256"].items():
        folder = "audio" if name == "io.py" else "eda"
        assert digest(ROOT/"src/speaker_id"/folder/name) == expected
    final = read("final_report_stats.json")
    # Paths in final report's artifact manifest are project-relative.
    for filename, expected in final["input_sha256"].items():
        assert digest(ROOT/filename) == expected, f"Final report stale: {filename}"
    assert final["builder_sha256"] == digest(ROOT/"src/speaker_id/eda/final_report.py")
    assert final["entrypoint_sha256"] == digest(ROOT/"scripts/eda/build_final_report.py")
    assert final["final_html_sha256"] == digest(reports/"final.html")
    result = {"status":"passed", "original_rows":len(names), "embedded_nonzero_rows":len(nonzero),
              "semantic_selected_files":len(semantic_rows), "calibration_role_rows":len(roles),
              "human_listening_completed":False, "calibration_group_overlap":0,
              "final_report_sha256":digest(reports/"final.html"),
              "checks":"Initial EDA integrity, all-data frozen embeddings, code/data/output hashes, selective semantic coverage, review row preservation, calibration split independence and final HTML freshness"}
    (reports/"final_artifact_validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
