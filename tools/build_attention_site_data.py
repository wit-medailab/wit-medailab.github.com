import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PANEL_ROOT = ROOT / "all_patient_attention_results_morphology_guided_filtered"
SOURCE_CASE_ROOT = ROOT / "case_csvs_all"
TARGET_IMAGE_ROOT = REPO_ROOT / "img" / "cases"
TARGET_DATA_DIR = REPO_ROOT / "data"


CASE_GROUPS = {
    "high_risk": [
        "16-02265",
        "16-32397",
    ],
    "low_risk": [
        "20-04611",
        "20-08950",
    ],
}


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_top_scores(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    def top_row(key: str):
        valid = [row for row in rows if row.get("x") != "-1" and row.get("y") != "-1"]
        best = max(valid, key=lambda row: float(row[key]))
        return {
            "x": int(float(best["x"])),
            "y": int(float(best["y"])),
            "percentile": round(float(best[key]), 4),
        }

    return {
        "pathology_self_percentile": top_row("pathology_self_percentile"),
        "clinical_guided_percentile": top_row("clinical_guided_percentile"),
        "cell_guided_percentile": top_row("cell_guided_percentile"),
    }


def build_case(group: str, patient_id: str):
    panel_dir = SOURCE_PANEL_ROOT / group / patient_id
    case_dir = SOURCE_CASE_ROOT / patient_id

    meta_path = case_dir / f"{patient_id}_guided_attention_meta.json"
    csv_path = case_dir / f"{patient_id}_guided_patch_attention_scores.csv"

    image_sources = {
        "panel": panel_dir / f"{patient_id}_interpretability_panel.png",
        "wholeSlide": panel_dir / f"{patient_id}_whole_slide_crop.png",
        "clinicalHeatmap": panel_dir / f"{patient_id}_clinical_guided_heatmap.png",
        "cellHeatmap": panel_dir / f"{patient_id}_cell_guided_heatmap.png",
        "clinicalPercentile": case_dir / f"{patient_id}_clinical_guided_percentile.png",
        "cellPercentile": case_dir / f"{patient_id}_cell_guided_percentile.png",
        "pathologySelf": case_dir / f"{patient_id}_pathology_self_percentile.png",
    }

    target_dir = TARGET_IMAGE_ROOT / patient_id
    image_paths = {}
    for key, src in image_sources.items():
        if src.exists():
            dst = target_dir / src.name
            copy_file(src, dst)
            image_paths[key] = f"img/cases/{patient_id}/{src.name}"

    meta = read_json(meta_path)
    top_scores = read_top_scores(csv_path)

    return {
        "id": patient_id,
        "label": patient_id,
        "riskGroup": group,
        "patientId": patient_id,
        "fold": f"fold{meta['fold_id']}",
        "patchCount": meta["patch_count"],
        "patchSize": f"{meta['stride']} x {meta['stride']}",
        "featureDim": meta["feature_dim"],
        "gridShape": f"{meta['grid_shape'][0]} x {meta['grid_shape'][1]}",
        "clinicalTokenCount": len(meta["clinical_cols"]),
        "cellGroupCount": len(meta["cell_group_names"]),
        "cellGroups": meta["cell_group_names"],
        "images": image_paths,
        "topScores": top_scores,
    }


def main():
    TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cases = []
    for group, patient_ids in CASE_GROUPS.items():
        for patient_id in patient_ids:
            cases.append(build_case(group, patient_id))

    payload = {
        "paperTitle": "BiCrossSurv attention visualization for prostate cancer prognosis",
        "paperDescription": "This page presents BiCrossSurv cross-modal attention results with risk-group filtering, case switching, interpretability panels, heatmaps, percentile maps, and auto-extracted metadata.",
        "riskGroups": [
            {"id": "all", "label": "All cases"},
            {"id": "high_risk", "label": "High risk"},
            {"id": "low_risk", "label": "Low risk"}
        ],
        "cases": cases,
    }

    with (TARGET_DATA_DIR / "cases.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
