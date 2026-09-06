"""Recompute corrected peak heights, preserving all other production CSV fields.

Run from the project root; --publish backs up the old CSVs before replacing them.
No trajectory, figure, statistical threshold, or non-height CSV field is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


REVISION = Path(__file__).resolve().parent
ROOT = REVISION.parents[1]
sys.path.insert(0, str(ROOT))

import boundary_arc_correlation_analysis as analysis


HEIGHT = "autocorrelation_peak_height_mean"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolved(table: pd.DataFrame) -> np.ndarray:
    return (table[HEIGHT].astype(float).to_numpy() >= 0.30) & (
        table["boundary_spacing_time_std"].astype(float).to_numpy()
        / table["boundary_spacing_mean"].astype(float).to_numpy()
        <= 0.10
    )


def refresh(filename: str, runner, publish: bool) -> dict:
    source = analysis.OUT / filename
    destination = REVISION / "particle_peak_heights" / filename
    backup = REVISION / "before" / "particle_peak_heights" / filename
    source_before = source.read_bytes()
    baseline = backup.read_bytes() if backup.exists() else source_before
    old = pd.read_csv(io.BytesIO(baseline), dtype=str, keep_default_na=False)
    keys = ["seed", "family"] if "family" in old.columns else ["seed"]
    rows = []
    for seed in range(1, 11):
        result = runner(seed)
        rows.extend(result if isinstance(result, list) else [result])
        print(f"{filename}: checked seed {seed}", flush=True)
    recomputed = pd.DataFrame(rows)
    indexed = {
        tuple(str(row[key]) for key in keys): row
        for row in recomputed.to_dict(orient="records")
    }
    if len(indexed) != len(old):
        raise RuntimeError(f"Seed/stream count changed: {filename}")

    updated = old.copy()
    differences = []
    for index, row in old.iterrows():
        key = tuple(row[name] for name in keys)
        fresh = indexed[key]
        for name in old.columns:
            if name == HEIGHT:
                continue
            if isinstance(fresh[name], (float, int, np.number)):
                same = np.isclose(
                    float(row[name]), float(fresh[name]), rtol=0.0, atol=1e-12
                )
            else:
                same = row[name] == str(fresh[name])
            if not same:
                raise RuntimeError(f"Non-height value changed: {filename}, {key}, {name}")
        corrected = float(fresh[HEIGHT])
        updated.at[index, HEIGHT] = format(corrected, ".17g")
        differences.append(
            {
                **dict(zip(keys, key)),
                "old_peak_height_mean": float(row[HEIGHT]),
                "corrected_peak_height_mean": corrected,
                "difference": corrected - float(row[HEIGHT]),
            }
        )

    unchanged = [name for name in old.columns if name != HEIGHT]
    if not updated[unchanged].equals(old[unchanged]):
        raise RuntimeError("A non-height string field was modified")
    classification_changes = int(np.count_nonzero(resolved(old) != resolved(updated)))
    if classification_changes:
        raise RuntimeError(f"Quality classification changed for {filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = updated.to_csv(index=False).encode("utf-8")
    destination.write_bytes(payload)

    if publish:
        if source.read_bytes() != source_before:
            raise RuntimeError(f"Production CSV changed during recomputation: {source}")
        if source_before not in (baseline, payload):
            raise RuntimeError(f"Refusing to overwrite unrelated CSV edits: {source}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(source, backup)
        source.write_bytes(payload)
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"Published readback mismatch: {source}")

    return {
        "source_csv": str(source),
        "backup_csv": str(backup),
        "revised_csv": str(destination),
        "row_count": len(old),
        "changed_column": HEIGHT,
        "all_other_fields_preserved_as_strings": True,
        "quality_classification_changes": classification_changes,
        "unresolved": updated.loc[~resolved(updated), keys].to_dict(orient="records"),
        "maximum_absolute_peak_height_change": max(abs(row["difference"]) for row in differences),
        "baseline_sha256": digest(baseline),
        "corrected_sha256": digest(payload),
        "published": publish,
        "rows": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    reports = [
        refresh("HalfPi_Boundary_Arc_Correlation_10Seeds.csv", analysis.halfpi_seed, args.publish),
        refresh("Pi_Boundary_Arc_Correlation_10Seeds.csv", analysis.pi_seed, args.publish),
    ]
    report = {
        "production_change": "Quadratic peak-height coefficient 1/2 corrected to 1/4.",
        "trajectory_or_figure_files_written": False,
        "thresholds_unchanged": {"minimum_mean_peak_height": 0.30, "maximum_temporal_cv": 0.10},
        "summary_json": str(analysis.OUT / "Lattice_Scale_10Seed_Summary.json"),
        "summary_json_changed": False,
        "summary_reason": "It contains classification and spacing aggregates, which are unchanged; it stores no per-row peak heights.",
        "datasets": reports,
    }
    path = REVISION / "particle_peak_heights" / "Peak_Height_Correction_Verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verification: {path}", flush=True)
    for result in reports:
        print(
            f"{result['row_count']} rows: only {HEIGHT} changed; "
            f"classification changes={result['quality_classification_changes']}; "
            f"published={result['published']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
