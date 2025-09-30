#!/usr/bin/env python3
"""
Pipeline to generate filtered Average Sensitivity vs Sparsity plot for a given eval.

Steps (idempotent; recompute only if outputs are missing unless --force-* is passed):
1) Combine sensitivity.csv and sensitivity_check.csv across all SAE subdirs under evals/{eval_name}
2) Filter features by sensitivity sanity check threshold (e.g., 90% => 0.90)
3) Aggregate non-weighted average sensitivity per SAE
4) Ensure sparsity map exists (sae_name -> L0 sparsity)
5) Join and save sparsity_sensitivity CSV
6) Plot Average Sensitivity vs Sparsity (log-x) for the sanity check threshold (e.g., 90%)

Produces:
- analysis/{eval}/{eval}_avg_sensitivity_check{pct}.csv
- analysis/{eval}/{eval}_sparsity_sensitivity_check{pct}.csv
- analysis/{eval}/{eval}_sparsity_sensitivity_check{pct}.png
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_eval_files(eval_dir: Path, file_name: str) -> List[Path]:
    files: List[Path] = []
    for root, _, filenames in os.walk(eval_dir, followlinks=True):
        if file_name in filenames:
            file_path = Path(root) / file_name
            # exclude files in 'old' folders
            if "old" not in file_path.parts:
                files.append(file_path)
    return files


def combine_sensitivity_like(eval_name: str, file_name: str, output_csv: Path) -> pd.DataFrame:
    eval_dir = get_repo_root() / "evals" / eval_name
    files = find_eval_files(eval_dir, file_name)
    if not files:
        raise FileNotFoundError(f"No {file_name} files found under {eval_dir}")

    frames = []
    for fp in sorted(files, key=lambda p: p.parts[-2]):
        try:
            df = pd.read_csv(fp)
            df["eval_name"] = eval_name
            df["sae_name"] = fp.parent.name
            # keep only the required columns if present
            cols = [c for c in ["eval_name", "sae_name", "feature_index", "sensitivity"] if c in df.columns]
            df = df[cols]
            frames.append(df)
        except Exception as e:
            print(f"Warning: failed to read {fp}: {e}")

    if not frames:
        raise RuntimeError(f"No valid data in any {file_name} under {eval_dir}")

    combined = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({combined.shape[0]} rows)")
    return combined


def load_or_build_combined(eval_name: str, which: str, force: bool = False) -> pd.DataFrame:
    """which is either 'sensitivity' or 'sensitivity_check'"""
    analysis_dir = get_repo_root() / "analysis" / eval_name
    combined_path = analysis_dir / f"{eval_name}_{which}.csv"
    if combined_path.exists() and not force:
        print(f"Found existing {combined_path}")
        return pd.read_csv(combined_path)

    file_name = f"{which}.csv"
    return combine_sensitivity_like(eval_name, file_name, combined_path)


def filter_by_check_threshold(sensitivity_df: pd.DataFrame, check_df: pd.DataFrame, pct: int) -> pd.DataFrame:
    threshold = pct / 100.0
    print(f"Filtering features with sensitivity_check >= {threshold:.2f} ({pct}%)")

    if not {"eval_name", "sae_name", "feature_index", "sensitivity"}.issubset(check_df.columns):
        raise ValueError("sensitivity_check DataFrame must include eval_name, sae_name, feature_index, sensitivity")

    keep_keys = check_df.loc[check_df["sensitivity"] >= threshold, ["eval_name", "sae_name", "feature_index"]]
    print(f"Features above threshold: {len(keep_keys)}")

    merged = sensitivity_df.merge(
        keep_keys,
        on=["eval_name", "sae_name", "feature_index"],
        how="inner",
    )
    print(f"Filtered sensitivity rows: {len(merged)} (from {len(sensitivity_df)})")
    return merged


def aggregate_avg_sensitivity(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if not {"sae_name", "sensitivity"}.issubset(filtered_df.columns):
        raise ValueError("filtered_df must include sae_name and sensitivity columns")
    agg = (
        filtered_df
        .groupby("sae_name", as_index=False)
        .agg(**{"Avg. Sensitivity": ("sensitivity", "mean")})
        .sort_values("Avg. Sensitivity", ascending=False)
    )
    print(f"Aggregated to {len(agg)} SAE rows")
    return agg


def get_sae_type(sae_name: str) -> str:
    mapping = {
        "_btopk_": "BatchTopK",
        "_matbtopk_": "MatryoshkaBatchTopK",
        "_gated_": "Gated",
        "_jumprelu_": "JumpReLu",
        "_topk_": "TopK",
        "_relu_": "Relu",
        "_pan_": "PAnneal",
    }
    for key, value in mapping.items():
        if key in sae_name:
            return value
    return "Unknown"


def build_sparsity_sensitivity(eval_name: str, agg_df: pd.DataFrame, analysis_dir: Path) -> pd.DataFrame:
    sparsity_map_csv = analysis_dir / f"{eval_name}_sparsity_map.csv"
    if not sparsity_map_csv.exists():
        print(f"Sparsity map not found at {sparsity_map_csv}. Attempting to create it...")
        try:
            import sys
            sys.path.append(str(get_repo_root() / "src"))
            from saebench_utils import create_sparsity_map  # type: ignore
            create_sparsity_map(eval_name=eval_name, type="core")
        except Exception as e:
            raise RuntimeError(f"Failed to create sparsity map automatically: {e}")

    smap = pd.read_csv(sparsity_map_csv)
    if not {"sae_name", "l0_sparsity"}.issubset(smap.columns):
        raise ValueError(f"Invalid sparsity map columns in {sparsity_map_csv}")

    df = agg_df.copy()
    df["L0 (Sparsity)"] = df["sae_name"].map({r["sae_name"]: r["l0_sparsity"] for _, r in smap.iterrows()})
    df["SAE Type"] = df["sae_name"].apply(get_sae_type)
    df = df[["SAE Type", "L0 (Sparsity)", "Avg. Sensitivity", "sae_name"]]
    df = df.sort_values(["SAE Type", "L0 (Sparsity)"])
    return df


def plot_avg_sensitivity_vs_sparsity(df: pd.DataFrame, output_png: Path, title_suffix: str = "") -> None:
    style = {
        "Gated": {"marker": "o", "color": "#CC6666"},
        "JumpReLu": {"marker": "^", "color": "#6699CC"},
        "BatchTopK": {"marker": "s", "color": "#D4A35A"},
        "MatryoshkaBatchTopK": {"marker": "D", "color": "#66CC99"},
        "TopK": {"marker": "v", "color": "#9966CC"},
        "Relu": {"marker": "P", "color": "#999999"},
        "PAnneal": {"marker": "X", "color": "#CC6699"},
        "Unknown": {"marker": "o", "color": "#777777"},
    }

    plt.figure(figsize=(8, 6))
    for sae_type, group in df.groupby("SAE Type"):
        st = style.get(sae_type, style["Unknown"])
        plt.plot(
            group["L0 (Sparsity)"],
            group["Avg. Sensitivity"],
            marker=st["marker"],
            color=st["color"],
            markersize=10,
            linestyle='-',
            label=sae_type,
        )

    plt.xscale('log')
    plt.xlabel('L0 (Sparsity, log scale)')
    plt.ylabel('Average Sensitivity')
    if title_suffix:
        plt.title(f'Average Sensitivity vs. Sparsity\n{title_suffix}')
    else:
        plt.title('Average Sensitivity vs. Sparsity')
    plt.ylim((0.8, 1.01))
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate filtered Avg Sensitivity vs Sparsity plot")
    parser.add_argument("--eval_name", type=str, default="run_1", help="Evaluation run name, e.g., run_1")
    parser.add_argument("--check", type=int, default=90, help="Sensitivity check threshold percentage (0-100)")
    parser.add_argument("--force-combine", action="store_true", help="Rebuild combined CSVs even if present")
    args = parser.parse_args()

    eval_name = args.eval_name
    pct = int(args.check)
    if pct < 0 or pct > 100:
        raise ValueError("--check must be between 0 and 100")

    repo_root = get_repo_root()
    analysis_dir = repo_root / "analysis" / eval_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load/combine sensitivity and sensitivity_check
    sensitivity_df = load_or_build_combined(eval_name, "sensitivity", force=args.force_combine)
    sens_check_df = load_or_build_combined(eval_name, "sensitivity_check", force=args.force_combine)

    # 2) Filter by threshold (or reuse existing filtered file if present)
    filtered_csv = analysis_dir / f"{eval_name}_sensitivity_check{pct:02d}.csv"
    if filtered_csv.exists() and not args.force_combine:
        print(f"Found existing filtered file: {filtered_csv}")
        filtered_df = pd.read_csv(filtered_csv)
        # ensure expected columns
        expected_cols = {"eval_name", "sae_name", "feature_index", "sensitivity"}
        if not expected_cols.issubset(filtered_df.columns):
            print("Existing filtered file missing expected columns; recomputing filter...")
            filtered_df = filter_by_check_threshold(sensitivity_df, sens_check_df, pct)
    else:
        filtered_df = filter_by_check_threshold(sensitivity_df, sens_check_df, pct)
        filtered_df.to_csv(filtered_csv, index=False)
        print(f"Wrote filtered per-feature CSV: {filtered_csv}")

    # 3) Aggregate to average sensitivity for each SAE
    agg_df = aggregate_avg_sensitivity(filtered_df)
    agg_csv = analysis_dir / f"{eval_name}_avg_sensitivity_check{pct:02d}.csv"
    agg_df.to_csv(agg_csv, index=False)
    print(f"Wrote aggregated CSV: {agg_csv}")

    # 4) Build sparsity_sensitivity table
    ss_df = build_sparsity_sensitivity(eval_name, agg_df, analysis_dir)
    ss_csv = analysis_dir / f"{eval_name}_sparsity_sensitivity_check{pct:02d}.csv"
    ss_df.to_csv(ss_csv, index=False)
    print(f"Wrote sparsity_sensitivity CSV: {ss_csv}")

    # 5) Plot
    png_out = analysis_dir / f"{eval_name}_sparsity_sensitivity_check{pct:02d}.png"
    plot_avg_sensitivity_vs_sparsity(ss_df, png_out, title_suffix=f"_check{pct:02d}")


if __name__ == "__main__":
    main()
