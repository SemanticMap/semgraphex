#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot tanh dynamics distributions from comparison.csv")
    ap.add_argument("csv", help="Path to comparison.csv")
    ap.add_argument("--outdir", default="plots_tanh", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "model" in df.columns:
        df = df[df["model"] == "tanh"].copy()
    df = df.sort_values("fine_node_count", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df["fine_node_count"], df["full_trajectory_relative_error"], marker="o", markersize=2, linewidth=1, label="D_full")
    plt.plot(df["fine_node_count"], df["post_transient_relative_error"], marker="o", markersize=2, linewidth=1, label="D_post")
    plt.plot(df["fine_node_count"], df["slow_coordinate_relative_error"], marker="o", markersize=2, linewidth=1, label="D_slow")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Number of nodes N (log scale, coarse-graining to the right)")
    plt.ylabel("Relative error")
    plt.title("tanh dynamics: errors across scales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "tanh_errors_vs_N.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.step(df["fine_node_count"], df["selected_r"], where="post")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Number of nodes N (log scale, coarse-graining to the right)")
    plt.ylabel("selected r")
    plt.title("Slow-space dimension r across scales")
    plt.tight_layout()
    plt.savefig(outdir / "tanh_r_vs_N.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["slow_coordinate_relative_error"], bins=40)
    plt.xlabel("D_slow")
    plt.ylabel("Count")
    plt.title("Distribution of slow-coordinate error")
    plt.tight_layout()
    plt.savefig(outdir / "tanh_hist_Dslow.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["post_transient_relative_error"], bins=40)
    plt.xlabel("D_post")
    plt.ylabel("Count")
    plt.title("Distribution of post-transient error")
    plt.tight_layout()
    plt.savefig(outdir / "tanh_hist_Dpost.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["full_trajectory_relative_error"], bins=40)
    plt.xlabel("D_full")
    plt.ylabel("Count")
    plt.title("Distribution of full-trajectory error")
    plt.tight_layout()
    plt.savefig(outdir / "tanh_hist_Dfull.png", dpi=200)
    plt.close()

    peaks = df.sort_values("slow_coordinate_relative_error", ascending=False).head(20)
    peaks[[
        "level",
        "fine_node_count",
        "coarse_node_count",
        "selected_r",
        "slow_coordinate_relative_error",
        "post_transient_relative_error",
        "full_trajectory_relative_error",
    ]].to_csv(outdir / "top20_Dslow_peaks.csv", index=False)

    print("Saved to", outdir.resolve())

if __name__ == "__main__":
    main()
