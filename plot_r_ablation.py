from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# R1/R2 grid used in the experiments
R_VALUES = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
]

# Base paths
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data.nosync" / "experiments" / "lmsv" / "state-dim_4_emission-dim_4_num-timesteps_200" / "agsf"
FIG_ROOT = ROOT / "figures"

METRIC = "MSE"  # metric label expected in averaged_errors.txt
AVERAGED_ERRORS_FILE = "averaged_errors.txt"


def _format_val(val: float) -> str:
    """Format float to match directory naming (e.g., 0.0001 -> '0.0001', 0.1 -> '0.1')."""
    return f"{val:g}"


def read_metric(path: Path, metric: str = METRIC) -> float:
    """Read a metric value from averaged_errors.txt; returns NaN if missing."""
    if not path.exists():
        return np.nan
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].lower() == metric.lower():
            try:
                return float(parts[1])
            except ValueError:
                return np.nan
    return np.nan


def collect_grid(alg: str, metric: str = METRIC) -> np.ndarray:
    """Collect a r1 x r2 grid of metric values for the given algorithm."""
    base = DATA_ROOT / alg
    grid = np.full((len(R_VALUES), len(R_VALUES)), np.nan)
    missing = []

    for i, r1 in enumerate(R_VALUES):
        for j, r2 in enumerate(R_VALUES):
            folder = base / (
                f"M_100_N_5_L_5_iters_1_ac1_prop_r1_{_format_val(r1)}_ac2_prop_r2_{_format_val(r2)}"
            )
            metric_path = folder / AVERAGED_ERRORS_FILE
            val = read_metric(metric_path, metric)
            grid[i, j] = val
            if np.isnan(val):
                missing.append(metric_path)

    if missing:
        print(f"[{alg}] Missing/NaN entries: {len(missing)}")
        for m in missing:
            print(f"  - {m}")
    return grid


def plot_heatmap(grid: np.ndarray, alg: str, out_dir: Path, metric: str = METRIC) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(grid, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(R_VALUES)))
    ax.set_yticks(range(len(R_VALUES)))
    ax.set_xticklabels([_format_val(v) for v in R_VALUES], rotation=45, ha="right")
    ax.set_yticklabels([_format_val(v) for v in R_VALUES])
    ax.set_xlabel("r2")
    ax.set_ylabel("r1")
    ax.set_title(f"{alg.upper()} {metric} over r1/r2")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    fig.tight_layout()

    out_path = out_dir / f"lmsv_{alg}_r1_r2_{metric.lower()}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot r1/r2 heatmaps for lagsf/uagsf.")
    parser.add_argument("--metric", default=METRIC, help="Metric label to extract from averaged_errors.txt (default: MSE)")
    parser.add_argument("--out-dir", type=Path, default=FIG_ROOT, help="Output directory for figures")
    return parser.parse_args()


def main():
    args = parse_args()
    metric = args.metric
    out_dir = args.out_dir

    algs = ["lagsf", "uagsf"]
    for alg in algs:
        grid = collect_grid(alg, metric)
        plot_heatmap(grid, alg, out_dir, metric)


if __name__ == "__main__":
    main()
