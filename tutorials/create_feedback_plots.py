import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.plotting import (
    plot_tax_period_rows_with_environment_metrics,
    plot_tax_bracket_snapshots_with_environment_table,
)


def _latest_result_dir(results_root):
    candidates = [
        path for path in Path(results_root).iterdir()
        if path.is_dir() and (path / "dense_logs_final.pkl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No result folders with dense_logs_final.pkl under {results_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_log(result_dir, episode):
    with (Path(result_dir) / "dense_logs_final.pkl").open("rb") as f:
        dense_logs = pickle.load(f)

    if isinstance(dense_logs, dict):
        return dense_logs[episode]
    return dense_logs[episode]


def main():
    parser = argparse.ArgumentParser(
        description="Create tax-period environment table and tax-bracket snapshot plots from a dense rollout log."
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Result folder containing dense_logs_final.pkl. Defaults to the newest folder in tutorials/results.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Dense-log episode index.")
    parser.add_argument("--period", type=int, default=100, help="Tax period length in timesteps.")
    parser.add_argument("--rate-disc", type=float, default=0.05, help="Planner tax-rate discretization.")
    parser.add_argument("--n-snapshots", type=int, default=10, help="Number of tax-period snapshots to show.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated PNG files.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir) if args.result_dir else _latest_result_dir(Path("results"))
    output_dir = Path(args.output_dir) if args.output_dir else result_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    log = _load_log(result_dir, args.episode)

    brackets = [0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3]

    fig, df_income, df_counts, df_outcomes, df_env, df_tax = plot_tax_bracket_snapshots_with_environment_table(
        log,
        brackets=brackets,
        period=args.period,
        rate_disc=args.rate_disc,
        n_snapshots=args.n_snapshots,
    )
    fig_path = output_dir / "tax_bracket_snapshots_with_environment_table.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_rows, *_ = plot_tax_period_rows_with_environment_metrics(
        log,
        brackets=brackets,
        period=args.period,
        rate_disc=args.rate_disc,
        n_snapshots=args.n_snapshots,
    )
    fig_rows_path = output_dir / "tax_period_rows_environment_snapshots.png"
    fig_rows.savefig(fig_rows_path, dpi=220, bbox_inches="tight")
    plt.close(fig_rows)

    df_income.to_csv(output_dir / "tax_period_income_by_agent_region.csv", index=False)
    df_counts.to_csv(output_dir / "tax_bracket_counts_by_region.csv", index=False)
    df_outcomes.to_csv(output_dir / "tax_period_outcomes_by_region.csv", index=False)
    df_env.to_csv(output_dir / "environment_metrics_by_tax_period_region.csv", index=False)
    df_tax.to_csv(output_dir / "tax_policy_by_period_region.csv", index=False)

    print(f"Saved {fig_path}")
    print(f"Saved {fig_rows_path}")
    print(f"Saved {output_dir / 'environment_metrics_by_tax_period_region.csv'}")
    print(f"Saved {output_dir / 'tax_bracket_counts_by_region.csv'}")


if __name__ == "__main__":
    main()
