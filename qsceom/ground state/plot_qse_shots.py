#!/usr/bin/env python3
"""Plot QSE shot mean/variance with no-shot baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qse import run_qse


def _parse_shots(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            continue
        values.append(value)
    if not values:
        raise ValueError("No positive shot values were provided.")
    return values


def plot_results(shots, means, variances, no_shot_energy, eig_index, plot_path):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install with: `pip install matplotlib`."
        ) from exc

    x = np.asarray(shots, dtype=float)
    y = np.asarray(means, dtype=float)
    var = np.asarray(variances, dtype=float)
    yerr = np.sqrt(np.clip(var, 0.0, None))

    style_params = {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 15,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }

    with mpl.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        ax.axhline(
            float(no_shot_energy),
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=f"No-shot QSE eig[{eig_index}]",
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color="blue",
            ecolor="blue",
            elinewidth=1.4,
            capsize=4,
            label="QSE shot mean ± 1σ",
        )
        ax.set_xlabel("Shots")
        ax.set_ylabel("Energy (Ha)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        ax.legend(loc="best")

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot QSE shot mean/variance with no-shot baseline."
    )
    parser.add_argument(
        "--shots",
        type=str,
        default="100,500,1000,5000",
        help="Comma-separated positive shot values.",
    )
    parser.add_argument("--num-runs", type=int, default=3, help="Runs per shot value.")
    parser.add_argument("--seed", type=int, default=170, help="Base seed.")
    parser.add_argument(
        "--eig-index",
        type=int,
        default=0,
        help="Eigenvalue index to track (default: first eigenvalue).",
    )
    parser.add_argument(
        "--shot-repeats",
        type=int,
        default=1,
        help="Internal shot repeats in run_qse.",
    )
    parser.add_argument(
        "--use-aer-estimator",
        action="store_true",
        help="Use qiskit_aer.primitives.Estimator for QSE matrix elements.",
    )
    parser.add_argument("--basis", type=str, default="sto-6g", help="Basis set.")
    parser.add_argument("--active-electrons", type=int, default=4, help="Active electrons.")
    parser.add_argument("--active-orbitals", type=int, default=4, help="Active orbitals.")
    parser.add_argument(
        "--atom",
        type=str,
        default=(
            "N 0.0 0.0 0.0; "
            "H 2.526315789473684 0.0 0.0; "
            "H -0.506 0.876 0.0; "
            "H -0.506 -0.876 0.0"
        ),
        help="Molecular geometry string.",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Optional output PNG path.",
    )
    parser.add_argument(
        "--print-all-eigs",
        action="store_true",
        help="Print all eigenvalues for each run.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0")
    if args.eig_index < 0:
        raise ValueError("--eig-index must be >= 0")
    if args.shot_repeats <= 0:
        raise ValueError("--shot-repeats must be > 0")

    shot_values = _parse_shots(args.shots)

    baseline = run_qse(
        atom=args.atom,
        basis=args.basis,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
        seed=args.seed,
        shots=0,
        uccsd_shots=0,
    )
    baseline_eigs = np.asarray(baseline["qse_eigenvalues"], dtype=float)
    if args.eig_index >= baseline_eigs.size:
        raise IndexError(
            f"--eig-index {args.eig_index} is out of range for {baseline_eigs.size} eigenvalues."
        )
    no_shot_energy = float(baseline_eigs[args.eig_index])
    print(f"No-shot QSE eig[{args.eig_index}] = {no_shot_energy:.12f} Ha")
    if args.print_all_eigs:
        print("No-shot eigenvalues:", baseline_eigs)

    means = []
    variances = []
    for shot in shot_values:
        values = []
        for run_idx in range(args.num_runs):
            run_seed = args.seed + run_idx
            res = run_qse(
                atom=args.atom,
                basis=args.basis,
                active_electrons=args.active_electrons,
                active_orbitals=args.active_orbitals,
                seed=run_seed,
                shots=int(shot),
                uccsd_shots=0,
                shot_repeats=args.shot_repeats,
                use_aer_estimator=args.use_aer_estimator,
            )
            eigvals = np.asarray(res["qse_eigenvalues"], dtype=float)
            if args.eig_index >= eigvals.size:
                raise IndexError(
                    f"--eig-index {args.eig_index} is out of range for {eigvals.size} eigenvalues."
                )
            value = float(eigvals[args.eig_index])
            values.append(value)
            if args.print_all_eigs:
                print(f"shots={shot} run={run_idx + 1}/{args.num_runs} eigs={eigvals}")
            else:
                print(
                    f"shots={shot} run={run_idx + 1}/{args.num_runs} "
                    f"eig[{args.eig_index}]={value:.12f} Ha"
                )

        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        var = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
        means.append(mean)
        variances.append(var)
        print(f"shots={shot}: mean={mean:.12f} Ha, variance={var:.12e}")

    plot_path = (
        Path(args.plot_file).resolve()
        if args.plot_file
        else Path(__file__).resolve().with_name("qse_shots_mean_variance.png")
    )
    plot_results(
        shots=shot_values,
        means=means,
        variances=variances,
        no_shot_energy=no_shot_energy,
        eig_index=args.eig_index,
        plot_path=plot_path,
    )
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()

