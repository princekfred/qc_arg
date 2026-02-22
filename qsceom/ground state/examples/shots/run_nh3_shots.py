#!/usr/bin/env python
"""Run NH3 QSC-EOM shot statistics for the first eigenvalue only.

Workflow
--------
1. Run ADAPT-VQE once with analytic expectations (no finite shots).
2. Run QSC-EOM once with ``shots=0`` to get the no-shot baseline.
3. For each requested shot count, run QSC-EOM ``N`` times and collect the
   first eigenvalue.
4. Compute mean/variance and generate a baseline + errorbar plot.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

NH3_QSCEOM_SHOTS_INPUT = {
    "name": "nh3_qsceom_shots_scan",
    "description": (
        "NH3 input for QSC-EOM shot study while keeping ADAPT-VQE analytic."
    ),
    "units": "angstrom",
    "symbols": ["N", "H", "H", "H"],
    "geometry": [
        [0.0, 0.0, 0.0],
        [2.526315789473684, 0.0, 0.0],
        [-0.506, 0.876, 0.0],
        [-0.506, -0.876, 0.0],
    ],
    "adaptvqe": {
        "shots": 0,
        "enabled": True,
        "note": (
            "Do not enable shot-noise sampling in ADAPT-VQE; keep analytic "
            "expectation values."
        ),
    },
    "qsceom": {
        "shots": [500, 1000, 10000],
        "note": "Run QSC-EOM separately for each shots value.",
    },
}


def _load_ground_modules():
    script_path = Path(__file__).resolve()
    ground_state_dir = script_path.parents[2]
    if str(ground_state_dir) not in sys.path:
        sys.path.insert(0, str(ground_state_dir))

    try:
        from adaptvqe import adapt_vqe
        from qsceom_par import qsc_eom
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Could not import adapt_vqe/qsc_eom from qsceom/ground state."
        ) from exc
    return adapt_vqe, qsc_eom


def _load_input():
    return NH3_QSCEOM_SHOTS_INPUT


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run NH3 QSC-EOM first-eigenvalue shot statistics "
            "(baseline no-shot + repeated finite-shot runs)."
        )
    )
    parser.add_argument(
        "--adapt-it",
        type=int,
        default=10,
        help="Number of ADAPT-VQE iterations used to obtain ansatz parameters.",
    )
    parser.add_argument(
        "--runs-per-shot",
        type=int,
        default=10,
        help="Number of independent QSC-EOM repeats for each finite-shot value.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-6g",
        help="Basis set for ADAPT-VQE and QSC-EOM.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total molecular charge.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2S spin value for the electronic structure problem.",
    )
    parser.add_argument(
        "--active-electrons",
        type=int,
        default=4,
        help="Number of active electrons.",
    )
    parser.add_argument(
        "--active-orbitals",
        type=int,
        default=4,
        help="Number of active orbitals.",
    )
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=500,
        help="Maximum optimizer iterations per ADAPT step.",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Optional output TXT path. Defaults to this file's directory.",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Optional output plot path. Defaults to this file's directory.",
    )
    return parser


def _plot_stats(shots, means, variances, no_shot_value, plot_path):
    try:
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires NumPy and matplotlib. Install with: "
            "`pip install numpy matplotlib`."
        ) from exc

    x = np.asarray(shots, dtype=float)
    y = np.asarray(means, dtype=float)
    var = np.asarray(variances, dtype=float)
    yerr = np.sqrt(np.clip(var, a_min=0.0, a_max=None))

    style_params = {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 15,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 600,
    }

    with mpl.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
        ax.axhline(
            no_shot_value,
            color="#b30000",
            linestyle="--",
            linewidth=1.8,
            label=f"No-shot baseline = {no_shot_value:.8f} Ha",
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color="#b30000",
            ecolor="#b30000",
            capsize=4,
            elinewidth=1.4,
            markersize=7,
            label="Finite-shot mean ± 1σ",
        )
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        ax.set_xlabel("QSC-EOM shot count")
        ax.set_ylabel("First eigenvalue (Hartree)")
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        ax.legend(loc="best", handlelength=2.6)

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.runs_per_shot <= 0:
        raise ValueError("--runs-per-shot must be > 0")
    if args.adapt_it <= 0:
        raise ValueError("--adapt-it must be > 0")

    adapt_vqe, qsc_eom = _load_ground_modules()
    input_cfg = _load_input()

    script_dir = Path(__file__).resolve().parent
    output_txt = (
        Path(args.output_txt)
        if args.output_txt is not None
        else script_dir / "nh3_shots.txt"
    )
    plot_path = (
        Path(args.plot_file)
        if args.plot_file is not None
        else script_dir / "nh3_shots.png"
    )

    symbols = list(input_cfg["symbols"])
    geometry = input_cfg["geometry"]
    qsceom_shots = [int(s) for s in input_cfg["qsceom"]["shots"]]
    adapt_shots_cfg = input_cfg.get("adaptvqe", {}).get("shots", 0)
    adapt_shots = None if adapt_shots_cfg in (None, 0) else int(adapt_shots_cfg)

    print("Running ADAPT-VQE once (analytic mode)...", flush=True)
    t0 = time.time()
    params, ash_excitation, energies = adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=args.adapt_it,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
        shots=adapt_shots,
        optimizer_maxiter=args.optimizer_maxiter,
    )
    adapt_time_s = time.time() - t0

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("NumPy is required for statistics.") from exc

    print("Running no-shot QSC-EOM baseline...", flush=True)
    t0 = time.time()
    eigvals_baseline, _ = qsc_eom(
        symbols=symbols,
        coordinates=geometry,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
        charge=args.charge,
        params=params,
        ash_excitation=ash_excitation,
        shots=0,
        basis=args.basis,
    )
    baseline_time_s = time.time() - t0
    baseline_first = float(np.asarray(eigvals_baseline, dtype=float)[0])
    adapt_ground_energy = float(np.asarray(energies, dtype=float)[-1])

    shot_results = []
    for shot in qsceom_shots:
        values = []
        times = []
        print(
            f"Running QSC-EOM shots={shot} for {args.runs_per_shot} repeats...",
            flush=True,
        )
        for run_idx in range(args.runs_per_shot):
            run_start = time.time()
            eigvals, _ = qsc_eom(
                symbols=symbols,
                coordinates=geometry,
                active_electrons=args.active_electrons,
                active_orbitals=args.active_orbitals,
                charge=args.charge,
                params=params,
                ash_excitation=ash_excitation,
                shots=shot,
                basis=args.basis,
            )
            elapsed = time.time() - run_start
            first_eig = float(np.asarray(eigvals, dtype=float)[0])
            values.append(first_eig)
            times.append(elapsed)
            print(
                f"  run {run_idx + 1:02d}/{args.runs_per_shot}: "
                f"first_eig={first_eig:.10f}, time={elapsed:.2f}s",
                flush=True,
            )

        arr = np.asarray(values, dtype=float)
        shot_results.append(
            {
                "shots": int(shot),
                "runs": int(args.runs_per_shot),
                "first_eig_values": [float(v) for v in arr.tolist()],
                "mean_first_eig": float(arr.mean()),
                "variance_first_eig": float(arr.var(ddof=1)) if len(arr) > 1 else 0.0,
                "stddev_first_eig": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "mean_runtime_s": float(np.asarray(times, dtype=float).mean()),
            }
        )

    means = [entry["mean_first_eig"] for entry in shot_results]
    variances = [entry["variance_first_eig"] for entry in shot_results]
    _plot_stats(
        shots=qsceom_shots,
        means=means,
        variances=variances,
        no_shot_value=baseline_first,
        plot_path=plot_path,
    )

    report_lines = [
        "NH3 QSC-EOM First-Eigenvalue Shot Statistics",
        "",
        "Input:",
        f"  symbols: {symbols}",
        f"  geometry (angstrom): {geometry}",
        f"  basis: {args.basis}",
        f"  charge: {args.charge}",
        f"  spin (2S): {args.spin}",
        f"  active_electrons: {args.active_electrons}",
        f"  active_orbitals: {args.active_orbitals}",
        f"  adapt_it: {args.adapt_it}",
        f"  adapt_shots: {0 if adapt_shots is None else int(adapt_shots)}",
        f"  qsceom_shots: {qsceom_shots}",
        f"  runs_per_shot: {args.runs_per_shot}",
        "",
        "ADAPT-VQE:",
        f"  runtime_s: {float(adapt_time_s):.6f}",
        f"  num_params: {int(len(params))}",
        f"  num_excitations: {int(len(ash_excitation))}",
        f"  adapt_ground_energy_hartree: {adapt_ground_energy:.12f}",
        "",
        "No-shot QSC-EOM baseline:",
        f"  runtime_s: {float(baseline_time_s):.6f}",
        f"  first_eig: {baseline_first:.12f}",
        "",
        "Finite-shot statistics (first eigenvalue):",
    ]

    for entry in shot_results:
        report_lines.append(
            "  shots={shots}: mean={mean:.12f}, variance={var:.12e}, stddev={std:.12e}, "
            "mean_runtime_s={rt:.6f}".format(
                shots=entry["shots"],
                mean=entry["mean_first_eig"],
                var=entry["variance_first_eig"],
                std=entry["stddev_first_eig"],
                rt=entry["mean_runtime_s"],
            )
        )
        report_lines.append(
            f"    first_eig_values: {entry['first_eig_values']}"
        )

    report_lines.extend(
        [
            "",
            f"Plot file: {plot_path}",
            "",
        ]
    )
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nSummary (first eigenvalue only):", flush=True)
    print(f"  ADAPT ground energy: {adapt_ground_energy:.10f}", flush=True)
    print(f"  no-shot baseline: {baseline_first:.10f}", flush=True)
    for entry in shot_results:
        print(
            "  shots={shots}: mean={mean:.10f}, variance={var:.10e}, stddev={std:.10e}".format(
                shots=entry["shots"],
                mean=entry["mean_first_eig"],
                var=entry["variance_first_eig"],
                std=entry["stddev_first_eig"],
            ),
            flush=True,
        )
    print(f"\nWrote stats TXT: {output_txt}", flush=True)
    print(f"Wrote plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
