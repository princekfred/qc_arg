#!/usr/bin/env python3
"""Run NH3 QSE shot study and plot UCCSD baseline + QSE mean/variance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qse import run_qse


def plot_results(shots, means, variances, uccsd_no_shot_energy, plot_path):
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
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
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
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }

    with mpl.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        ax.axhline(
            float(uccsd_no_shot_energy),
            color="black",
            linestyle="-",
            linewidth=1.8,
            label="UCCSD (no shots)",
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color="blue",
            ecolor="blue",
            elinewidth=1.4,
            capsize=4,
            #label="QSE mean ± 1σ",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Shots")
        ax.set_ylabel("Energy (Ha)")
        ax.set_xticks(x)
        shot_labels = []
        for v in x:
            iv = int(v)
            if iv == 1000:
                shot_labels.append(r"$10^3$")
            elif iv == 10000:
                shot_labels.append(r"$10^4$")
            else:
                shot_labels.append(str(iv))
        ax.set_xticklabels(shot_labels)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        #ax.legend(loc="best", handlelength=2.6)

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main():
    num_runs = 10
    shot_values = [500, 1000, 10000]
    seed0 = 170

    num_spartial_orbital = 4
    active_electrons = 4
    atom = (
        "N 0.0 0.0 0.0; "
        "H 2.526315789473684 0.0 0.0; "
        "H -0.506 0.876 0.0; "
        "H -0.506 -0.876 0.0"
    )

    # First run without shots (analytic baseline).
    # UCCSD is always analytic (no shots); only QSE uses `shots`.
    baseline = run_qse(
        atom=atom,
        basis="sto-6g",
        active_electrons=active_electrons,
        active_orbitals=num_spartial_orbital,
        seed=seed0,
        shots=0,
        uccsd_shots=0,
        use_aer_estimator=True,
    )
    hf_no_shot_energy = float(baseline["hf_energy"])
    uccsd_no_shot_energy = float(baseline["ground_energy"])
    qse_no_shot_first = float(baseline["qse_eigenvalues"][0])
    output_path = Path(__file__).resolve().with_name("nh3_qse_shots.txt")
    print(f"UCCSD energy (Ha): {uccsd_no_shot_energy:.12f}")
    print(f"qse gr (Ha): {qse_no_shot_first:.12f}")

    means = []
    variances = []
    for shot in shot_values:
        if int(shot) <= 0:
            continue
        values = []
        for run_idx in range(num_runs):
            run_seed = seed0 + run_idx
            results = run_qse(
                atom=atom,
                basis="sto-6g",
                active_electrons=active_electrons,
                active_orbitals=num_spartial_orbital,
                seed=run_seed,
                shots=int(shot),
                uccsd_shots=0,
                use_aer_estimator=True,
            )
            qse_gr = float(results["qse_eigenvalues"][0])
            values.append(qse_gr)
            print(
                f"shots={int(shot)} run={run_idx + 1}/{num_runs}: "
                f"qse_gr_eig={qse_gr:.12f} Ha"
            )

        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        variance = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
        means.append(mean)
        variances.append(variance)
        print(
            f"shots={int(shot)}: mean={mean:.12f} Ha, variance={variance:.12e}"
        )

    plot_path = Path(__file__).resolve().with_name("nh3_qse_shots_plot.png")
    plot_results(
        shots=shot_values,
        means=means,
        variances=variances,
        uccsd_no_shot_energy=uccsd_no_shot_energy,
        plot_path=plot_path,
    )
    print(
        f"QSE gr energy (Ha): {qse_no_shot_first:.12f}"
    )
    lines = [
        "NH3 QSE shot study",
        f"HF energy (Ha): {hf_no_shot_energy:.12f}",
        f"UCCSD energy (Ha, no shots): {uccsd_no_shot_energy:.12f}",
        f"QSE first eigenvalue (Ha, no shots): {qse_no_shot_first:.12f}",
        "",
        f"shot_values: {shot_values}",
        f"runs_per_shot: {num_runs}",
        "",
    ]
    for shot, mean, variance in zip(shot_values, means, variances):
        lines.append(
            f"shots={int(shot)}: mean={float(mean):.12f} Ha, variance={float(variance):.12e}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TXT written to: {output_path}")
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
