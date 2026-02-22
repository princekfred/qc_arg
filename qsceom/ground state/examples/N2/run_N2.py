#!/usr/bin/env python3
"""Potential-energy scan for N2 with ADAPT-VQE, QSC-EOM, and FCI.

Scan geometry:
    [[0.0, 0.0, d],
     [0.0, 0.0, -0.5488]]
where d is sampled from 0.4 to 2.0 Angstrom over 30 points by default.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


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
            "Could not import ground-state modules from qsceom/ground state."
        ) from exc
    return adapt_vqe, qsc_eom


def _compute_fci_ground(symbols, geometry, basis, charge, spin):
    try:
        import numpy as np
        from pyscf import fci, gto, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FCI calculation requires NumPy and PySCF. Install with: "
            "`pip install numpy pyscf`."
        ) from exc

    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(len(symbols))]
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.build()

    if mol.spin == 0 and mol.nelectron % 2 == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.level_shift = 0.5
    mf.diis_space = 12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run()

    cisolver = fci.FCI(mf)
    e0, _ = cisolver.kernel(nroots=1)
    return float(np.atleast_1d(np.asarray(e0, dtype=float))[0])


def _plot_curves(
    d_values,
    adapt_energies,
    qsceom_energies,
    fci_energies,
    adapt_errors,
    qsceom_errors,
    plot_path,
):
    try:
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires NumPy and matplotlib. Install with: "
            "`pip install numpy matplotlib`."
        ) from exc

    x = np.asarray(d_values, dtype=float)
    adapt = np.asarray(adapt_energies, dtype=float)
    qsc = np.asarray(qsceom_energies, dtype=float)
    fci = np.asarray(fci_energies, dtype=float)
    err_adapt = np.asarray(adapt_errors, dtype=float)
    err_qsc = np.asarray(qsceom_errors, dtype=float)

    # Avoid non-positive values on a log axis.
    log_floor = 1e-16
    err_adapt = np.clip(err_adapt, log_floor, None)
    err_qsc = np.clip(err_qsc, log_floor, None)

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
        fig, (ax_e, ax_err) = plt.subplots(
            2,
            1,
            figsize=(7.0, 7.0),
            sharex=True,
            constrained_layout=True,
        )

        ax_e.plot(x, adapt, color="black", marker="o", linestyle="-", label="ADAPT-VQE")
        ax_e.plot(x, qsc, color="blue", marker="v", linestyle="--", label="q-sc-EOM")
        ax_e.plot(x, fci, color="red", marker="1", linestyle="-.", label="FCI")
        ax_e.set_ylabel("Energy (Ha)")
        ax_e.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax_e.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        ax_e.legend(loc="best", handlelength=2.6)

        ax_err.plot(
            x,
            err_adapt,
            color="black",
            marker="o",
            linestyle="-",
            label="|ADAPT - FCI|",
        )
        ax_err.plot(
            x,
            err_qsc,
            color="blue",
            marker="v",
            linestyle="-",
            label="|QSC-EOM - FCI|",
        )
        ax_err.set_yscale("log")
        ax_err.set_xlabel("d (Ä)")
        ax_err.set_ylabel("Error from FCI (Ha)")
        ax_err.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax_err.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        ax_err.legend(loc="best", handlelength=2.6)

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="N2 potential-energy scan with ADAPT-VQE, QSC-EOM, and FCI."
    )
    parser.add_argument(
        "--d-min",
        type=float,
        default=0.4,
        help="Minimum d value in Angstrom.",
    )
    parser.add_argument(
        "--d-max",
        type=float,
        default=2.0,
        help="Maximum d value in Angstrom.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=30,
        help="Number of scan points between d-min and d-max.",
    )
   
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-6g",
        help="Basis set used in ADAPT-VQE, QSC-EOM, and FCI.",
    )
    parser.add_argument(
        "--adapt-it",
        type=int,
        default=6,
        help="Number of ADAPT iterations used at each scan point.",
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
        "--charge",
        type=int,
        default=0,
        help="Total molecular charge.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2S spin value used by PySCF.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="QSC-EOM shot count (0 means analytic mode).",
    )
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=500,
        help="Maximum optimizer iterations per ADAPT step.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output report TXT path.",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Optional output plot path (PNG).",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.num_points <= 1:
        raise ValueError("--num-points must be > 1")
    if args.adapt_it <= 0:
        raise ValueError("--adapt-it must be > 0")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("NumPy is required for this script.") from exc

    adapt_vqe, qsc_eom = _load_ground_modules()
    symbols = ["N", "N"]
    d_values = np.linspace(float(args.d_min), float(args.d_max), int(args.num_points))
    shots_adapt = None if args.shots == 0 else int(args.shots)

    adapt_energies = []
    qsceom_energies = []
    fci_energies = []
    adapt_errors = []
    qsceom_errors = []

    report_lines = ["===== N2 Potential Energy Scan ====="]
    start_all = time.time()
    for i, d in enumerate(d_values):
        geometry = [[0.0, 0.0, float(d)], [0.0, 0.0, float(-0.5488)]]
        print(
            f"[{i + 1:02d}/{len(d_values):02d}] d={d:.6f} Angstrom: running ADAPT/QSC-EOM/FCI...",
            flush=True,
        )

        params, ash_excitation, energies = adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=int(args.adapt_it),
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            shots=shots_adapt,
            optimizer_maxiter=args.optimizer_maxiter,
        )
        eigvals, _ = qsc_eom(
            symbols=symbols,
            coordinates=geometry,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            charge=args.charge,
            params=params,
            ash_excitation=ash_excitation,
            shots=args.shots,
            basis=args.basis,
        )
        fci_ground = _compute_fci_ground(
            symbols=symbols,
            geometry=geometry,
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
        )

        adapt_ground = float(np.asarray(energies, dtype=float)[-1])
        qsceom_ground = float(np.asarray(eigvals, dtype=float)[0])
        adapt_err = abs(adapt_ground - fci_ground)
        qsceom_err = abs(qsceom_ground - fci_ground)

        adapt_energies.append(adapt_ground)
        qsceom_energies.append(qsceom_ground)
        fci_energies.append(float(fci_ground))
        adapt_errors.append(float(adapt_err))
        qsceom_errors.append(float(qsceom_err))

        report_lines.append(
            "d={d:.6f} Angstrom | ADAPT={adapt:.12f} | QSC-EOM={qsc:.12f} | "
            "FCI={fci:.12f} | |ADAPT-FCI|={ea:.6e} | |QSC-EOM-FCI|={eq:.6e}".format(
                d=float(d),
                adapt=adapt_ground,
                qsc=qsceom_ground,
                fci=float(fci_ground),
                ea=adapt_err,
                eq=qsceom_err,
            )
        )

    elapsed_all = time.time() - start_all
    report_lines.extend(
        [
            "",
            "Summary:",
            f"points: {len(d_values)}",
            f"d range (Angstrom): [{float(d_values[0]):.6f}, {float(d_values[-1]):.6f}]",
            f"basis: {args.basis}",
            f"adapt_it: {args.adapt_it}",
            f"active_electrons: {args.active_electrons}",
            f"active_orbitals: {args.active_orbitals}",
            f"shots: {args.shots}",
            f"total_runtime_s: {elapsed_all:.3f}",
        ]
    )

    if args.output_file is None:
        output_path = Path(__file__).resolve().parent / "n2_potential_output.txt"
    else:
        output_path = Path(args.output_file)
    if args.plot_file is None:
        plot_path = output_path.with_name("n2_potential_plot.png")
    else:
        plot_path = Path(args.plot_file)

    _plot_curves(
        d_values=d_values,
        adapt_energies=adapt_energies,
        qsceom_energies=qsceom_energies,
        fci_energies=fci_energies,
        adapt_errors=adapt_errors,
        qsceom_errors=qsceom_errors,
        plot_path=plot_path,
    )

    report_lines.extend(
        [
            f"plot_file: {plot_path}",
            "",
        ]
    )
    report = "\n".join(report_lines)
    print("\n" + report, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

