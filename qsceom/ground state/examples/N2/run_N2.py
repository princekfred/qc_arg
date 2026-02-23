#!/usr/bin/env python3
"""Simple N2 potential-energy scan with ADAPT-VQE, QSC-EOM, and FCI.

Geometry at each point:
    [[0.0, 0.0, d],
     [0.0, 0.0, -0.5488]]
with d sampled from 0.4 to 2.0 Angstrom over 30 points (default).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def load_modules():
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


def fci_ground_energy(symbols, geometry, basis, charge, spin):
    try:
        import numpy as np
        from pyscf import fci, gto, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FCI requires NumPy and PySCF. Install with: `pip install numpy pyscf`."
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


def plot_results(d, adapt, qsceom, fci, err_adapt, err_qsceom, plot_path):
    try:
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires NumPy and matplotlib. Install with: "
            "`pip install numpy matplotlib`."
        ) from exc

    x = np.asarray(d, dtype=float)
    adapt = np.asarray(adapt, dtype=float)
    qsceom = np.asarray(qsceom, dtype=float)
    fci = np.asarray(fci, dtype=float)
    err_adapt = np.clip(np.asarray(err_adapt, dtype=float), 1e-16, None)
    err_qsceom = np.clip(np.asarray(err_qsceom, dtype=float), 1e-16, None)

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
        "legend.fontsize": 9,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }

    with mpl.rc_context(style_params):
        fig, (ax_energy, ax_err) = plt.subplots(
            2, 1, figsize=(7.0, 7.0), sharex=True, constrained_layout=True
        )

        # Energy curves.
        ax_energy.plot(x, adapt, "o-", color="black", label="ADAPT-VQE")
        ax_energy.plot(x, qsceom, "v-", color="blue", label="QSC-EOM first")
        ax_energy.plot(x, fci, "1-", color="red", label="FCI")
        ax_energy.set_ylabel("Energy (Hartree)")
        ax_energy.grid(True, which="both", alpha=0.25)
        ax_energy.legend(loc="best")

        # Error from FCI (log scale).
        ax_err.plot(x, err_adapt, "o-", color="black", label="|ADAPT - FCI|")
        ax_err.plot(x, err_qsceom, "v-", color="blue", label="|QSC-EOM - FCI|")
        ax_err.set_yscale("log")
        ax_err.set_xlabel("Bond Distance (Å)")
        ax_err.set_ylabel("Error from FCI (Hartree)")
        ax_err.grid(True, which="both", alpha=0.25)
        ax_err.legend(loc="best")

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="N2 potential-energy scan with ADAPT-VQE, QSC-EOM, and FCI."
    )
    parser.add_argument("--d-min", type=float, default=0.4, help="Minimum d (Angstrom).")
    parser.add_argument("--d-max", type=float, default=2.0, help="Maximum d (Angstrom).")
    parser.add_argument("--num-points", type=int, default=30, help="Number of d points.")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Basis set.")
    parser.add_argument("--adapt-it", type=int, default=6, help="ADAPT iterations.")
    parser.add_argument("--active-electrons", type=int, default=4, help="Active electrons.")
    parser.add_argument("--active-orbitals", type=int, default=4, help="Active orbitals.")
    parser.add_argument("--charge", type=int, default=0, help="Total charge.")
    parser.add_argument("--spin", type=int, default=0, help="2S spin value.")
    parser.add_argument("--shots", type=int, default=0, help="QSC-EOM shots (0 = analytic).")
    parser.add_argument(
        "--optimizer-maxiter", type=int, default=500, help="ADAPT optimizer max iterations."
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
        help="Optional output plot PNG path.",
    )
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
        raise ImportError("This script requires NumPy.") from exc

    adapt_vqe, qsc_eom = load_modules()
    symbols = ["N", "N"]
    d_vals = np.linspace(float(args.d_min), float(args.d_max), int(args.num_points))
    adapt_shots = None if args.shots == 0 else int(args.shots)

    adapt_energies = []
    qsceom_energies = []
    fci_energies = []
    adapt_errors = []
    qsceom_errors = []
    report_lines = ["===== N2 Potential Energy Scan ====="]

    t_start = time.time()
    for i, d_val in enumerate(d_vals):
        geometry = [[0.0, 0.0, float(d_val)], [0.0, 0.0, -0.5488]]
        print(
            f"[{i + 1:02d}/{len(d_vals):02d}] d={d_val:.6f} Angstrom: running ADAPT/QSC-EOM/FCI...",
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
            shots=adapt_shots,
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
        e_fci = fci_ground_energy(
            symbols=symbols,
            geometry=geometry,
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
        )

        e_adapt = float(np.asarray(energies, dtype=float)[-1])
        e_qsc = float(np.asarray(eigvals, dtype=float)[0])
        err_a = abs(e_adapt - e_fci)
        err_q = abs(e_qsc - e_fci)

        adapt_energies.append(e_adapt)
        qsceom_energies.append(e_qsc)
        fci_energies.append(e_fci)
        adapt_errors.append(err_a)
        qsceom_errors.append(err_q)

        report_lines.append(
            (
                "d={d:.6f} Angstrom | ADAPT={ea:.12f} | QSC-EOM={eq:.12f} | "
                "FCI={ef:.12f} | |ADAPT-FCI|={da:.6e} | |QSC-EOM-FCI|={dq:.6e}"
            ).format(d=float(d_val), ea=e_adapt, eq=e_qsc, ef=e_fci, da=err_a, dq=err_q)
        )

    elapsed = time.time() - t_start
    report_lines.extend(
        [
            "",
            "Summary:",
            f"points: {len(d_vals)}",
            f"d range (Angstrom): [{float(d_vals[0]):.6f}, {float(d_vals[-1]):.6f}]",
            f"basis: {args.basis}",
            f"adapt_it: {args.adapt_it}",
            f"active_electrons: {args.active_electrons}",
            f"active_orbitals: {args.active_orbitals}",
            f"shots: {args.shots}",
            f"total_runtime_s: {elapsed:.3f}",
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

    plot_results(
        d_vals,
        adapt_energies,
        qsceom_energies,
        fci_energies,
        adapt_errors,
        qsceom_errors,
        plot_path,
    )

    report_lines.extend([f"plot_file: {plot_path}", ""])
    report = "\n".join(report_lines)
    print("\n" + report, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
