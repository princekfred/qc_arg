#!/usr/bin/env python3
"""Run H2 ground-state ADAPT-VQE at 0.735 Angstrom by default."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_adapt_vqe():
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="H2 ground-state ADAPT-VQE example (default bond length: 0.735 A)."
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=0.735,
        help="H-H bond length in Angstrom.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="Basis set used in the calculation.",
    )
    parser.add_argument(
        "--adapt-it",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="One or more ADAPT iteration counts to run (e.g. --adapt-it 2 3 4).",
    )
    parser.add_argument(
        "--active-electrons",
        type=int,
        default=2,
        help="Number of active electrons.",
    )
    parser.add_argument(
        "--active-orbitals",
        type=int,
        default=2,
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
        help="2S spin value used by PySCF (0 for singlet).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Measurement shots (0 means analytic mode).",
    )
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=2000,
        help="Maximum optimizer iterations per ADAPT step.",
    )
    parser.add_argument(
        "--fci-nroots",
        type=int,
        default=4,
        help="Number of FCI roots to compute for the reference section.",
    )
    parser.add_argument(
        "--skip-fci",
        action="store_true",
        help="Skip computing/printing the FCI reference section.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output report file path.",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Optional metrics plot image path (default: h2_error_plot.png).",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    return parser.parse_args(argv)


def _compute_fci(symbols, geometry, basis, charge, spin, nroots):
    try:
        import numpy as np
        from pyscf import fci, gto, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FCI section requires PySCF and NumPy. Install with: "
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
    cisolver.nroots = int(nroots)
    energies, _ = cisolver.kernel()
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    return energies


def _plot_metrics(
    iterations,
    adapt_errors,
    qsceom_errors,
    adapt_max_gradients,
    plot_path,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install with: `pip install matplotlib`."
        ) from exc

    def _finite_xy(xs, ys):
        x_out = []
        y_out = []
        for x, y in zip(xs, ys):
            if y == y:  # NaN-safe finite check for this use case.
                x_out.append(x)
                y_out.append(y)
        return x_out, y_out

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("H2 ADAPT/QSC-EOM Metrics vs ADAPT Iteration")
    ax.set_xlabel("ADAPT iterations")
    ax.set_ylabel("Error / Gradient (Hartree)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)

    lines = []
    labels = []

    x, y = _finite_xy(iterations, adapt_errors)
    if y:
        ln = ax.plot(x, y, marker="o", label="ADAPT gr error")[0]
        lines.append(ln)
        labels.append(ln.get_label())

    x, y = _finite_xy(iterations, qsceom_errors)
    if y:
        ln = ax.plot(x, y, marker="o", label="qsceom gr error")[0]
        lines.append(ln)
        labels.append(ln.get_label())

    x, y = _finite_xy(iterations, adapt_max_gradients)
    if y:
        ln = ax.plot(x, y, marker="^", label="ADAPT max gradient")[0]
        lines.append(ln)
        labels.append(ln.get_label())

    if lines:
        ax.legend(lines, labels, loc="best")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    if args.fci_nroots <= 0:
        raise ValueError("--fci-nroots must be > 0")
    adapt_vqe, qsc_eom = _load_adapt_vqe()

    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, args.bond_length]]

    fci_energies = None
    fci_ground = None
    if not args.skip_fci:
        fci_energies = _compute_fci(
            symbols=symbols,
            geometry=geometry,
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
            nroots=args.fci_nroots,
        )
        fci_ground = float(fci_energies[0])

    shots = None if args.shots == 0 else args.shots
    reports = ["===== H2 Ground-State ADAPT-VQE ====="]
    iterations = []
    adapt_errors = []
    qsceom_errors = []
    adapt_max_gradients = []
    for adapt_it in args.adapt_it:
        params, ash_excitation, energies, adapt_gradients, adapt_hamiltonian, adapt_qubits = adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=adapt_it,
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            shots=shots,
            optimizer_maxiter=args.optimizer_maxiter,
            return_max_gradients=True,
            return_hamiltonian=True,
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
            hamiltonian=adapt_hamiltonian,
            qubits=adapt_qubits,
        )

        adapt_ground = float(energies[-1])
        qsc_ground = float(eigvals[0])
        lines = [
            f"Bond length (Angstrom): {args.bond_length}",
            f"Basis: {args.basis}",
            f"ADAPT iterations: {adapt_it}",
            f"Adapt gr energy (Hartree): {adapt_ground}",
            f"qsceom gr energy (Hartree): {qsc_ground}",
        ]
        lines.extend(
            f"qsceom excited state {state_idx} energy (Hartree): {float(state_energy)}"
            for state_idx, state_energy in enumerate(eigvals[1:], start=1)
        )
        if len(adapt_gradients) > 0:
            lines.append(
                f"ADAPT max gradient: {adapt_gradients[-1]}"
            )
            adapt_max_gradients.append(float(adapt_gradients[-1]))
        else:
            adapt_max_gradients.append(float("nan"))
        if fci_ground is not None:
            adapt_err = abs(adapt_ground - fci_ground)
            qsceom_err = abs(qsc_ground - fci_ground)
            lines.extend(
                [
                    f"ADAPT gr error (Hartree): {adapt_err}",
                    f"qsceom gr error (Hartree): {qsceom_err}",
                ]
            )
            adapt_errors.append(float(adapt_err))
            qsceom_errors.append(float(qsceom_err))
        else:
            adapt_errors.append(float("nan"))
            qsceom_errors.append(float("nan"))
        iterations.append(int(adapt_it))
        reports.append("\n".join(lines))

    if not args.skip_fci:
        fci_lines = [
            "===== FCI Reference =====",
            f"Bond length (Angstrom): {args.bond_length}",
            f"Basis: {args.basis}",
            f"Charge: {args.charge}",
            f"Spin (2S): {args.spin}",
            f"Requested FCI roots: {args.fci_nroots}",
            f"FCI gr energy (Hartree): {float(fci_energies[0])}",
        ]
        fci_lines.extend(
            f"FCI excited state {state_idx} energy (Hartree): {float(state_energy)}"
            for state_idx, state_energy in enumerate(fci_energies[1:], start=1)
        )
        reports.append("\n".join(fci_lines))

    report = "\n\n".join(reports) + "\n"
    print(report, end="")

    if args.output_file is None:
        output_path = Path(__file__).resolve().parent / "h2_ground_output.txt"
    else:
        output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    if args.plot_file is None:
        plot_path = output_path.with_name("h2_error_plot.png")
    else:
        plot_path = Path(args.plot_file)
    _plot_metrics(
        iterations=iterations,
        adapt_errors=adapt_errors,
        qsceom_errors=qsceom_errors,
        adapt_max_gradients=adapt_max_gradients,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()
