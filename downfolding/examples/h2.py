#!/usr/bin/env python3
"""Run the H2 ground-state ADAPT-VQE example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_adapt_vqe():
    script_path = Path(__file__).resolve()
    downfolding_dir = script_path.parents[1]
    if str(downfolding_dir) not in sys.path:
        sys.path.insert(0, str(downfolding_dir))

    try:
        from adaptvqe import adapt_vqe
        from qsceom_par import qsc_eom
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not import ADAPT-VQE/QSC-EOM modules from {downfolding_dir}."
        ) from exc
    return adapt_vqe, qsc_eom

def _build_h2_molecule(bond_length):
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]]
    return symbols, geometry

def build_parser():
    parser = argparse.ArgumentParser(
        description="H2 ground-state ADAPT-VQE example."
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
        default=[4],
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
        help=(
            "Total FCI roots including the ground state; N roots provide up to "
            "N-1 excited states."
        ),
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


def _format_excited_state_table(qsceom_states, fci_states):
    headers = ["S/N", "State", "QSC-EOM (Ha)", "FCI (Ha)"]
    state_count = max(len(qsceom_states), len(fci_states))
    rows = []
    for state_offset in range(state_count):
        qsceom_energy = (
            f"{qsceom_states[state_offset]:.10f}"
            if state_offset < len(qsceom_states)
            else "N/A"
        )
        fci_energy = (
            f"{fci_states[state_offset]:.10f}"
            if state_offset < len(fci_states)
            else "N/A"
        )
        rows.append(
            [
                str(state_offset + 1),
                f"Excited state {state_offset + 1}",
                qsceom_energy,
                fci_energy,
            ]
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        if rows
        else len(headers[column])
        for column in range(len(headers))
    ]

    def border(left, junction, right):
        return left + junction.join("-" * (width + 2) for width in widths) + right

    def format_row(cells, header=False):
        formatted = []
        for column, cell in enumerate(cells):
            if header:
                formatted.append(cell.center(widths[column]))
            elif column < 2:
                formatted.append(cell.ljust(widths[column]))
            else:
                formatted.append(cell.rjust(widths[column]))
        return "| " + " | ".join(formatted) + " |"

    return [
        border("+", "+", "+"),
        format_row(headers, header=True),
        border("+", "+", "+"),
        *(format_row(row) for row in rows),
        border("+", "+", "+"),
    ]


def main():
    args = parse_args()
    if args.fci_nroots <= 0:
        raise ValueError("--fci-nroots must be > 0")
    adapt_vqe, qsc_eom = _load_adapt_vqe()

    basis = args.basis
    bond_length = args.bond_length
    symbols, geometry = _build_h2_molecule(bond_length)

    fci_energies = None
    if not args.skip_fci:
        fci_energies = _compute_fci(
            symbols=symbols,
            geometry=geometry,
            basis=basis,
            charge=args.charge,
            spin=args.spin,
            nroots=args.fci_nroots,
        )

    shots = None if args.shots == 0 else args.shots
    fci_summary = (
        f"FCI roots (incl. ground): {len(fci_energies)} | "
        f"FCI excited states: {len(fci_energies) - 1} | "
        f"FCI ground: {float(fci_energies[0]):.10f} Ha"
        if fci_energies is not None
        else "FCI: skipped"
    )
    reports = [
        "# H2 Energy Report",
        (
            f"Bond: {bond_length:g} Angstrom | Basis: {basis} | "
            f"Active space: {args.active_electrons}e/{args.active_orbitals}o"
        ),
        (
            f"Charge: {args.charge} | Spin (2S): {args.spin} | "
            f"ADAPT iterations: {', '.join(str(value) for value in args.adapt_it)} | "
            f"{fci_summary}"
        ),
        "",
    ]
    for adapt_it in args.adapt_it:
        params, ash_excitation, energies, adapt_hamiltonian, adapt_qubits = adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=adapt_it,
            basis=basis,
            charge=args.charge,
            spin=args.spin,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            shots=shots,
            optimizer_maxiter=args.optimizer_maxiter,
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
            basis=basis,
            hamiltonian=adapt_hamiltonian,
            qubits=adapt_qubits,
        )

        adapt_ground = float(energies[-1])
        qsc_ground = float(eigvals[0])
        qsceom_states = [float(state_energy) for state_energy in eigvals[1:]]
        fci_states = (
            [float(state_energy) for state_energy in fci_energies[1:]]
            if fci_energies is not None
            else []
        )
        reports.extend(
            [
                f"### ADAPT iteration {adapt_it}",
                (
                    f"ADAPT ground: {adapt_ground:.10f} Ha | "
                    f"QSC-EOM ground: {qsc_ground:.10f} Ha"
                ),
            ]
        )
        reports.extend(_format_excited_state_table(qsceom_states, fci_states))
        reports.append("")

    report = "\n".join(reports).rstrip() + "\n"
    print(report, end="")

    if args.output_file is None:
        output_path = Path(__file__).resolve().parents[1] / "output" / "h2_output.txt"
    else:
        output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
