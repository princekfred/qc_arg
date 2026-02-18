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


def main():
    args = parse_args()
    if args.fci_nroots <= 0:
        raise ValueError("--fci-nroots must be > 0")
    adapt_vqe, qsc_eom = _load_adapt_vqe()

    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, args.bond_length]]

    shots = None if args.shots == 0 else args.shots
    reports = []
    for adapt_it in args.adapt_it:
        params, ash_excitation, energies = adapt_vqe(
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

        lines = [
            "===== H2 Ground-State ADAPT-VQE =====",
            f"Bond length (Angstrom): {args.bond_length}",
            f"Basis: {args.basis}",
            f"ADAPT iterations: {adapt_it}",
            f"Adapt gr energy (Hartree): {energies[-1]}",
            f"QSC-EOM gr energy (Hartree): {eigvals[0]}",
        ]
        reports.append("\n".join(lines))

    if not args.skip_fci:
        fci_energies = _compute_fci(
            symbols=symbols,
            geometry=geometry,
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
            nroots=args.fci_nroots,
        )
        fci_lines = [
            "===== FCI Reference =====",
            f"Bond length (Angstrom): {args.bond_length}",
            f"Basis: {args.basis}",
            f"Charge: {args.charge}",
            f"Spin (2S): {args.spin}",
            f"Requested FCI roots: {args.fci_nroots}",
            #f"FCI energies (Hartree): {fci_energies}",
            f"FCI gr energy (Hartree): {fci_energies[0]}",
        ]
        reports.append("\n".join(fci_lines))

    report = "\n\n".join(reports) + "\n"
    print(report, end="")

    if args.output_file is None:
        output_path = Path(__file__).resolve().parent / "h2_ground_output.txt"
    else:
        output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
