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


def parse_args():
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
        default=3,
        help="Number of ADAPT iterations.",
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
        "--output-file",
        type=str,
        default=None,
        help="Optional output report file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapt_vqe, qsc_eom = _load_adapt_vqe()

    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, args.bond_length]]

    shots = None if args.shots == 0 else args.shots
    params, ash_excitation, energies = adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=args.adapt_it,
        basis=args.basis,
        charge=args.charge,
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
        f"ADAPT iterations: {args.adapt_it}",
        f"Ground-state energy (Hartree): {energies[-1]}",
        f"QSC-EOM eigenvalues (Hartree): {eigvals}",
    ]
    report = "\n".join(lines) + "\n"
    print(report, end="")

    if args.output_file is None:
        output_path = Path(__file__).resolve().parent / "h2_ground_output.txt"
    else:
        output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
