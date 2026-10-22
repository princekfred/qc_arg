#!/usr/bin/env python3
"""Compute FCI reference energies for arbitrary molecules with PySCF."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="General FCI reference runner for user-provided molecular geometries."
    )
    parser.add_argument(
        "--atom",
        action="append",
        nargs=4,
        metavar=("SYMBOL", "X", "Y", "Z"),
        default=None,
        help=(
            "Atom entry as SYMBOL X Y Z. Repeat this argument per atom, "
            "for example: --atom H 0 0 0 --atom H 0 0 0.735"
        ),
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="Coordinate unit passed to PySCF (default: angstrom).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="Basis set used in the calculation.",
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
        "--nroots",
        type=int,
        default=4,
        help="Number of FCI roots (states) to compute.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output report file path.",
    )
    return parser.parse_args()


def _parse_atoms(atom_args):
    if atom_args is None:
        # Default to the H2 demo geometry if no explicit atoms are provided.
        return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))]

    atoms = []
    for entry in atom_args:
        symbol, x, y, z = entry
        atoms.append((symbol, (float(x), float(y), float(z))))
    if len(atoms) == 0:
        raise ValueError("At least one --atom entry is required.")
    return atoms


def main():
    args = parse_args()

    if args.nroots <= 0:
        raise ValueError("--nroots must be > 0")

    try:
        import numpy as np
        from pyscf import fci, gto, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "run_fci.py requires PySCF and NumPy. Install with: "
            "`pip install numpy pyscf`."
        ) from exc

    atoms = _parse_atoms(args.atom)

    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = args.unit
    mol.basis = args.basis
    mol.charge = args.charge
    mol.spin = args.spin
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
    cisolver.nroots = int(args.nroots)
    energies, _ = cisolver.kernel()
    energies = np.atleast_1d(np.asarray(energies, dtype=float))

    atom_lines = [f"{sym} {xyz[0]} {xyz[1]} {xyz[2]}" for sym, xyz in atoms]
    lines = [
        "===== FCI Reference =====",
        f"Unit: {args.unit}",
        f"Basis: {args.basis}",
        f"Charge: {args.charge}",
        f"Spin (2S): {args.spin}",
        f"Requested FCI roots: {args.nroots}",
        "Atoms:",
        *atom_lines,
        f"FCI energies (Hartree): {energies}",
        f"Ground-state FCI energy (Hartree): {energies[0]}",
    ]
    report = "\n".join(lines) + "\n"
    print(report, end="")

    if args.output_file is None:
        output_path = Path(__file__).resolve().parent / "fci_output.txt"
    else:
        output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
