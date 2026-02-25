#!/usr/bin/env python
"""Run H2 with selectable ground-state method, then QSC-EOM.

Example
-------
python "qsceom/excited state/examples/H2/run_h2.py" --ground-method uccsd --basis sto-3g
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make imports robust for both layouts:
# 1) package layout: qsceom/UCCSD.py
# 2) local layout: qsceom/excited state/UCCSD.py
script_path = Path(__file__).resolve()
repo_root = script_path.parents[3]
package_dir = repo_root / "qsceom"

cleaned_path = []
for entry in sys.path:
    try:
        if Path(entry).resolve() == package_dir:
            continue
    except Exception:
        pass
    cleaned_path.append(entry)
sys.path[:] = cleaned_path

if str(repo_root) in sys.path:
    sys.path.remove(str(repo_root))
sys.path.insert(0, str(repo_root))

try:
    from qsceom.UCCSD import gs_exact
    from qsceom.adaptvqe import adapt_vqe
    from qsceom.qsceom import qsceom
except ModuleNotFoundError:
    module_root = script_path.parents[2]
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))
    from UCCSD import gs_exact
    from adaptvqe import adapt_vqe
    from qsceom import qsceom


def parse_args():
    """Build command-line arguments for the H2 example runner."""
    parser = argparse.ArgumentParser(
        description="H2 example: UCCSD/ADAPT-VQE ground state + QSC-EOM excited states."
    )
    parser.add_argument(
        "--ground-method",
        choices=["adapt", "adapt_vqe", "adaptvqe", "uccsd"],
        default="adapt",
        help="Ground-state method. Use `adapt` or `uccsd`.",
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
        help="Basis set used by both ground-state and QSC-EOM calculations.",
    )
    parser.add_argument(
        "--ground-shots",
        type=int,
        default=0,
        help="Shots for ground-state run. Use 0 for analytic mode.",
    )
    parser.add_argument(
        "--excited-shots",
        type=int,
        default=0,
        help="Shots for QSC-EOM run. Use 0 for analytic mode.",
    )
    parser.add_argument(
        "--uccsd-max-iter",
        type=int,
        default=1000,
        help="Maximum gradient-descent iterations for UCCSD.",
    )
    parser.add_argument(
        "--adapt-it",
        type=int,
        default=3,
        help="Number of ADAPT-VQE operator-selection iterations.",
    )
    parser.add_argument(
        "--adapt-optimizer-maxiter",
        type=int,
        default=2000,
        help="Maximum SciPy optimizer iterations per ADAPT iteration.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        
    )
    return parser.parse_args()


def normalize_ground_method(method: str) -> str:
    if method in ("adapt", "adapt_vqe", "adaptvqe"):
        return "adapt"
    if method == "uccsd":
        return "uccsd"
    raise ValueError("ground_method must be one of: adapt, uccsd")


def main():
    args = parse_args()
    method = normalize_ground_method(args.ground_method)

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, args.bond_length]])
    active_electrons = 2
    active_orbitals = 2
    charge = 0

    ground_shots = None if args.ground_shots == 0 else args.ground_shots

    if method == "uccsd":
        params, ground_energy = gs_exact(
            symbols=symbols,
            geometry=geometry,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            charge=charge,
            basis=args.basis,
            shots=ground_shots,
            max_iter=args.uccsd_max_iter,
            return_energy=True,
        )
        excited_state_energies = qsceom(
            symbols=symbols,
            geometry=geometry,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            charge=charge,
            params=params,
            shots=args.excited_shots,
            basis=args.basis,
        )
    else:
        params, ash_excitation, energies = adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=args.adapt_it,
            basis=args.basis,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            charge=charge,
            shots=ground_shots,
            optimizer_maxiter=args.adapt_optimizer_maxiter,
        )
        ground_energy = energies[-1]
        excited_state_energies = qsceom(
            symbols=symbols,
            geometry=geometry,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            charge=charge,
            params=params,
            shots=args.excited_shots,
            ash_excitation=ash_excitation,
            basis=args.basis,
        )

    excited_sorted = np.real_if_close(np.asarray(excited_state_energies[0]))

    lines = [
        "===== H2 QSC-EOM Run =====",
        f"Bond length (Angstrom): {args.bond_length}",
        f"Basis set: {args.basis}",
        f"Ground-state method: {method}",
        f"Ground-state energy (Hartree): {ground_energy}",
        f"QSC-EOM eigenvalues (Hartree): {excited_sorted}",
    ]
    report = "\n".join(lines) + "\n"

    print()
    print(report, end="")

    
    if args.output_file is None:
        suffix = "adapt" if method == "adapt" else "uccsd"
        output_path = Path(__file__).resolve().parent / f"h2_output_{suffix}.txt"
    else:
        output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    #print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
