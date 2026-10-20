#!/usr/bin/env python3
"""Run ADAPT-VQE and QSC-EOM for square H4 with 1.5 Angstrom sides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SYMBOLS = ["H"] * 4
COORDINATES_ANGSTROM = [
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [1.5, 1.5, 0.0],
    [0.0, 1.5, 0.0],
]
BASIS = "sto-3g"
ACTIVE_ELECTRONS = 4
ACTIVE_ORBITALS = 4
CHARGE = 0
SPIN = 0
DEFAULT_ADAPT_IT = 4  # The 16-step ansatz targets a higher singlet at this geometry.
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "output" / "h4sq_output.txt"
PLOT_FILE = Path(__file__).resolve().parents[1] / "output" / "h4sq_fci_vs_qsceom.png"


def _compute_fci_roots(nroots: int):
    import numpy as np
    from pyscf import fci, gto, scf

    mol = gto.M(
        atom=[(symbol, tuple(xyz)) for symbol, xyz in zip(SYMBOLS, COORDINATES_ANGSTROM)],
        unit="angstrom",
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.level_shift = 0.5
    mf.diis_space = 12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run()

    # Include all spin sectors with N_alpha = N_beta = 2, as in QSC-EOM.
    solver = fci.FCI(mf, singlet=False)
    solver.nroots = nroots
    energies, _ = solver.kernel()
    energies = np.sort(np.atleast_1d(np.asarray(energies, dtype=float)))
    if len(energies) != nroots:
        raise RuntimeError(f"Expected {nroots} FCI roots; got {len(energies)}")
    return energies


def _plot_spectra(qsceom_energies, fci_energies) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_indices = list(range(len(fci_energies)))
    qsc_ground = float(qsceom_energies[0])
    fci_ground = float(fci_energies[0])
    fig, (ax_total, ax_gap) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
    )
    series = (
        ("FCI", fci_energies, fci_ground, "#0072B2", "o"),
        ("QSC-EOM", qsceom_energies, qsc_ground, "#D55E00", "s"),
    )
    for label, energies, ground, color, marker in series:
        totals = [float(value) for value in energies]
        ax_total.plot(
            state_indices, totals, label=label, color=color, marker=marker, markersize=4
        )
        ax_gap.plot(
            state_indices,
            [value - ground for value in totals],
            label=label,
            color=color,
            marker=marker,
            markersize=4,
        )

    ax_total.set_ylabel("Total energy (Ha)")
    ax_gap.set_ylabel("Excitation energy (Ha)")
    ax_gap.set_xlabel("Energy rank (0 = ground; spectra sorted separately)")
    ax_gap.set_xticks(range(0, len(state_indices), 2))
    ax_total.legend()
    for axis in (ax_total, ax_gap):
        axis.grid(alpha=0.25)
    fig.suptitle("Square H4, 1.5 Å, STO-3G: FCI and QSC-EOM spectra")
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapt-it",
        type=int,
        default=DEFAULT_ADAPT_IT,
        help=f"ADAPT iterations (default: {DEFAULT_ADAPT_IT})",
    )
    parser.add_argument(
        "--serial", action="store_true", help="Run without MPI (one process)"
    )
    args = parser.parse_args()
    if args.adapt_it < 1:
        parser.error("--adapt-it must be at least 1")

    if args.serial:
        # The solver treats mpi4py as optional and then uses its serial path.
        sys.modules["mpi4py"] = None

    repo_root = Path(__file__).resolve().parents[2]
    solver_dir = repo_root / "qsceomdf"
    # Both ADAPT-VQE and the exc.inite helper used by QSC-EOM live here.
    ground_state_dir = repo_root / "qsceom" / "ground state"
    sys.path.insert(0, str(solver_dir))
    sys.path.append(str(ground_state_dir))

    from adaptvqe import adapt_vqe
    from qsceom_par import qsc_eom

    params, excitations, energies, hamiltonian, qubits = adapt_vqe(
        symbols=SYMBOLS,
        geometry=COORDINATES_ANGSTROM,
        adapt_it=args.adapt_it,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        active_electrons=ACTIVE_ELECTRONS,
        active_orbitals=ACTIVE_ORBITALS,
        shots=None,
        optimizer_maxiter=500,
        return_hamiltonian=True,
    )
    eigvals, _ = qsc_eom(
        symbols=SYMBOLS,
        coordinates=COORDINATES_ANGSTROM,
        active_electrons=ACTIVE_ELECTRONS,
        active_orbitals=ACTIVE_ORBITALS,
        charge=CHARGE,
        params=params,
        ash_excitation=excitations,
        shots=0,
        basis=BASIS,
        hamiltonian=hamiltonian,
        qubits=qubits,
    )

    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    if rank == 0:
        if len(eigvals) != 27:
            raise RuntimeError(f"Expected 27 QSC-EOM roots; got {len(eigvals)}")
        fci_energies = _compute_fci_roots(nroots=27)
        _plot_spectra(eigvals, fci_energies)

        lines = [
            f"Output file: {OUTPUT_FILE}",
            f"Plot file: {PLOT_FILE}",
            "H4 square, side = 1.5 Angstrom, basis = sto-3g",
            f"ADAPT iterations: {args.adapt_it}",
            f"ADAPT-VQE ground energy (Ha): {float(energies[-1]):.12f}",
            f"QSC-EOM ground energy (Ha): {float(eigvals[0]):.12f}",
            f"FCI ground energy (Ha): {float(fci_energies[0]):.12f}",
            f"Lowest-root energy difference, QSC-EOM - FCI (Ha): "
            f"{float(eigvals[0] - fci_energies[0]):.12f}",
            "Root indices are sorted separately by energy; they do not assign matching states.",
            "FCI roots have N_alpha = N_beta = 2 and may have different total spins.",
            "QSC-EOM spans 27 configurations; full FCI has 36 states in this spin sector.",
        ]
        ground_energy = float(eigvals[0])
        lines.append(f"QSC-EOM excited states: {len(eigvals) - 1}")
        lines.append("State  Total energy (Ha)  Excitation energy (Ha)")
        for state, energy in enumerate(eigvals[1:], start=1):
            total_energy = float(energy)
            lines.append(
                f"{state:>5d}  {total_energy:>17.12f}  "
                f"{total_energy - ground_energy:>22.12f}"
            )

        lines.extend(
            [
                "",
                f"Lowest FCI states shown: {len(fci_energies)} of 36 "
                f"(ground + {len(fci_energies) - 1} excited)",
                "State  Total energy (Ha)  Excitation energy (Ha)",
            ]
        )
        fci_ground = float(fci_energies[0])
        for state, energy in enumerate(fci_energies):
            total_energy = float(energy)
            lines.append(
                f"{state:>5d}  {total_energy:>17.12f}  "
                f"{total_energy - fci_ground:>22.12f}"
            )

        report = "\n".join(lines) + "\n"
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
        print(report, end="")


if __name__ == "__main__":
    main()
