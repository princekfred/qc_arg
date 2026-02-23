"""Main QSE runner extracted from ``qse_qisk_as.ipynb``."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper

try:  # Package usage: from qse.qse import run_qse
    from .excitations import all_excitations
    from .functions import (
        build_active_space_problem,
        build_qse_matrices,
        compute_frozen_core_energy,
        extract_ground_energy,
        run_uccsd_ground_state,
        solve_generalized_eigenproblem,
    )
except ImportError:  # Script usage from inside qse/ directory.
    from excitations import all_excitations
    from functions import (
        build_active_space_problem,
        build_qse_matrices,
        compute_frozen_core_energy,
        extract_ground_energy,
        run_uccsd_ground_state,
        solve_generalized_eigenproblem,
    )


def run_qse(
    atom: str,
    basis: str = "sto-6g",
    active_electrons: int = 4,
    active_orbitals: int = 4,
    seed: int = 170,
    shots: int = 0,
    uccsd_shots: int = 0,
    optimizer: Any | None = None,
) -> dict[str, Any]:
    """Run UCCSD + QSE and return energies and intermediate matrices."""
    mapper = JordanWignerMapper()
    active_problem = build_active_space_problem(
        atom=atom,
        basis=basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    qubit_hamiltonian = mapper.map(active_problem.hamiltonian.second_q_op())
    frozen_core_energy = compute_frozen_core_energy(
        atom=atom,
        basis=basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    ground_result, bound_circuit, optimal_point = run_uccsd_ground_state(
        active_problem=active_problem,
        mapper=mapper,
        seed=seed,
        shots=uccsd_shots,
        optimizer=optimizer,
    )
    ground_energy = extract_ground_energy(ground_result.total_energies)

    num_spin_orbitals = active_problem.num_spatial_orbitals * 2
    excitations = all_excitations(num_spin_orbitals=num_spin_orbitals)
    mapped_excitations = [mapper.map(excitation) for excitation in excitations]

    m_matrix, s_matrix = build_qse_matrices(
        state_circuit=bound_circuit,
        qubit_hamiltonian=qubit_hamiltonian,
        mapped_excitations=mapped_excitations,
        shots=shots,
        seed=seed,
    )
    eigvals, eigvecs, cond_number = solve_generalized_eigenproblem(
        h_matrix=m_matrix,
        overlap_matrix=s_matrix,
    )

    qse_eigenvalues = np.asarray(eigvals, dtype=float) + float(frozen_core_energy)
    return {
        "ground_energy": float(ground_energy),
        "qse_eigenvalues": qse_eigenvalues,
        "qse_eigenvectors": eigvecs,
        "condition_number": cond_number,
        "M": m_matrix,
        "S": s_matrix,
        "optimal_point": optimal_point,
        "num_excitations": len(excitations),
        "frozen_core_energy": float(frozen_core_energy),
    }


__all__ = ["run_qse"]
