"""Shared functions for QSE workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyscf.mcscf as mcscf
import scipy.linalg
from pyscf import gto, scf
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.utils import algorithm_globals
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


def build_active_space_problem(
    atom: str,
    basis: str,
    active_electrons: int,
    active_orbitals: int,
):
    """Build and return the transformed active-space electronic problem."""
    driver = PySCFDriver(atom=atom, basis=basis)
    problem = driver.run()
    transformer = ActiveSpaceTransformer(
        num_electrons=active_electrons,
        num_spatial_orbitals=active_orbitals,
    )
    return transformer.transform(problem)


def compute_frozen_core_energy(
    atom: str,
    basis: str,
    active_electrons: int,
    active_orbitals: int,
) -> float:
    """Compute frozen-core correction via CASCI (matching notebook logic)."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, ncas=active_orbitals, nelecas=active_electrons)
    mc.kernel()
    return float(mc.e_tot - mc.e_cas)


def _build_estimator(shots: int, seed: int):
    if shots <= 0:
        return Estimator()

    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Shot-based VQE requires qiskit-aer. Install with: `pip install qiskit-aer`."
        ) from exc

    return AerEstimator(
        run_options={"shots": int(shots), "seed": seed},
        transpile_options={"seed_transpiler": seed},
    )


def run_uccsd_ground_state(
    active_problem,
    mapper: JordanWignerMapper,
    seed: int = 170,
    shots: int = 0,
    optimizer: Any | None = None,
):
    """Run UCCSD-VQE for the active-space problem and return bound state circuit."""
    algorithm_globals.random_seed = seed
    if optimizer is None:
        optimizer = SLSQP()

    ansatz = UCCSD(
        active_problem.num_spatial_orbitals,
        active_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            active_problem.num_spatial_orbitals,
            active_problem.num_particles,
            mapper,
        ),
    )

    vqe = VQE(_build_estimator(shots=shots, seed=seed), ansatz, optimizer)
    vqe.initial_point = np.zeros(ansatz.num_parameters)
    solver = GroundStateEigensolver(mapper, vqe)
    result = solver.solve(active_problem)

    raw_optimal_point = getattr(result.raw_result, "optimal_point", None)
    if raw_optimal_point is None:
        raw_optimal_point = np.zeros(ansatz.num_parameters)
    optimal_point = np.asarray(raw_optimal_point, dtype=float)
    bound_circuit = ansatz.assign_parameters(
        dict(zip(ansatz.parameters, optimal_point)),
        inplace=False,
    )
    return result, bound_circuit, optimal_point


def extract_ground_energy(total_energies: Sequence[float]) -> float:
    energies = np.asarray(total_energies, dtype=float).reshape(-1)
    if energies.size == 0:
        raise ValueError("No ground-state energies found in result.")
    return float(energies[0])


def _sampled_expectation(
    statevector: Statevector,
    operator: SparsePauliOp,
    shots: int,
    rng: np.random.Generator,
) -> complex:
    total = 0.0 + 0.0j
    for pauli, coeff in zip(operator.paulis, operator.coeffs):
        label = pauli.to_label()
        term = SparsePauliOp.from_list([(label, 1.0)])
        mean = float(np.real(statevector.expectation_value(term)))
        clipped = float(np.clip(mean, -1.0, 1.0))
        p_plus = 0.5 * (1.0 + clipped)
        count_plus = rng.binomial(shots, p_plus)
        sampled_mean = 2.0 * count_plus / shots - 1.0
        total += coeff * sampled_mean
    return complex(total)


def evaluate_expectation(
    statevector: Statevector,
    operator: SparsePauliOp,
    shots: int = 0,
    rng: np.random.Generator | None = None,
) -> complex:
    if shots <= 0:
        return complex(statevector.expectation_value(operator))
    if rng is None:
        rng = np.random.default_rng()
    return _sampled_expectation(statevector, operator, shots=shots, rng=rng)


def build_qse_matrices(
    state_circuit,
    qubit_hamiltonian: SparsePauliOp,
    mapped_excitations: Sequence[SparsePauliOp],
    shots: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build QSE effective Hamiltonian (M) and overlap (S) matrices."""
    size = len(mapped_excitations) + 1
    m_matrix = np.zeros((size, size), dtype=complex)
    s_matrix = np.zeros((size, size), dtype=complex)

    identity = SparsePauliOp.from_list([("I" * qubit_hamiltonian.num_qubits, 1.0)])
    generators = [identity, *mapped_excitations]
    statevector = Statevector.from_instruction(state_circuit)
    rng = np.random.default_rng(seed) if shots > 0 else None

    for i, op_i in enumerate(generators):
        op_i_dag = op_i.adjoint()
        for j, op_j in enumerate(generators):
            m_op = op_i_dag @ qubit_hamiltonian @ op_j
            s_op = op_i_dag @ op_j
            m_matrix[i, j] = evaluate_expectation(
                statevector, m_op, shots=shots, rng=rng
            )
            s_matrix[i, j] = evaluate_expectation(
                statevector, s_op, shots=shots, rng=rng
            )

    # Keep matrices Hermitian up to numerical/shot noise.
    m_matrix = 0.5 * (m_matrix + m_matrix.conj().T)
    s_matrix = 0.5 * (s_matrix + s_matrix.conj().T)
    return m_matrix, s_matrix


def solve_generalized_eigenproblem(
    h_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    stabilization: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve Hc = ESc and return sorted eigenpairs plus cond(S)."""
    cond_number = float(np.linalg.cond(overlap_matrix))
    stabilized_s = overlap_matrix + stabilization * np.eye(
        overlap_matrix.shape[0], dtype=complex
    )

    try:
        eigvals, eigvecs = scipy.linalg.eigh(h_matrix, stabilized_s)
    except Exception:
        eigvals, eigvecs = scipy.linalg.eig(h_matrix, stabilized_s)

    eigvals = np.real_if_close(np.asarray(eigvals))
    eigvals = np.real(np.asarray(eigvals, dtype=np.complex128))
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs, cond_number
