"""MPI-parallel QSC-EOM solver for ground-state ansatzes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Sequence

try:
    from mpi4py import MPI as _MPI
except ImportError:  # pragma: no cover
    _MPI = None

qml = None
np = None
_inite = None


def _require_quantum_deps():
    global qml, np
    if qml is not None and np is not None:
        return
    try:
        import pennylane as _qml
        from pennylane import numpy as _np
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qsc_eom requires PennyLane and a quantum chemistry backend. "
            "Install with: `pip install pennylane pyscf` "
            "(and optionally `pip install pennylane-lightning`)."
        ) from exc
    qml = _qml
    np = _np


def _load_inite():
    global _inite
    if _inite is not None:
        return _inite

    try:
        from .exc import inite as _inite_local

        _inite = _inite_local
        return _inite
    except Exception:
        pass

    try:
        from exc import inite as _inite_local

        _inite = _inite_local
        return _inite
    except Exception:
        pass

    exc_path = Path(__file__).with_name("exc.py")
    spec = importlib.util.spec_from_file_location("qsceom_ground_exc", exc_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load inite() helper from {exc_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _inite = mod.inite
    return _inite


def _normalize_shots(shots: Optional[int]) -> Optional[int]:
    if shots is None or shots == 0:
        return None
    if shots < 0:
        raise ValueError("shots must be >= 0")
    return int(shots)


def _make_device(qubits: int, shots: Optional[int]):
    _require_quantum_deps()
    try:
        return qml.device("lightning.qubit", wires=qubits, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=qubits, shots=shots)


def _mpi_context():
    if _MPI is None:
        return None, 1, 0, None
    comm = _MPI.COMM_WORLD
    return comm, comm.Get_size(), comm.Get_rank(), _MPI.SUM


def _apply_ansatz(params, wires, s_wires, d_wires, hf_state, ash_excitation):
    _require_quantum_deps()
    if ash_excitation is None:
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return

    for i, excitation in enumerate(ash_excitation):
        if len(excitation) == 4:
            qml.FermionicDoubleExcitation(
                weight=params[i],
                wires1=list(range(excitation[0], excitation[1] + 1)),
                wires2=list(range(excitation[2], excitation[3] + 1)),
            )
        elif len(excitation) == 2:
            qml.FermionicSingleExcitation(
                weight=params[i],
                wires=list(range(excitation[0], excitation[1] + 1)),
            )
        else:
            raise ValueError("Each excitation must have length 2 or 4")


def qsc_eom(
    symbols: Sequence[str],
    coordinates,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    params,
    ash_excitation=None,
    shots: int = 0,
    basis: str = "sto-3g",
):
    """Build and diagonalize the QSC-EOM M-matrix.

    Returns
    -------
    tuple
        ``(eigvals, eigvecs)`` sorted by ascending eigenvalue.
    """

    _require_quantum_deps()
    inite = _load_inite()
    norm_shots = _normalize_shots(shots)
    params = np.asarray(params)
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if len(symbols) != coordinates.shape[0]:
        raise ValueError("len(symbols) must match number of coordinate rows")

    if params.ndim != 1:
        raise ValueError("params must be a 1D array-like")
    if ash_excitation is not None and len(ash_excitation) != len(params):
        raise ValueError("len(ash_excitation) must match len(params)")

    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        basis=basis,
        method="pyscf",
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
        unit="angstrom",
    )
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)
    hf_state = qml.qchem.hf_state(active_electrons, qubits)

    excitation_configs = inite(active_electrons, qubits)
    dev = _make_device(qubits, norm_shots)

    @qml.qnode(dev)
    def circuit_d(curr_params, occ):
        for w in occ:
            qml.X(wires=w)
        _apply_ansatz(curr_params, wires, s_wires, d_wires, hf_state, ash_excitation)
        return qml.expval(hamiltonian)

    @qml.qnode(dev)
    def circuit_od(curr_params, occ1, occ2):
        for w in occ1:
            qml.X(wires=w)

        first = -1
        for v in occ2:
            if v not in occ1:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])
        for v in occ1:
            if v not in occ2:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])

        _apply_ansatz(curr_params, wires, s_wires, d_wires, hf_state, ash_excitation)
        return qml.expval(hamiltonian)

    comm, size, rank, mpi_sum = _mpi_context()
    mat_size = len(excitation_configs)

    m_diag_local = np.zeros(mat_size)
    for i in range(rank, mat_size, size):
        m_diag_local[i] = circuit_d(params, excitation_configs[i])

    if comm is None:
        m_diag = m_diag_local
    else:
        m_diag = np.zeros(mat_size)
        comm.Allreduce(m_diag_local, m_diag, op=mpi_sum)

    m_local = np.zeros((mat_size, mat_size))
    flat_idx = 0
    for i in range(mat_size):
        for j in range(i + 1):
            if flat_idx % size == rank:
                if i == j:
                    m_tmp = m_diag[i]
                else:
                    m_tmp = (
                        circuit_od(params, excitation_configs[i], excitation_configs[j])
                        - m_diag[i] / 2.0
                        - m_diag[j] / 2.0
                    )
                m_local[i, j] = m_tmp
                m_local[j, i] = m_tmp
            flat_idx += 1

    if comm is None:
        m_matrix = m_local
    else:
        m_matrix = np.zeros_like(m_local)
        comm.Allreduce(m_local, m_matrix, op=mpi_sum)

    eigvals, eigvecs = np.linalg.eigh(m_matrix)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]
