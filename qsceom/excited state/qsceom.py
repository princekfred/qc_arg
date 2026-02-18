"""QSC-EOM excited-state solver.

This module intentionally preserves the existing M-matrix construction:
1) diagonal terms from ``circuit_d``
2) off-diagonal terms from ``circuit_od``
3) the same shift ``M_ij = Mtmp - M_ii/2 - M_jj/2``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pennylane as qml
from pennylane import numpy as np

try:
    from .excitations import inite
except ImportError:  # pragma: no cover
    from excitations import inite


def _normalize_shots(shots: Optional[int]) -> Optional[int]:
    if shots is None or shots == 0:
        return None
    if shots < 0:
        raise ValueError("shots must be >= 0")
    return int(shots)


def _make_device(qubits: int, shots: Optional[int]):
    try:
        return qml.device("lightning.qubit", wires=qubits, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=qubits, shots=shots)


def _apply_ansatz(params, wires, s_wires, d_wires, hf_state, ash_excitation):
    """Apply either UCCSD or ADAPT-selected fermionic excitations."""
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


def qsceom(
    symbols: Sequence[str],
    geometry,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    params,
    shots: int = 0,
    ash_excitation=None,
    basis: str = "sto-3g",
):
    """Compute QSC-EOM eigenvalues for one excitation-space construction.

    Returns
    -------
    list
        A list containing one sorted eigenvalue vector, preserving legacy API.
    """

    norm_shots = _normalize_shots(shots)
    params = np.asarray(params)
    if params.ndim != 1:
        raise ValueError("params must be a 1D array-like")
    if ash_excitation is not None and len(ash_excitation) != len(params):
        raise ValueError("len(ash_excitation) must match len(params)")

    # Build the electronic Hamiltonian.
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
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

    null_state = np.zeros(qubits, int)
    excitation_configs = inite(active_electrons, qubits)
    dev = _make_device(qubits, norm_shots)

    @qml.qnode(dev)
    def circuit_d(curr_params, occ, ansatz_wires, single_wires, double_wires, init_state):
        for w in occ:
            qml.X(wires=w)
        _apply_ansatz(curr_params, ansatz_wires, single_wires, double_wires, init_state, ash_excitation)
        return qml.expval(hamiltonian)

    @qml.qnode(dev)
    def circuit_od(curr_params, occ1, occ2, ansatz_wires, single_wires, double_wires, init_state):
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

        _apply_ansatz(curr_params, ansatz_wires, single_wires, double_wires, init_state, ash_excitation)
        return qml.expval(hamiltonian)

    # Keep the original M-matrix construction logic unchanged.
    m_matrix = np.zeros((len(excitation_configs), len(excitation_configs)))
    for i in range(len(excitation_configs)):
        for j in range(len(excitation_configs)):
            if i == j:
                m_matrix[i, i] = circuit_d(
                    params,
                    excitation_configs[i],
                    wires,
                    s_wires,
                    d_wires,
                    null_state,
                )

    for i in range(len(excitation_configs)):
        for j in range(len(excitation_configs)):
            if i != j:
                mtmp = circuit_od(
                    params,
                    excitation_configs[i],
                    excitation_configs[j],
                    wires,
                    s_wires,
                    d_wires,
                    null_state,
                )
                m_matrix[i, j] = mtmp - m_matrix[i, i] / 2.0 - m_matrix[j, j] / 2.0

    eigvals = np.linalg.eigvals(m_matrix)
    return [np.sort(eigvals)]
