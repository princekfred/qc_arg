"""Ground-state solver based on the UCCSD ansatz."""

from __future__ import annotations

from typing import Optional

import pennylane as qml
from pennylane import numpy as np


def _normalize_shots(shots: Optional[int]) -> Optional[int]:
    if shots is None or shots == 0:
        return None
    if shots < 0:
        raise ValueError("shots must be >= 0")
    return int(shots)


def _make_device(qubits: int, shots: Optional[int]):
    """Create a PennyLane simulator device with a robust fallback."""
    try:
        return qml.device("lightning.qubit", wires=qubits, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=qubits, shots=shots)


def gs_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    basis: str = "sto-3g",
    shots: Optional[int] = None,
    max_iter: int = 10000,
    return_energy: bool = False,
    stepsize: float = 2.0,
    verbose: bool = True,
):
    """Optimize UCCSD parameters for the molecular ground state.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ``["H", "H"]``.
    geometry
        Atomic coordinates in Angstrom.
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    charge
        Total molecular charge.
    basis
        Basis set for the PySCF-backed Hamiltonian.
    shots
        Number of shots for sampling mode. Use ``0`` or ``None`` for analytic mode.
    max_iter
        Maximum gradient-descent iterations.
    return_energy
        If ``True``, return both ``(params, energy)``.
    stepsize
        Gradient-descent stepsize.
    verbose
        Print the final ground-state energy when ``True``.
    """

    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if stepsize <= 0:
        raise ValueError("stepsize must be > 0")

    norm_shots = _normalize_shots(shots)

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
    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    wires = range(qubits)
    params = np.zeros(len(singles) + len(doubles), dtype=float)
    dev = _make_device(qubits, norm_shots)

    # Define the qnode.
    if norm_shots is None:
        @qml.qnode(dev, interface="autograd", diff_method="adjoint")
        def circuit(curr_params, ansatz_wires, single_wires, double_wires, init_state):
            qml.UCCSD(curr_params, ansatz_wires, single_wires, double_wires, init_state)
            return qml.expval(hamiltonian)
    else:
        @qml.qnode(dev, interface="autograd")
        def circuit(curr_params, ansatz_wires, single_wires, double_wires, init_state):
            qml.UCCSD(curr_params, ansatz_wires, single_wires, double_wires, init_state)
            return qml.expval(hamiltonian)

    optimizer = qml.GradientDescentOptimizer(stepsize=stepsize)
    for _ in range(int(max_iter)):
        params, _ = optimizer.step_and_cost(
            circuit,
            params,
            ansatz_wires=wires,
            single_wires=s_wires,
            double_wires=d_wires,
            init_state=hf_state,
        )

    ground_energy = circuit(params, wires, s_wires, d_wires, hf_state)
    if verbose:
        #print("Ground state energy:", ground_energy)

    #if return_energy:
        return params, ground_energy
    return params
