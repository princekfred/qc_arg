"""Excitation builders used by the QSE workflow."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit_nature.second_q.operators import FermionicOp


def closed_shell_occupied_spin_orbitals(num_spin_orbitals: int) -> list[int]:
    """Return occupied spin-orbital indices for a closed-shell reference."""
    if num_spin_orbitals % 2 != 0:
        raise ValueError("num_spin_orbitals must be even.")
    if num_spin_orbitals < 4:
        raise ValueError("num_spin_orbitals must be at least 4.")

    occupied: list[int] = []
    half = num_spin_orbitals // 2
    for idx in range(num_spin_orbitals // 4):
        occupied.append(idx)
        occupied.append(idx + half)
    return occupied


def all_excitations(
    num_spin_orbitals: int,
    occupied: Sequence[int] | None = None,
) -> list[FermionicOp]:
    """Generate single and double excitations following the notebook logic."""
    if occupied is None:
        occupied = closed_shell_occupied_spin_orbitals(num_spin_orbitals)

    occupied_set = set(occupied)
    half = num_spin_orbitals // 2
    excitations: list[FermionicOp] = []

    for i in range(num_spin_orbitals):
        for j in range(i + 1, num_spin_orbitals):
            same_spin_single = (i < half and j < half) or (i >= half and j >= half)
            if same_spin_single and (i in occupied_set and j not in occupied_set):
                excitations.append(
                    FermionicOp(
                        {f"+_{j} -_{i}": 1.0},
                        num_spin_orbitals=num_spin_orbitals,
                    )
                )

            for k in range(j + 1, num_spin_orbitals):
                for l in range(k + 1, num_spin_orbitals):
                    same_spin_double = (
                        (i < half and j < half and k < half and l < half)
                        or (i >= half and j >= half and k >= half and l >= half)
                    )
                    if (
                        same_spin_double
                        and i in occupied_set
                        and j in occupied_set
                        and k not in occupied_set
                        and l not in occupied_set
                    ):
                        excitations.append(
                            FermionicOp(
                                {f"+_{l} +_{k} -_{i} -_{j}": 1.0},
                                num_spin_orbitals=num_spin_orbitals,
                            )
                        )

    for i in range(half):
        for j in range(half, num_spin_orbitals):
            for k in range(half):
                for l in range(half, num_spin_orbitals):
                    if i != k and j != l and i < k and j < l:
                        if (
                            i in occupied_set
                            and j in occupied_set
                            and k not in occupied_set
                            and l not in occupied_set
                        ):
                            excitations.append(
                                FermionicOp(
                                    {f"+_{l} +_{k} -_{i} -_{j}": 1.0},
                                    num_spin_orbitals=num_spin_orbitals,
                                )
                            )

    return excitations


__all__ = ["all_excitations", "closed_shell_occupied_spin_orbitals"]
