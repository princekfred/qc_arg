"""Excitation builders matching qse_qisk_as notebook style."""

try:
    from .functions import FermionicOp
except ImportError:
    from functions import FermionicOp


def get_occupied_orbitals(num_spin_orbitals):
    occupied = []
    for i in range(num_spin_orbitals // 4):
        occupied.append(i)
        occupied.append(i + num_spin_orbitals // 2)
    return occupied


def all_excitations(num_spin_orbitals, occupied=None):
    
    if occupied is None:
        occupied = get_occupied_orbitals(num_spin_orbitals)

    excitations = []

    for i in range(num_spin_orbitals):
        for j in range(i + 1, num_spin_orbitals):
            if i != j and (
                (i < num_spin_orbitals // 2 and j < num_spin_orbitals // 2)
                or (i >= num_spin_orbitals // 2 and j >= num_spin_orbitals // 2)
            ):
                if i in occupied and j not in occupied:
                    excitation = FermionicOp(
                        {f"+_{j} -_{i}": 1.0},
                        num_spin_orbitals=num_spin_orbitals,
                    )
                    excitations.append(excitation)

            for k in range(j + 1, num_spin_orbitals):
                for l in range(k + 1, num_spin_orbitals):
                    if i != j and k != l and (
                        (
                            i < num_spin_orbitals // 2
                            and j < num_spin_orbitals // 2
                            and k < num_spin_orbitals // 2
                            and l < num_spin_orbitals // 2
                        )
                        or (
                            i >= num_spin_orbitals // 2
                            and j >= num_spin_orbitals // 2
                            and k >= num_spin_orbitals // 2
                            and l >= num_spin_orbitals // 2
                        )
                    ):
                        if (
                            i in occupied
                            and k not in occupied
                            and j in occupied
                            and l not in occupied
                        ):
                            excitation = FermionicOp(
                                {f"+_{l} +_{k} -_{i} -_{j}": 1.0},
                                num_spin_orbitals=num_spin_orbitals,
                            )
                            excitations.append(excitation)

    for i in range(num_spin_orbitals // 2):
        for j in range(num_spin_orbitals // 2, num_spin_orbitals):
            for k in range(num_spin_orbitals // 2):
                for l in range(num_spin_orbitals // 2, num_spin_orbitals):
                    if i != k and j != l and i < k and j < l:
                        if (
                            i in occupied
                            and k not in occupied
                            and j in occupied
                            and l not in occupied
                        ):
                            exc = FermionicOp(
                                {f"+_{l} +_{k} -_{i} -_{j}": 1.0},
                                num_spin_orbitals=num_spin_orbitals,
                            )
                            excitations.append(exc)

    return excitations


__all__ = ["all_excitations", "get_occupied_orbitals"]
