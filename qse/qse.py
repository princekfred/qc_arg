"""Main QSE runner that follows qse_qisk_as notebook logic."""

try:
    from .functions import (
        ActiveSpaceTransformer,
        AerEstimator,
        Estimator,
        GroundStateEigensolver,
        HartreeFock,
        JordanWignerMapper,
        PySCFDriver,
        SLSQP,
        SparsePauliOp,
        Statevector,
        UCCSD,
        VQE,
        algorithm_globals,
        gto,
        mcscf,
        np,
        scf,
        scipy,
    )
    from .excitations import all_excitations
except ImportError:
    from functions import (
        ActiveSpaceTransformer,
        AerEstimator,
        Estimator,
        GroundStateEigensolver,
        HartreeFock,
        JordanWignerMapper,
        PySCFDriver,
        SLSQP,
        SparsePauliOp,
        Statevector,
        UCCSD,
        VQE,
        algorithm_globals,
        gto,
        mcscf,
        np,
        scf,
        scipy,
    )
    from excitations import all_excitations


def build_active_problem(atom, basis, active_electrons, num_spartial_orbital):
    driver = PySCFDriver(atom=atom, basis=basis)
    problem = driver.run()
    active_transformer = ActiveSpaceTransformer(
        num_electrons=active_electrons,
        num_spatial_orbitals=num_spartial_orbital,
    )
    active_problem = active_transformer.transform(problem)
    return active_problem


def get_frozen_core_energy(atom, basis, active_electrons, num_spartial_orbital):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, ncas=num_spartial_orbital, nelecas=active_electrons)
    mc.kernel()
    frozen_core_energy = mc.e_tot - mc.e_cas
    return frozen_core_energy


def solve_ground_state(active_problem, mapper, seed=170):
    algorithm_globals.random_seed = seed

    # UCCSD-VQE is kept analytic (no shots).
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

    vqe = VQE(Estimator(), ansatz, SLSQP())
    vqe.initial_point = np.zeros(ansatz.num_parameters)

    solver = GroundStateEigensolver(mapper, vqe)
    result = solver.solve(active_problem)
    psi_vqe = result.raw_result.optimal_point

    return result, ansatz, psi_vqe


def get_statevector(ansatz, psi_vqe):
    ansatz = ansatz.assign_parameters(
        dict(zip(ansatz.parameters, psi_vqe)),
        inplace=False,
    )
    statevector = Statevector.from_instruction(ansatz)
    return ansatz, statevector


def sampled_expectation(statevector, operator, shots, rng):
    total = 0.0 + 0.0j
    for pauli, coeff in zip(operator.paulis, operator.coeffs):
        label = pauli.to_label()
        term = SparsePauliOp.from_list([(label, 1.0)])
        mean = float(np.real(Statevector(statevector).expectation_value(term)))
        clipped = float(np.clip(mean, -1.0, 1.0))
        p_plus = 0.5 * (1.0 + clipped)
        n_plus = rng.binomial(int(shots), p_plus)
        sampled_mean = 2.0 * n_plus / int(shots) - 1.0
        total += coeff * sampled_mean
    return total


def build_qse_matrices(
    statevector,
    mapper,
    qubit_op,
    excitations,
    ansatz=None,
    shots=0,
    seed=42,
    shot_repeats=1,
    use_aer_estimator=False,
):
    num_excitations = len(excitations)
    mapped_excitations = [mapper.map(excitation) for excitation in excitations]
    M = np.zeros((num_excitations + 1, num_excitations + 1), dtype=complex)
    S = np.zeros((num_excitations + 1, num_excitations + 1), dtype=complex)
    eigenvalues_shots = []
    ev = None

    if shots and ansatz is not None:
        estimator = None
        if use_aer_estimator:
            estimator = AerEstimator(
                run_options={"shots": int(shots)},
                transpile_options={"seed_transpiler": int(seed)},
                approximation=True,
            )
        I = SparsePauliOp.from_list([("I" * qubit_op.num_qubits, 1.0)])
        rng = np.random.default_rng(int(seed))

        for _ in range(int(shot_repeats)):
            for i in range(len(excitations) + 1):
                for j in range(len(excitations) + 1):
                    if i > 0:
                        op_i = mapped_excitations[i - 1]
                        oi = op_i.adjoint() @ qubit_op
                    if j > 0:
                        op_j = mapped_excitations[j - 1]
                        oj = qubit_op @ op_j
                    if i > 0 and j > 0:
                        op = op_i.adjoint() @ qubit_op @ op_j

                    if i == j == 0:
                        if estimator is not None:
                            M[i, j] = estimator.run(ansatz, qubit_op).result().values[0]
                            S[i, j] = estimator.run(ansatz, I).result().values[0]
                        else:
                            M[i, j] = sampled_expectation(
                                statevector, qubit_op, shots, rng
                            )
                            S[i, j] = 1.0
                    elif i == 0 and j > 0:
                        if estimator is not None:
                            M[i, j] = estimator.run(ansatz, oj).result().values[0]
                            S[i, j] = estimator.run(ansatz, op_j).result().values[0]
                        else:
                            M[i, j] = sampled_expectation(statevector, oj, shots, rng)
                            S[i, j] = sampled_expectation(statevector, op_j, shots, rng)
                    elif i > 0 and j == 0:
                        if estimator is not None:
                            M[i, j] = estimator.run(ansatz, oi).result().values[0]
                            S[i, j] = estimator.run(ansatz, op_i.adjoint()).result().values[0]
                        else:
                            M[i, j] = sampled_expectation(statevector, oi, shots, rng)
                            S[i, j] = sampled_expectation(
                                statevector, op_i.adjoint(), shots, rng
                            )
                    else:
                        if estimator is not None:
                            M[i, j] = estimator.run(ansatz, op).result().values[0]
                            S[i, j] = estimator.run(ansatz, op_i.adjoint() @ op_j).result().values[0]
                        else:
                            M[i, j] = sampled_expectation(statevector, op, shots, rng)
                            S[i, j] = sampled_expectation(
                                statevector, op_i.adjoint() @ op_j, shots, rng
                            )

            try:
                eigval, ev = scipy.linalg.eigh(M, S)
            except Exception:
                eigval, ev = scipy.linalg.eig(M, S)
                eigval = np.real(np.asarray(eigval))
            eigenvalues_shots.append(np.asarray(eigval))

        return M, S, eigenvalues_shots, ev

    for i in range(len(excitations) + 1):
        for j in range(len(excitations) + 1):
            if i == j == 0:
                M[i, j] = Statevector(statevector).expectation_value(qubit_op)
                S[i, j] = 1.0
            elif i == 0 and j > 0:
                op_j = mapped_excitations[j - 1]
                oj = qubit_op @ op_j
                M[i, j] = Statevector(statevector).expectation_value(oj)
                S[i, j] = Statevector(statevector).expectation_value(op_j)
            elif i > 0 and j == 0:
                op_i = mapped_excitations[i - 1]
                oi = op_i.adjoint() @ qubit_op
                M[i, j] = Statevector(statevector).expectation_value(oi)
                S[i, j] = Statevector(statevector).expectation_value(op_i.adjoint())
            else:
                op_i = mapped_excitations[i - 1]
                op_j = mapped_excitations[j - 1]
                op = op_i.adjoint() @ qubit_op @ op_j
                M[i, j] = Statevector(statevector).expectation_value(op)
                S[i, j] = Statevector(statevector).expectation_value(
                    op_i.adjoint() @ op_j
                )

    eigval, ev = scipy.linalg.eigh(M, S)
    eigenvalues_shots.append(np.asarray(eigval))
    return M, S, eigenvalues_shots, ev


def solve_qse_eigenproblem(M, S):
    eig, ev = scipy.linalg.eigh(M, S)
    return eig, ev


def run_qse(
    atom,
    basis="sto-6g",
    active_electrons=4,
    active_orbitals=4,
    seed=170,
    shots=0,
    uccsd_shots=0,
    shot_repeats=1,
    use_aer_estimator=False,
):
    # UCCSD shots are intentionally disabled in this workflow.
    if uccsd_shots not in (0, None):
        raise ValueError(
            "UCCSD shots are disabled. Use uccsd_shots=0 and control only QSE shots via `shots`."
        )
    if shots is None:
        shots = 0

    num_spartial_orbital = active_orbitals
    num_spin_orbitals = num_spartial_orbital * 2

    excitations = all_excitations(num_spin_orbitals)

    active_problem = build_active_problem(
        atom=atom,
        basis=basis,
        active_electrons=active_electrons,
        num_spartial_orbital=num_spartial_orbital,
    )

    core = get_frozen_core_energy(
        atom=atom,
        basis=basis,
        active_electrons=active_electrons,
        num_spartial_orbital=num_spartial_orbital,
    )

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(active_problem.hamiltonian.second_q_op())

    result, ansatz, psi_vqe = solve_ground_state(
        active_problem=active_problem,
        mapper=mapper,
        seed=seed,
    )
    gr = result.total_energies
    ground_energy = float(np.asarray(gr).reshape(-1)[0])

    ansatz, statevector = get_statevector(ansatz, psi_vqe)

    M, S, eigenvalues_shots, ev = build_qse_matrices(
        statevector=statevector,
        mapper=mapper,
        qubit_op=qubit_op,
        excitations=excitations,
        ansatz=ansatz,
        shots=shots,
        seed=seed,
        shot_repeats=shot_repeats,
        use_aer_estimator=use_aer_estimator,
    )

    cond_num = np.linalg.cond(S)
    eig = np.asarray(eigenvalues_shots[-1]) + core

    return {
        "ground_energy": ground_energy,
        "qse_eigenvalues": eig,
        "qse_eigenvectors": ev,
        "qse_eigenvalues_shots": [np.asarray(v) + core for v in eigenvalues_shots],
        "condition_number": cond_num,
        "M": M,
        "S": S,
        "optimal_point": psi_vqe,
        "num_excitations": len(excitations),
        "frozen_core_energy": core,
    }


__all__ = ["run_qse"]
