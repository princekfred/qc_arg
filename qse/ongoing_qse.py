import numpy as np
from pathlib import Path
from qiskit import transpile
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA,SLSQP, POWELL
from qiskit.primitives import Estimator
from qiskit_algorithms.utils import algorithm_globals
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.quantum_info import Statevector, SparsePauliOp
import scipy
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
import numpy as np
from pyscf import gto, scf
import pyscf.mcscf as mcscf

#Create the all possible excitations
num_spartial_orbital = 4
num_spin_orbitals = num_spartial_orbital * 2

# Initialize the mapper
mapper = JordanWignerMapper()
#Create an identity operator
I = FermionicOp({'': 1.0}, num_spin_orbitals =num_spin_orbitals)
I = mapper.map(I)

 #list of occupied orbitals
occupied = []
for i in range(num_spin_orbitals//4):
    occupied.append(i)
    occupied.append(i+num_spin_orbitals//2)
#occupied = [0,4,1,5]
    
# Generate all possible single excitations
excitations = []
def all_excitations(num_spin_orbitals):
    for i in range(num_spin_orbitals):
        for j in range(i+1, num_spin_orbitals):
            # Prevent electrons from moving from alpha spin to beta spin and beta spin to alpha
            if i != j and ((i < num_spin_orbitals // 2 and j < num_spin_orbitals // 2) or (i >= num_spin_orbitals // 2 and j >= num_spin_orbitals // 2)):
                # Only consider excitations where the first two alpha and beta spins are filled with electrons
                if (i in occupied and j not in occupied): 
                    excitation = FermionicOp({f'+_{j} -_{i}': 1.0}, num_spin_orbitals=num_spin_orbitals)
                    excitations.append(excitation)
    
        #Generate possible double excitations
        #Double excitations all from alpha or beta orbitals
            for k in range(j+1, num_spin_orbitals):
                for l in range(k+1, num_spin_orbitals):
                    if i != j and k != l and ((i < num_spin_orbitals // 2 and j < num_spin_orbitals // 2 and k < num_spin_orbitals // 2 and l < num_spin_orbitals // 2) or (i >= num_spin_orbitals // 2 and j >= num_spin_orbitals // 2 and k >= num_spin_orbitals // 2 and l >= num_spin_orbitals // 2)):
                        # Only consider excitations where the first two alpha and beta spins are filled with electrons
                        if (i in occupied and k not in occupied and j in occupied and l not in occupied): 
                            excitation = FermionicOp({f'+_{l} +_{k} -_{i} -_{j}': 1.0}, num_spin_orbitals=num_spin_orbitals)
                            excitations.append(excitation)
  
    for i in range(num_spin_orbitals // 2):
        for j in range(num_spin_orbitals // 2, num_spin_orbitals):
            for k in range(num_spin_orbitals // 2):
                for l in range(num_spin_orbitals // 2, num_spin_orbitals):
                    if i != k and j != l and i < k and j < l:
                        # Condition to ensure one alpha and one beta excitation
                        if (i in occupied and k not in occupied and j in occupied and l not in occupied): 
                           # Create the FermionicOp and add to double_exc list
                            exc = FermionicOp({f'+_{l} +_{k} -_{i} -_{j}': 1.0}, num_spin_orbitals=num_spin_orbitals)
                            excitations.append(exc)

    return excitations
excitations = all_excitations(num_spin_orbitals)
print(len(excitations))

distances = np.linspace(0.60, 2.46, 2)
#for d in distances:
atom = f"N 0.0 0.0 0.0; H 2.526315789473684 0.0 0.0; H -0.506 0.876 0.0; H -0.506 -0.876 0.0"
basis = "sto-6g"
driver = PySCFDriver(atom=atom,  basis=basis)

problem = driver.run()

active_electrons = 4
active_transformer = ActiveSpaceTransformer(num_electrons=active_electrons, num_spatial_orbitals=num_spartial_orbital)
active_problem = active_transformer.transform(problem)

seed = 170
algorithm_globals.random_seed = seed

mol = gto.M(atom = atom, basis=basis, verbose = 0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, ncas=num_spartial_orbital, nelecas=active_electrons)
mc.kernel()

frozen_core_energy = mc.e_tot - mc.e_cas
core = frozen_core_energy
print("Frozen-core energy =", core)


# Initialize the mapper
mapper = JordanWignerMapper()
        
# Map the electronic problem to a qubit operator
qubit_op = mapper.map(active_problem.hamiltonian.second_q_op())
        
# Initialize the UCCSD ansatz with Hartree-Fock initial state
ansatz = UCCSD(
    active_problem.num_spatial_orbitals,
    active_problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        active_problem.num_spatial_orbitals,
        active_problem.num_particles,
        mapper
    ),
)

vqe = VQE(Estimator(), ansatz, SLSQP())
vqe.initial_point = np.zeros(ansatz.num_parameters)
nr = active_problem.nuclear_repulsion_energy 
# Calculate the exact energy
#creating a ground state eigensolver(vqe)
#print("NR", nr)
solver = GroundStateEigensolver(mapper, vqe)
result = solver.solve(active_problem)
#print(f"Computing for bond length: {d:.2f} Å")

gr = result.total_energies

print(f"uccsd energy = {gr}")
uccsd_energy = float(np.asarray(gr, dtype=float).reshape(-1)[0])
# Extract the ground state wavefunction parameters

psi_vqe = result.raw_result.optimal_point

from qiskit.circuit import ParameterVector
ansatz = ansatz.assign_parameters(dict(zip(ansatz.parameters, psi_vqe)), inplace=False)

from qiskit_aer import AerSimulator
simulator = AerSimulator(method='statevector')
qc = transpile(ansatz, simulator)
qc.save_statevector()

aer_result = simulator.run(qc).result()
statevector = aer_result.get_statevector(qc)


# Initialize the matrix M
num_excitations = len(excitations)
M = np.zeros((num_excitations +1, num_excitations +1), dtype=complex)
S = np.zeros((num_excitations +1, num_excitations +1), dtype=complex)
# Compute the matrix elements
for i in range(len(excitations) +1):
    for j in range(len(excitations)+1):
        G_i = excitations[i-1]
        G_j = excitations[j-1]
        op_i = mapper.map(G_i)
        op_j = mapper.map(G_j)
        op = op_i.adjoint()@qubit_op@op_j
        oj = qubit_op@op_j
        oi = op_i.adjoint()@qubit_op
                
        if i == j == 0:
            M[i, j] = Statevector(statevector).expectation_value(qubit_op)
            S[i, j] = 1.0
        elif i==0 and j > 0:
            M[i, j] = Statevector(statevector).expectation_value(oj)
            S[i, j] = Statevector(statevector).expectation_value(op_j)
        elif i>0 and j==0:
            M[i, j] = Statevector(statevector).expectation_value(oi)
            S[i, j] = Statevector(statevector).expectation_value(op_i.adjoint())
        else:
            M[i, j] = Statevector(statevector).expectation_value(op)
            S[i, j] = Statevector(statevector).expectation_value(op_i.adjoint()@op_j)                      

#cond_num = np.linalg.cond(S)
#print("condition number:", cond_num)

eig, ev = scipy.linalg.eigh(M, S) 
#eigval_exact.append(eigval)
print("Eigenvalues from QSE:", eig + core)
qse_no_shot_first = float(np.asarray(np.real_if_close(eig + core), dtype=float)[0])



from qiskit_aer.primitives import Estimator


def plot_shot_stats(shots, means, variances, no_shot_value, plot_path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    x = np.asarray(shots, dtype=float)
    y = np.asarray(means, dtype=float)
    var = np.asarray(variances, dtype=float)
    yerr = np.sqrt(np.clip(var, a_min=0.0, a_max=None))

    style_params = {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 15,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
    with mpl.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        ax.axhline(
            float(no_shot_value),
            color="black",
            linestyle="-",
            linewidth=1.8,
            label="QSE (no shots)",
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color="blue",
            ecolor="blue",
            elinewidth=1.4,
            capsize=4,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Shots")
        ax.set_ylabel("Energy (Ha)")
        ax.set_xticks(x)
        shot_labels = []
        for v in x:
            iv = int(v)
            if iv == 1000:
                shot_labels.append(r"$10^3$")
            elif iv == 10000:
                shot_labels.append(r"$10^4$")
            else:
                shot_labels.append(str(iv))
        ax.set_xticklabels(shot_labels)
        #ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        #ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        #ax.legend(loc="best", handlelength=2.6)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


num_runs = 10
shot_values = [500, 1000, 10000]
means = []
variances = []
shot_records = []

for shot in shot_values:
    estimator = Estimator(
        run_options={"shots": int(shot)},
        transpile_options={"seed_transpiler": 42},
        approximation=True,
    )
    run_values = []
    for t in range(num_runs):
        Mn = np.zeros((num_excitations + 1, num_excitations + 1), dtype=complex)
        Sn = np.zeros((num_excitations + 1, num_excitations + 1), dtype=complex)
        for i in range(len(excitations) + 1):
            for j in range(len(excitations) + 1):
                G_i = excitations[i - 1]
                G_j = excitations[j - 1]
                op_i = mapper.map(G_i)
                op_j = mapper.map(G_j)
                op = op_i.adjoint() @ qubit_op @ op_j
                oj = qubit_op @ op_j
                oi = op_i.adjoint() @ qubit_op

                if i == j == 0:
                    Mn[i, j] = estimator.run(ansatz, qubit_op).result().values[0]
                    Sn[i, j] = estimator.run(ansatz, I).result().values[0]
                elif i == 0 and j > 0:
                    Mn[i, j] = estimator.run(ansatz, oj).result().values[0]
                    Sn[i, j] = estimator.run(ansatz, op_j).result().values[0]
                elif i > 0 and j == 0:
                    Mn[i, j] = estimator.run(ansatz, oi).result().values[0]
                    Sn[i, j] = estimator.run(ansatz, op_i.adjoint()).result().values[0]
                else:
                    Mn[i, j] = estimator.run(ansatz, op).result().values[0]
                    Sn[i, j] = estimator.run(ansatz, op_i.adjoint() @ op_j).result().values[0]

        Mn = np.nan_to_num(Mn, nan=0.0, posinf=0.0, neginf=0.0)
        Sn = np.nan_to_num(Sn, nan=0.0, posinf=0.0, neginf=0.0)
        Mn = 0.5 * (Mn + Mn.conj().T)
        Sn = 0.5 * (Sn + Sn.conj().T)

        try:
            eigval_shot, _ = scipy.linalg.eigh(Mn, Sn)
        except Exception:
            eigval_shot, _ = scipy.linalg.eig(Mn, Sn)
            eigval_shot = np.real_if_close(np.asarray(eigval_shot))
            if np.iscomplexobj(eigval_shot):
                eigval_shot = np.real(eigval_shot)
            order = np.argsort(np.asarray(eigval_shot, dtype=float))
            eigval_shot = np.asarray(eigval_shot, dtype=float)[order]

        eig_with_core = np.asarray(np.real_if_close(eigval_shot + core), dtype=float)
        first_eig = float(eig_with_core[0])
        run_values.append(first_eig)
        print(f"shots={int(shot)} run={t + 1:02d}/{num_runs}: first_eig={first_eig:.12f}")

    arr = np.asarray(run_values, dtype=float)
    mean = float(arr.mean())
    variance = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
    means.append(mean)
    variances.append(variance)
    shot_records.append((int(shot), [float(v) for v in arr.tolist()], mean, variance))
    print(f"shots={int(shot)}: mean={mean:.12f}, variance={variance:.12e}")


plot_path = Path(__file__).resolve().with_name("ongoing_nh3_qse_shots_plot.png")
plot_shot_stats(
    shots=shot_values,
    means=means,
    variances=variances,
    no_shot_value=qse_no_shot_first,
    plot_path=plot_path,
)

txt_path = Path(__file__).resolve().with_name("ongoing_nh3_qse_shots.txt")
report_lines = [
    "NH3 QSE Shot Statistics (ongoing_qse)",
    f"HF energy (Ha): {float(mf.e_tot):.12f}",
    f"UCCSD energy (Ha, no shots): {uccsd_energy:.12f}",
    f"QSE first eigenvalue (Ha, no shots): {qse_no_shot_first:.12f}",
    f"shot_values: {shot_values}",
    f"runs_per_shot: {num_runs}",
    "",
]
for shot, values, mean, variance in shot_records:
    report_lines.append(
        f"shots={shot}: mean={mean:.12f}, variance={variance:.12e}"
    )
    report_lines.append(f"  first_eig_values: {values}")
report_lines.extend(["", f"Plot file: {plot_path}", ""])
txt_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Wrote stats TXT: {txt_path}")
print(f"Wrote plot: {plot_path}")
