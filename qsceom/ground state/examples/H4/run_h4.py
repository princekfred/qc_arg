#!/usr/bin/env python3
"""Run linear H4 ground-state ADAPT-VQE at 3.0 Angstrom spacing by default."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_adapt_vqe():
    script_path = Path(__file__).resolve()
    ground_state_dir = script_path.parents[2]
    if str(ground_state_dir) not in sys.path:
        sys.path.insert(0, str(ground_state_dir))

    try:
        from adaptvqe import adapt_vqe
        from qsceom_par import qsc_eom
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Could not import ground-state modules from qsceom/ground state."
        ) from exc
    return adapt_vqe, qsc_eom


def build_parser():
    parser = argparse.ArgumentParser(
        description="Linear H4 ground-state ADAPT-VQE example (default spacing: 3.0 A)."
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=3.0,
        help="H-H spacing in Angstrom for linear H4.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="Basis set used in the calculation.",
    )
    parser.add_argument(
        "--adapt-it",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        help="One or more ADAPT iteration counts to run (e.g. --adapt-it 2 3 4).",
    )
    parser.add_argument(
        "--active-electrons",
        type=int,
        default=4,
        help="Number of active electrons.",
    )
    parser.add_argument(
        "--active-orbitals",
        type=int,
        default=4,
        help="Number of active orbitals.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total molecular charge.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2S spin value used by PySCF (0 for singlet).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Measurement shots (0 means analytic mode).",
    )
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=500,
        help="Maximum optimizer iterations per ADAPT step.",
    )
    parser.add_argument(
        "--fci-nroots",
        type=int,
        default=4,
        help="Number of FCI roots to compute for the reference section.",
    )
    parser.add_argument(
        "--skip-fci",
        action="store_true",
        help="Skip computing/printing the FCI reference section.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output report file path.",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help=(
            "Optional metrics plot image path (default: h4_error_plot.png). "
            "Overlap fidelities are saved as <stem>_fidelity<suffix>."
        ),
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    return parser.parse_args(argv)


def _compute_fci(symbols, geometry, basis, charge, spin, nroots):
    try:
        import numpy as np
        from pyscf import fci, gto, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FCI section requires PySCF and NumPy. Install with: "
            "`pip install numpy pyscf`."
        ) from exc

    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(len(symbols))]
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.build()

    if mol.spin == 0 and mol.nelectron % 2 == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.level_shift = 0.5
    mf.diis_space = 12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run()

    cisolver = fci.FCI(mf)
    cisolver.nroots = int(nroots)
    energies, _ = cisolver.kernel()
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    return energies


def _build_fci_overlap_context(
    symbols,
    geometry,
    basis,
    charge,
    active_electrons,
    active_orbitals,
):
    try:
        import numpy as np
        import pennylane as qml
        from exc import inite
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fidelity calculation requires PennyLane and NumPy. Install with: "
            "`pip install numpy pennylane pyscf`."
        ) from exc

    coordinates = np.asarray(geometry, dtype=float)
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
    matrix = qml.matrix(hamiltonian, wire_order=range(qubits))
    _, eigvecs = np.linalg.eigh(np.asarray(matrix))
    fci_states = np.asarray(eigvecs, dtype=complex)
    for idx in range(fci_states.shape[1]):
        fci_states[:, idx] /= np.linalg.norm(fci_states[:, idx])
    fci_state = np.asarray(fci_states[:, 0], dtype=complex)
    fci_state /= np.linalg.norm(fci_state)

    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)
    excitation_configs = inite(active_electrons, qubits)
    dev = qml.device("lightning.qubit", wires=qubits)

    def _apply_ansatz(curr_params, curr_ash_excitation):
        if curr_ash_excitation is None:
            qml.UCCSD(curr_params, wires, s_wires, d_wires, hf_state)
            return
        for i, excitation in enumerate(curr_ash_excitation):
            if len(excitation) == 4:
                qml.FermionicDoubleExcitation(
                    weight=curr_params[i],
                    wires1=list(range(excitation[0], excitation[1] + 1)),
                    wires2=list(range(excitation[2], excitation[3] + 1)),
                )
            elif len(excitation) == 2:
                qml.FermionicSingleExcitation(
                    weight=curr_params[i],
                    wires=list(range(excitation[0], excitation[1] + 1)),
                )
            else:
                raise ValueError("Each excitation must have length 2 or 4")

    @qml.qnode(dev)
    def adapt_state(curr_params, curr_ash_excitation):
        qml.BasisState(hf_state, wires=range(qubits))
        _apply_ansatz(curr_params, curr_ash_excitation)
        return qml.state()

    @qml.qnode(dev)
    def qsceom_basis_state(curr_params, curr_ash_excitation, occ):
        for w in occ:
            qml.X(wires=w)
        _apply_ansatz(curr_params, curr_ash_excitation)
        return qml.state()

    return {
        "np": np,
        "fci_state": fci_state,
        "fci_states": fci_states,
        "adapt_state_fn": adapt_state,
        "qsceom_basis_state_fn": qsceom_basis_state,
        "excitation_configs": excitation_configs,
    }


def _compute_adapt_fci_fidelity(params, ash_excitation, overlap_context):
    np = overlap_context["np"]
    fci_state = overlap_context["fci_state"]
    adapt_state_fn = overlap_context["adapt_state_fn"]

    adapt_state = np.asarray(adapt_state_fn(np.asarray(params), ash_excitation), dtype=complex)
    adapt_state /= np.linalg.norm(adapt_state)
    return float(abs(np.vdot(fci_state, adapt_state)) ** 2)


def _compute_qsceom_fci_fidelity(
    params,
    ash_excitation,
    qsceom_ground_vec,
    overlap_context,
):
    fci_state = overlap_context["fci_state"]
    np = overlap_context["np"]
    qsceom_state = _build_qsceom_state_from_vec(
        params=params,
        ash_excitation=ash_excitation,
        qsceom_vec=qsceom_ground_vec,
        overlap_context=overlap_context,
    )
    return float(abs(np.vdot(fci_state, qsceom_state)) ** 2)


def _build_qsceom_state_from_vec(
    params,
    ash_excitation,
    qsceom_vec,
    overlap_context,
):
    np = overlap_context["np"]
    fci_states = overlap_context["fci_states"]
    basis_state_fn = overlap_context["qsceom_basis_state_fn"]
    excitation_configs = overlap_context["excitation_configs"]

    coeffs = np.asarray(qsceom_vec, dtype=complex)
    if len(coeffs) != len(excitation_configs):
        raise ValueError("QSC-EOM eigenvector dimension does not match excitation configs")

    qsceom_state = np.zeros(fci_states.shape[0], dtype=complex)
    for coeff, occ in zip(coeffs, excitation_configs):
        basis_state = np.asarray(
            basis_state_fn(np.asarray(params), ash_excitation, occ),
            dtype=complex,
        )
        qsceom_state = qsceom_state + coeff * basis_state
    qsceom_state /= np.linalg.norm(qsceom_state)
    return qsceom_state


def _plot_metrics(
    iterations,
    adapt_errors,
    qsceom_errors,
    adapt_max_gradients,
    adapt_fidelities,
    qsceom_fidelities,
    plot_path,
):
    try:
        import math
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install with: `pip install matplotlib`."
        ) from exc

    def _finite_xy(xs, ys, positive_only=False):
        x_out = []
        y_out = []
        for x, y in zip(xs, ys):
            y = float(y)
            if not math.isfinite(y):
                continue
            if positive_only and y <= 0.0:
                continue
            x_out.append(int(x))
            y_out.append(y)
        return x_out, y_out

    def _save_publication_figure(fig, output_path):
        save_path = output_path if output_path.suffix else output_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if save_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            save_kwargs["dpi"] = 600
        fig.savefig(save_path, **save_kwargs)

        return save_path

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
        "legend.fontsize": 9,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }

    with mpl.rc_context(style_params):
        fig, ax1 = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        ax1.set_xlabel("ADAPT iterations")
        ax1.set_ylabel("Error from FCI (Ha)")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        #ax1.grid(True, which="major", alpha=0.25, linewidth=0.6)
        #ax1.grid(True, which="minor", alpha=0.12, linewidth=0.4)

        has_positive = False
        for series in (adapt_errors, qsceom_errors, adapt_max_gradients):
            if any(math.isfinite(float(v)) and float(v) > 0.0 for v in series):
                has_positive = True
                break
        if has_positive:
            ax1.set_yscale("log")

        lines = []
        labels = []

        x, y = _finite_xy(iterations, adapt_errors, positive_only=has_positive)
        if y:
            line = ax1.plot(
                x,
                y,
                color="#1f77b4",
                marker="o",
                linestyle="-",
                label="ADAPT-VQE energy error",
            )[0]
            lines.append(line)
            labels.append(line.get_label())

        x, y = _finite_xy(iterations, qsceom_errors, positive_only=has_positive)
        if y:
            line = ax1.plot(
                x,
                y,
                color="#b30000",
                marker="o",
                linestyle="-",
                label="q-sc-EOM energy error",
            )[0]
            lines.append(line)
            labels.append(line.get_label())

        x, y = _finite_xy(iterations, adapt_max_gradients, positive_only=has_positive)
        if y:
            line = ax1.plot(
                x,
                y,
                color="#466964",
                marker="^",
                linestyle="--",
                label="ADAPT max gradient",
            )[0]
            lines.append(line)
            labels.append(line.get_label())

        if lines:
            ax1.legend(lines, labels, loc="best", handlelength=2.6)

        normalized_plot_path = _save_publication_figure(fig, plot_path)
        plt.close(fig)

        fidelity_plot_path = normalized_plot_path.with_name(
            f"{normalized_plot_path.stem}_fidelity{normalized_plot_path.suffix}"
        )
        fidelity_fig, fidelity_ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        fidelity_ax.set_xlabel("ADAPT iterations")
        fidelity_ax.set_ylabel("Overlap fidelity")
        fidelity_ax.set_ylim(0.0, 1.01)
        fidelity_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fidelity_ax.xaxis.set_minor_locator(AutoMinorLocator())
        fidelity_ax.yaxis.set_minor_locator(AutoMinorLocator())
        fidelity_ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        fidelity_ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)

        fidelity_lines = []
        x, y = _finite_xy(iterations, adapt_fidelities)
        if y:
            line = fidelity_ax.plot(
                x,
                y,
                color="#1f77b4",
                marker="o",
                linestyle="-",
                label="ADAPT-FCI fidelity",
            )[0]
            fidelity_lines.append(line)

        x, y = _finite_xy(iterations, qsceom_fidelities)
        if y:
            line = fidelity_ax.plot(
                x,
                y,
                color="#b30000",
                marker="x",
                linestyle="-",
                label="q-sc-EOM-FCI fidelity",
            )[0]
            fidelity_lines.append(line)

        if fidelity_lines:
            fidelity_ax.legend(loc="best", handlelength=2.6)

        _save_publication_figure(fidelity_fig, fidelity_plot_path)
        plt.close(fidelity_fig)


def main():
    args = parse_args()
    if args.fci_nroots <= 0:
        raise ValueError("--fci-nroots must be > 0")
    adapt_vqe, qsc_eom = _load_adapt_vqe()

    symbols = ["H", "H", "H", "H"]
    geometry = [[0.0, 0.0, i * args.spacing] for i in range(len(symbols))]

    fci_energies = None
    fci_ground = None
    overlap_context = None
    if not args.skip_fci:
        fci_energies = _compute_fci(
            symbols=symbols,
            geometry=geometry,
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
            nroots=args.fci_nroots,
        )
        fci_ground = float(fci_energies[0])
        overlap_context = _build_fci_overlap_context(
            symbols=symbols,
            geometry=geometry,
            basis=args.basis,
            charge=args.charge,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
        )

    shots = None if args.shots == 0 else args.shots
    reports = ["===== H4 Ground-State ADAPT-VQE ====="]
    iterations = []
    adapt_errors = []
    qsceom_errors = []
    adapt_max_gradients = []
    adapt_fidelities = []
    qsceom_fidelities = []
    for adapt_it in args.adapt_it:
        params, ash_excitation, energies, adapt_gradients = adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=adapt_it,
            basis=args.basis,
            charge=args.charge,
            spin=args.spin,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            shots=shots,
            optimizer_maxiter=args.optimizer_maxiter,
            return_max_gradients=True,
        )
        eigvals, eigvecs = qsc_eom(
            symbols=symbols,
            coordinates=geometry,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            charge=args.charge,
            params=params,
            ash_excitation=ash_excitation,
            shots=args.shots,
            basis=args.basis,
        )

        adapt_ground = float(energies[-1])
        qsc_ground = float(eigvals[0])
        overlap_fidelity = None
        if overlap_context is not None:
            overlap_fidelity = _compute_adapt_fci_fidelity(
                params=params,
                ash_excitation=ash_excitation,
                overlap_context=overlap_context,
            )
            qsceom_overlap_fidelity = _compute_qsceom_fci_fidelity(
                params=params,
                ash_excitation=ash_excitation,
                qsceom_ground_vec=eigvecs[:, 0],
                overlap_context=overlap_context,
            )
        else:
            qsceom_overlap_fidelity = None
        lines = [
            f"H-H spacing (Angstrom): {args.spacing}",
            f"Basis: {args.basis}",
            f"ADAPT iterations: {adapt_it}",
            f"Adapt gr energy (Ha): {adapt_ground}",
            f"qsceom gr energy (Ha): {qsc_ground}",
        ]
        if len(adapt_gradients) > 0:
            lines.append(
                f"ADAPT max gradient: {adapt_gradients[-1]}"
            )
            adapt_max_gradients.append(float(adapt_gradients[-1]))
        else:
            adapt_max_gradients.append(float("nan"))
        if overlap_fidelity is not None:
            lines.append(f"adapt-FCI overlap fidelity: {overlap_fidelity}")
            adapt_fidelities.append(float(overlap_fidelity))
        else:
            adapt_fidelities.append(float("nan"))
        if qsceom_overlap_fidelity is not None:
            lines.append(f"qsceom-FCI overlap fidelity: {qsceom_overlap_fidelity}")
            qsceom_fidelities.append(float(qsceom_overlap_fidelity))
        else:
            qsceom_fidelities.append(float("nan"))
        if fci_ground is not None:
            adapt_err = abs(adapt_ground - fci_ground)
            qsceom_err = abs(qsc_ground - fci_ground)
            lines.extend(
                [
                    f"ADAPT gr error (Ha): {adapt_err}",
                    f"qsceom gr error (Ha): {qsceom_err}",
                ]
            )
            adapt_errors.append(float(adapt_err))
            qsceom_errors.append(float(qsceom_err))
        else:
            adapt_errors.append(float("nan"))
            qsceom_errors.append(float("nan"))
        iterations.append(int(adapt_it))
        reports.append("\n".join(lines))

    if not args.skip_fci:
        fci_lines = [
            "===== FCI Reference =====",
            f"H-H spacing (Angstrom): {args.spacing}",
            f"Basis: {args.basis}",
            f"Charge: {args.charge}",
            f"Spin (2S): {args.spin}",
            f"Requested FCI roots: {args.fci_nroots}",
            #f"FCI energies (Ha): {fci_energies}",
            f"FCI gr energy (Ha): {fci_energies[0]}",
        ]
        reports.append("\n".join(fci_lines))

    report = "\n\n".join(reports) + "\n"
    print(report, end="")

    if args.output_file is None:
        output_path = Path(__file__).resolve().parent / "h4_ground_output.txt"
    else:
        output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    if args.plot_file is None:
        plot_path = output_path.with_name("h4_error_plot.png")
    else:
        plot_path = Path(args.plot_file)
    _plot_metrics(
        iterations=iterations,
        adapt_errors=adapt_errors,
        qsceom_errors=qsceom_errors,
        adapt_max_gradients=adapt_max_gradients,
        adapt_fidelities=adapt_fidelities,
        qsceom_fidelities=qsceom_fidelities,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()
