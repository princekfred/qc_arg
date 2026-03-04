#!/usr/bin/env python3
"""Plot NH3 and NH3qsceom shot errors on a single figure.

NH3 errors are computed as:
    error(shots) = mean_first_eig(shots) - no_shot_first_eig

If the second report already contains explicit per-shot errors, they are used.
Otherwise, the same formula is applied using its no-shot reference and means.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _resolve_input_file(path: Path, script_dir: Path) -> Path:
    if path.is_file():
        return path

    candidates = [
        Path.cwd() / path,
        script_dir / path,
        script_dir / path.name,
        script_dir.parent / path.name,
        script_dir.parent / "qsceom/ground state/examples/shots" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return path


def _parse_shot_means(text: str) -> dict[int, float]:
    means: dict[int, float] = {}
    pattern = re.compile(rf"shots\s*=\s*(\d+)\s*:\s*mean\s*=\s*({_FLOAT_RE})")
    for match in pattern.finditer(text):
        means[int(match.group(1))] = float(match.group(2))
    return means


def _parse_shot_errors(text: str) -> dict[int, float]:
    errors: dict[int, float] = {}
    pattern = re.compile(rf"shots\s*=\s*(\d+)\s*:\s*.*?\berror\s*=\s*({_FLOAT_RE})")
    for match in pattern.finditer(text):
        errors[int(match.group(1))] = float(match.group(2))
    return errors


def _parse_shot_variances(text: str) -> dict[int, float]:
    variances: dict[int, float] = {}
    pattern = re.compile(rf"shots\s*=\s*(\d+)\s*:\s*.*?\bvariance\s*=\s*({_FLOAT_RE})")
    for match in pattern.finditer(text):
        variances[int(match.group(1))] = float(match.group(2))
    return variances


def _parse_no_shot_reference(text: str) -> float | None:
    patterns = [
        rf"QSE first eigenvalue\s*\(Ha,\s*no shots\)\s*:\s*({_FLOAT_RE})",
        rf"first_eig\s*:\s*({_FLOAT_RE})",
        rf"qse gr\s*\(Ha\)\s*:\s*({_FLOAT_RE})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return float(match.group(1))
    return None


def parse_report_errors(path: Path) -> tuple[dict[int, float], float | None, dict[int, float]]:
    text = path.read_text(encoding="utf-8")
    variances = _parse_shot_variances(text)
    explicit_errors = _parse_shot_errors(text)
    if explicit_errors:
        return explicit_errors, _parse_no_shot_reference(text), variances

    means = _parse_shot_means(text)
    no_shot = _parse_no_shot_reference(text)
    if not means:
        raise ValueError(f"No shot means found in {path}")
    if no_shot is None:
        raise ValueError(
            f"Could not find a no-shot first-eigenvalue reference in {path}"
        )
    computed_errors = {shot: mean - no_shot for shot, mean in means.items()}
    return computed_errors, no_shot, variances


def _shot_labels(shots: list[int]) -> list[str]:
    labels = []
    for shot in shots:
        if shot == 1000:
            labels.append(r"$10^3$")
        elif shot == 10000:
            labels.append(r"$10^4$")
        else:
            labels.append(str(shot))
    return labels


def plot_errors(
    nh3_errors: dict[int, float],
    nh3_variances: dict[int, float],
    nh3qsceom_errors: dict[int, float],
    nh3qsceom_variances: dict[int, float],
    output_path: Path,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib and numpy. Install with: "
            "`pip install matplotlib numpy`."
        ) from exc

    nh3_shots = np.asarray(sorted(nh3_errors), dtype=float)
    nh3_vals = np.asarray([nh3_errors[int(s)] for s in nh3_shots], dtype=float)
    nh3_yerr = np.asarray(
        [np.sqrt(max(float(nh3_variances.get(int(s), 0.0)), 0.0)) for s in nh3_shots],
        dtype=float,
    )
    nh3qsceom_shots = np.asarray(sorted(nh3qsceom_errors), dtype=float)
    nh3qsceom_vals = np.asarray([nh3qsceom_errors[int(s)] for s in nh3qsceom_shots], dtype=float)
    nh3qsceom_yerr = np.asarray(
        [
            np.sqrt(max(float(nh3qsceom_variances.get(int(s), 0.0)), 0.0))
            for s in nh3qsceom_shots
        ],
        dtype=float,
    )

    all_shots = sorted(set(int(s) for s in nh3_shots) | set(int(s) for s in nh3qsceom_shots))
    xticks = np.asarray(all_shots, dtype=float)

    style_params = {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 15,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "legend.fontsize": 11,
    }

    with mpl.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        ax.axhline(0.0, color="#1f1f1f", linewidth=1.6)
        ax.errorbar(
            nh3_shots,
            nh3_vals,
            yerr=nh3_yerr,
            fmt="o--",
            ecolor="#1F51B4",
            capsize=4,
            color="#1F51B4",
            label="QSE error",
        )
        ax.errorbar(
            nh3qsceom_shots,
            nh3qsceom_vals,
            yerr=nh3qsceom_yerr,
            fmt="o--",
            ecolor="#b30000",
            capsize=4,
            color="#b30000",
            label="q-sc-EOM error",
        )
        ax.set_xscale("log")
        ax.set_xticks(xticks)
        ax.set_xticklabels(_shot_labels(all_shots))
        ax.set_xlabel("Shot count")
        ax.set_ylabel("Error (Ha)")
        ax.legend(loc="best", handlelength=2.6)
        ax.set_ylim(-0.8, 0.05)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute shot errors from report text files and plot NH3 + NH3qsceom "
            "errors on one figure."
        )
    )
    parser.add_argument(
        "--nh3-file",
        type=Path,
        default=Path("nh3_qse_shots.txt"),
        help="Path to NH3 QSE shot report text file.",
    )
    parser.add_argument(
        "--nh3qsceom-file",
        type=Path,
        default=Path("nh3_qsceom_shots.txt"),
        help="Path to NH3 QSC-EOM shot report text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("nh3_shots_error_.png"),
        help="Output image path for combined error plot.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    nh3_file = _resolve_input_file(args.nh3_file, script_dir)
    nh3qsceom_file = _resolve_input_file(args.nh3qsceom_file, script_dir)
    if not nh3_file.is_file():
        raise FileNotFoundError(f"NH3 report not found: {nh3_file}")
    if not nh3qsceom_file.is_file():
        raise FileNotFoundError(f"NH3 QSC-EOM report not found: {nh3qsceom_file}")

    nh3_errors, nh3_ref, nh3_variances = parse_report_errors(nh3_file)
    nh3qsceom_errors, nh3qsceom_ref, nh3qsceom_variances = parse_report_errors(nh3qsceom_file)
    plot_errors(
        nh3_errors=nh3_errors,
        nh3_variances=nh3_variances,
        nh3qsceom_errors=nh3qsceom_errors,
        nh3qsceom_variances=nh3qsceom_variances,
        output_path=args.output,
    )

    print(f"NH3 no-shot reference: {nh3_ref}")
    for shot in sorted(nh3_errors):
        print(f"NH3 shots={shot}: error={nh3_errors[shot]:.12f}")
        if shot in nh3_variances:
            print(f"NH3 shots={shot}: variance={nh3_variances[shot]:.12e}")
    print(f"NH3qsceom no-shot reference: {nh3qsceom_ref}")
    for shot in sorted(nh3qsceom_errors):
        print(f"NH3qsceom shots={shot}: error={nh3qsceom_errors[shot]:.12f}")
        if shot in nh3qsceom_variances:
            print(f"NH3qsceom shots={shot}: variance={nh3qsceom_variances[shot]:.12e}")
    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()
