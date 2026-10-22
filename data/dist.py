#!/usr/bin/env python3
"""Degree-of-distortion analysis

The geometric components are:
    - cage rotation
    - residual non-rigid shape deformation
    - local isotropic bond-scale strain
    - In off-centering change
    - coordination-number change
    - ligand exchange
    - In-Br bond-length dispersion
    - Br-cage shift relative to In
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

KB_EV_PER_K = 8.617333262145e-5
GPA_A3_TO_EV = 0.006241509074
EPS = 1.0e-14


FEATURE_LABELS = {
    "rotation": "Cage rotation",
    "shape": "Residual shape deformation",
    "scale": "Local isotropic strain",
    "offcentering": "In off-centering change",
    "coordination": "Coordination change",
    "exchange": "Ligand exchange",
    "bond_dispersion": "Bond-length dispersion",
    "cage_shift": "Br-cage shift relative to In",
}


REQUIRED_COLUMNS = {
    "temperature_K",
    "frame",
    "in_index",
    "mean_InBr_bond_A",
    "std_InBr_bond_A",
    "best_local_scale",
    "rotation_angle_deg",
    "residual_shape_rmsd_A",
    "In_offcentering_vector_change_A",
    "reference_effective_cn",
    "delta_effective_cn",
    "ligand_exchange",
    "Br_cage_relative_to_In_shift_A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate energy-coupled degree of distortion for In-Br frameworks."
    )
    parser.add_argument(
        "--structure-pattern",
        default="InBr_framework*_data_*K.csv",
        help="Glob pattern for per-In structural CSV files.",
    )
    parser.add_argument(
        "--energy-pattern",
        default="../../{T}k/OUTCAR",
        help=(
            "Energy-file pattern. {T} is replaced by the integer temperature. "
            "Use an empty string to disable energy analysis."
        ),
    )
    parser.add_argument(
        "--energy-format",
        choices=("outcar_toten", "oszicar_E", "simple"),
        default="outcar_toten",
        help=(
            "outcar_toten: VASP 'free energy TOTEN'; oszicar_E: MD E= field; "
            "simple: one energy value per non-comment line or last numeric column."
        ),
    )
    parser.add_argument(
        "--reference",
        default="",
        help="Optional structure file used to determine the number of atoms.",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=0,
        help="Number of atoms. Overrides --reference when >0.",
    )
    parser.add_argument(
        "--ensemble",
        choices=("nvt", "npt"),
        default="nvt",
        help="Use U for NVT or H=U+PV for NPT thermodynamic analysis.",
    )
    parser.add_argument(
        "--pressure-gpa",
        type=float,
        default=0.0,
        help="External pressure for NPT enthalpy, in GPa.",
    )
    parser.add_argument(
        "--baseline-max-temperature",
        type=float,
        default=270.0,
        help="Maximum temperature used to define the low-temperature baseline.",
    )
    parser.add_argument(
        "--baseline-polynomial-degree",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Polynomial degree for low-temperature thermal/energy baselines.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
        help="Number of sampled frames per uncertainty block.",
    )
    parser.add_argument(
        "--free-energy-bins",
        type=int,
        default=40,
        help="Number of bins in F(D) profiles; <=0 disables profile output.",
    )
    parser.add_argument(
        "--distorted-percentile",
        type=float,
        default=95.0,
        help="Low-T percentile defining a distorted local environment.",
    )
    parser.add_argument(
        "--onset-sigma",
        type=float,
        default=3.0,
        help="Anomaly threshold in baseline standard deviations.",
    )
    parser.add_argument(
        "--onset-persistence",
        type=int,
        default=2,
        help="Number of consecutive temperatures required for onset detection.",
    )
    parser.add_argument(
        "--energy-index-mode",
        choices=("auto", "direct", "rank", "proportional"),
        default="auto",
        help="How structural frame indices are mapped onto energy samples.",
    )
    parser.add_argument(
        "--energy-offset",
        type=int,
        default=0,
        help=(
            "Integer offset added to structural frame indices for direct energy "
            "mapping. Use 1 or -1 only after checking the trajectory/OUTCAR count."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="InBr_degree_of_distortion",
        help="Prefix for all generated files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate PNG plots.",
    )
    return parser.parse_args()


def temperature_from_filename(path: Path) -> float | None:
    match = re.search(r"(?:_|-)(\d+(?:\.\d+)?)K(?:\.|$)", path.name)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)K", path.name)
    return float(match.group(1)) if match else None


def discover_structure_files(pattern: str) -> list[Path]:
    from glob import glob

    paths = [Path(p) for p in glob(pattern)]
    recognized = [(temperature_from_filename(path), path) for path in paths]
    recognized = [(t, p) for t, p in recognized if t is not None]
    recognized.sort(key=lambda item: item[0])
    if not recognized:
        raise FileNotFoundError(f"No structural files matched: {pattern}")
    return [path for _, path in recognized]


def read_structure_data(pattern: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in discover_structure_files(pattern):
        df = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        inferred_t = temperature_from_filename(path)
        if "temperature_K" not in df or df["temperature_K"].isna().all():
            df["temperature_K"] = inferred_t
        df["source_file"] = str(path)
        frames.append(df)
    output = pd.concat(frames, ignore_index=True)
    for column in REQUIRED_COLUMNS - {"temperature_K", "frame", "in_index"}:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["temperature_K"] = pd.to_numeric(output["temperature_K"], errors="coerce")
    output["frame"] = pd.to_numeric(output["frame"], errors="coerce").astype("Int64")
    output["in_index"] = pd.to_numeric(output["in_index"], errors="coerce").astype("Int64")
    output = output.dropna(subset=["temperature_K", "frame", "in_index"])
    return output


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(np.abs(denominator) > EPS)
    return numerator / denominator


def construct_dimensionless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct transparent, dimensionless In-Br distortion components."""
    bond = df["mean_InBr_bond_A"].abs().where(df["mean_InBr_bond_A"].abs() > EPS)
    reference_cn = df["reference_effective_cn"].abs().clip(lower=1.0)
    scale = df["best_local_scale"].where(df["best_local_scale"] > EPS)

    features = pd.DataFrame(index=df.index)
    features["rotation"] = np.abs(np.deg2rad(df["rotation_angle_deg"])) / math.pi
    features["shape"] = safe_ratio(df["residual_shape_rmsd_A"].abs(), bond)
    features["scale"] = np.abs(np.log(scale))
    features["offcentering"] = safe_ratio(
        df["In_offcentering_vector_change_A"].abs(), bond
    )
    features["coordination"] = (
        df["delta_effective_cn"].abs() / reference_cn
    )
    features["exchange"] = df["ligand_exchange"].clip(lower=0.0, upper=1.0)
    features["bond_dispersion"] = safe_ratio(df["std_InBr_bond_A"].abs(), bond)
    features["cage_shift"] = safe_ratio(
        df["Br_cage_relative_to_In_shift_A"].abs(), bond
    )

    return features.replace([np.inf, -np.inf], np.nan)


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    if not math.isfinite(sigma) or sigma < 1.0e-10:
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
    if not math.isfinite(sigma) or sigma < 1.0e-10:
        sigma = 1.0
    return median, sigma


def robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Iterative sigma-clipped polynomial fit, returned in np.polyval order."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < degree + 1:
        raise ValueError(
            f"Need at least {degree + 1} finite points for degree-{degree} fit."
        )
    active = np.ones(len(x), dtype=bool)
    coeff = np.polyfit(x, y, degree)
    for _ in range(8):
        coeff = np.polyfit(x[active], y[active], degree)
        residual = y - np.polyval(coeff, x)
        _, sigma = robust_scale(residual[active])
        new_active = np.abs(residual) <= 4.0 * sigma
        if new_active.sum() < degree + 1 or np.array_equal(new_active, active):
            break
        active = new_active
    return coeff


def fit_feature_models(
    data: pd.DataFrame,
    features: pd.DataFrame,
    baseline_max_t: float,
    degree: int,
) -> tuple[dict, pd.DataFrame]:
    baseline_mask = data["temperature_K"] <= baseline_max_t
    baseline_temperatures = sorted(data.loc[baseline_mask, "temperature_K"].unique())
    if len(baseline_temperatures) < degree + 2:
        raise ValueError(
            "Insufficient low-temperature points. Increase "
            "--baseline-max-temperature or reduce polynomial degree."
        )

    models: dict[str, dict] = {}
    z_fixed = pd.DataFrame(index=data.index)
    z_excess = pd.DataFrame(index=data.index)

    for feature in features.columns:
        values = features[feature]
        low_values = values[baseline_mask]
        fixed_center, fixed_scale = robust_scale(low_values.to_numpy())

        temp_means = (
            pd.DataFrame(
                {"temperature_K": data["temperature_K"], "value": values}
            )
            .groupby("temperature_K", as_index=False)["value"]
            .mean()
        )
        fit_means = temp_means[temp_means["temperature_K"] <= baseline_max_t]
        coeff = robust_polyfit(
            fit_means["temperature_K"].to_numpy(),
            fit_means["value"].to_numpy(),
            degree,
        )
        predicted = np.polyval(coeff, data["temperature_K"].to_numpy(dtype=float))
        residual = values.to_numpy(dtype=float) - predicted
        _, residual_scale = robust_scale(residual[baseline_mask.to_numpy()])

        z_fixed[feature] = (values - fixed_center) / fixed_scale
        z_excess[feature] = residual / residual_scale
        models[feature] = {
            "fixed_center": fixed_center,
            "fixed_scale": fixed_scale,
            "thermal_coefficients": coeff.tolist(),
            "residual_scale": residual_scale,
        }

    return models, pd.concat(
        {
            "fixed": z_fixed,
            "excess": z_excess,
        },
        axis=1,
    )


def rms_available(z: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    finite = np.isfinite(z.to_numpy(dtype=float))
    count = finite.sum(axis=1)
    squared = np.where(finite, z.to_numpy(dtype=float) ** 2, 0.0)
    result = np.sqrt(np.divide(squared.sum(axis=1), count, where=count > 0))
    result[count == 0] = np.nan
    return pd.Series(result, index=z.index), pd.Series(count, index=z.index)


def block_statistics(values: Iterable[float], block_size: int) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    if block_size <= 1:
        sem = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
        return mean, std, sem, len(array)
    n_blocks = len(array) // block_size
    if n_blocks < 2:
        sem = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
        return mean, std, sem, len(array)
    trimmed = array[: n_blocks * block_size]
    blocks = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    sem = float(np.std(blocks, ddof=1) / math.sqrt(n_blocks))
    return mean, std, sem, len(array)


def parse_outcar_toten(path: Path) -> np.ndarray:
    pattern = re.compile(
        r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)", re.IGNORECASE
    )
    values: list[float] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    return np.asarray(values, dtype=float)


def parse_oszicar_E(path: Path) -> np.ndarray:
    pattern = re.compile(r"\bE=\s*([-+0-9.Ee]+)")
    values: list[float] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if "T=" not in line:
                continue
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    return np.asarray(values, dtype=float)


def parse_simple_energy(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?", stripped)
            if numbers:
                values.append(float(numbers[-1]))
    return np.asarray(values, dtype=float)


def read_energy(path: Path, energy_format: str) -> np.ndarray:
    if energy_format == "outcar_toten":
        return parse_outcar_toten(path)
    if energy_format == "oszicar_E":
        return parse_oszicar_E(path)
    return parse_simple_energy(path)


def format_temperature_for_path(temperature: float) -> str:
    if abs(temperature - round(temperature)) < 1.0e-8:
        return str(int(round(temperature)))
    return f"{temperature:g}"


def determine_n_atoms(args: argparse.Namespace) -> int:
    if args.n_atoms > 0:
        return args.n_atoms
    if args.reference:
        try:
            from ase.io import read

            atoms = read(args.reference)
            return len(atoms)
        except Exception as exc:
            raise RuntimeError(
                f"Could not determine atom count from {args.reference}: {exc}"
            ) from exc
    return 0


def map_energy_to_frames(
    frames: np.ndarray,
    energies: np.ndarray,
    mode: str,
    offset: int = 0,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=int)
    if len(energies) == 0:
        return np.full(len(frames), np.nan)

    unique_frames = np.unique(frames)
    direct_indices = unique_frames + int(offset)
    direct_possible = (int(np.min(direct_indices)) >= 0) and (
        int(np.max(direct_indices)) < len(energies)
    )
    rank_possible = len(unique_frames) == len(energies)

    chosen = mode
    if mode == "auto":
        if direct_possible:
            chosen = "direct"
        elif rank_possible:
            chosen = "rank"
        else:
            return np.full(len(frames), np.nan)

    if chosen == "direct":
        if not direct_possible:
            raise IndexError(
                f"Maximum structural frame {np.max(unique_frames)} exceeds "
                f"the {len(energies)} available energy samples."
            )
        return energies[frames + int(offset)]

    if chosen == "rank":
        if not rank_possible:
            raise ValueError(
                "Rank mapping requires the number of unique structural frames "
                "to equal the number of energy samples."
            )
        mapping = {frame: energies[i] for i, frame in enumerate(sorted(unique_frames))}
        return np.asarray([mapping[frame] for frame in frames], dtype=float)

    # Explicit proportional mapping is available for converted trajectories, but
    # it should be used only when both files span exactly the same time interval.
    positions = np.linspace(0, len(energies) - 1, len(unique_frames))
    mapping = {
        frame: energies[int(round(position))]
        for frame, position in zip(sorted(unique_frames), positions)
    }
    return np.asarray([mapping[frame] for frame in frames], dtype=float)


def attach_energy(
    site_data: pd.DataFrame,
    frame_data: pd.DataFrame,
    args: argparse.Namespace,
    n_atoms: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not args.energy_pattern:
        return site_data, frame_data, {}
    if n_atoms <= 0:
        raise ValueError(
            "Energy analysis requires --n-atoms or --reference to normalize energy."
        )

    metadata: dict[str, dict] = {}
    site_data = site_data.copy()
    frame_data = frame_data.copy()
    site_data["thermodynamic_energy_eV"] = np.nan
    frame_data["thermodynamic_energy_eV"] = np.nan

    for temperature in sorted(frame_data["temperature_K"].unique()):
        tag = format_temperature_for_path(float(temperature))
        path = Path(args.energy_pattern.replace("{T}", tag))
        if not path.is_file():
            print(f"Warning: energy file not found for {temperature:g} K: {path}")
            continue
        energies = read_energy(path, args.energy_format)
        if len(energies) == 0:
            print(f"Warning: no energies parsed from {path}")
            continue

        frame_mask = frame_data["temperature_K"] == temperature
        frames = frame_data.loc[frame_mask, "frame"].to_numpy(dtype=int)
        mapped = map_energy_to_frames(
            frames, energies, args.energy_index_mode, args.energy_offset
        )

        if args.ensemble == "npt":
            volumes = frame_data.loc[frame_mask, "cell_volume_A3"].to_numpy(dtype=float)
            mapped = mapped + args.pressure_gpa * volumes * GPA_A3_TO_EV

        frame_data.loc[frame_mask, "thermodynamic_energy_eV"] = mapped
        energy_by_frame = dict(zip(frames, mapped))
        site_mask = site_data["temperature_K"] == temperature
        site_data.loc[site_mask, "thermodynamic_energy_eV"] = site_data.loc[
            site_mask, "frame"
        ].map(energy_by_frame)

        metadata[tag] = {
            "path": str(path),
            "n_parsed": int(len(energies)),
            "n_mapped": int(np.isfinite(mapped).sum()),
        }

    site_data["thermodynamic_energy_eV_per_atom"] = (
        site_data["thermodynamic_energy_eV"] / n_atoms
    )
    frame_data["thermodynamic_energy_eV_per_atom"] = (
        frame_data["thermodynamic_energy_eV"] / n_atoms
    )
    return site_data, frame_data, metadata


def aggregate_frames(site_data: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "degree_of_distortion_fixed": ["mean", "std", "count"],
        "degree_of_distortion_excess": ["mean", "std", "count"],
        "distorted_environment": ["mean"],
        "n_valid_fixed_features": ["mean"],
        "n_valid_excess_features": ["mean"],
    }
    if "cell_volume_A3" in site_data.columns:
        aggregations["cell_volume_A3"] = ["mean"]

    frame = site_data.groupby(["temperature_K", "frame"], as_index=False).agg(aggregations)
    frame.columns = [
        "_".join([str(part) for part in column if str(part)])
        if isinstance(column, tuple)
        else column
        for column in frame.columns
    ]
    rename = {
        "temperature_K": "temperature_K",
        "frame": "frame",
        "degree_of_distortion_fixed_mean": "mean_degree_of_distortion_fixed",
        "degree_of_distortion_fixed_std": "std_degree_of_distortion_fixed_across_In",
        "degree_of_distortion_fixed_count": "n_In",
        "degree_of_distortion_excess_mean": "mean_degree_of_distortion_excess",
        "degree_of_distortion_excess_std": "std_degree_of_distortion_excess_across_In",
        "distorted_environment_mean": "distorted_fraction",
        "n_valid_fixed_features_mean": "mean_n_valid_fixed_features",
        "n_valid_excess_features_mean": "mean_n_valid_excess_features",
        "cell_volume_A3_mean": "cell_volume_A3",
    }
    frame = frame.rename(columns=rename)
    return frame


def fit_energy_baseline(
    summary: pd.DataFrame,
    baseline_max_t: float,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = summary[
        np.isfinite(summary["mean_thermodynamic_energy_eV_per_atom"])
        & (summary["temperature_K"] <= baseline_max_t)
    ]
    if len(valid) < degree + 1:
        raise ValueError("Not enough low-temperature energy points for baseline fit.")
    coeff = robust_polyfit(
        valid["temperature_K"].to_numpy(dtype=float),
        valid["mean_thermodynamic_energy_eV_per_atom"].to_numpy(dtype=float),
        degree,
    )
    predicted = np.polyval(coeff, summary["temperature_K"].to_numpy(dtype=float))
    return coeff, predicted


def temperature_summary(
    frame_data: pd.DataFrame,
    n_atoms: int,
    block_size: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for temperature, group in frame_data.groupby("temperature_K", sort=True):
        group = group.sort_values("frame")
        fixed_mean, fixed_std, fixed_sem, n_frames = block_statistics(
            group["mean_degree_of_distortion_fixed"], block_size
        )
        excess_mean, excess_std, excess_sem, _ = block_statistics(
            group["mean_degree_of_distortion_excess"], block_size
        )
        fraction_mean, fraction_std, fraction_sem, _ = block_statistics(
            group["distorted_fraction"], block_size
        )

        row = {
            "temperature_K": float(temperature),
            "mean_degree_of_distortion_fixed": fixed_mean,
            "std_degree_of_distortion_fixed": fixed_std,
            "sem_degree_of_distortion_fixed": fixed_sem,
            "mean_degree_of_distortion_excess": excess_mean,
            "std_degree_of_distortion_excess": excess_std,
            "sem_degree_of_distortion_excess": excess_sem,
            "mean_distorted_fraction": fraction_mean,
            "std_distorted_fraction": fraction_std,
            "sem_distorted_fraction": fraction_sem,
            "distortion_fluctuation_NIn_varD": (
                float(group["n_In"].median())
                * float(np.var(group["mean_degree_of_distortion_fixed"], ddof=1))
                if len(group) > 1
                else 0.0
            ),
            "n_frames": n_frames,
            "n_In": int(round(float(group["n_In"].median()))),
        }

        if "thermodynamic_energy_eV" in group:
            valid = group.dropna(subset=["thermodynamic_energy_eV"])
            if len(valid):
                energy_pa = valid["thermodynamic_energy_eV_per_atom"].to_numpy(dtype=float)
                energy_total = valid["thermodynamic_energy_eV"].to_numpy(dtype=float)
                q = valid["mean_degree_of_distortion_fixed"].to_numpy(dtype=float)
                e_mean, e_std, e_sem, n_energy = block_statistics(energy_pa, block_size)
                row.update(
                    {
                        "mean_thermodynamic_energy_eV_per_atom": e_mean,
                        "std_thermodynamic_energy_eV_per_atom": e_std,
                        "sem_thermodynamic_energy_eV_per_atom": e_sem,
                        "n_energy_frames": n_energy,
                    }
                )
                if len(valid) > 1:
                    var_total = float(np.var(energy_total, ddof=1))
                    row["configurational_heat_capacity_kB_per_atom"] = (
                        var_total
                        / (KB_EV_PER_K**2 * float(temperature) ** 2 * n_atoms)
                    )
                    covariance = float(np.cov(q, energy_total, ddof=1)[0, 1])
                    row["energy_distortion_correlation"] = float(
                        np.corrcoef(q, energy_total)[0, 1]
                    )
                    row["covariance_dD_dT_per_K"] = covariance / (
                        KB_EV_PER_K * float(temperature) ** 2
                    )
                else:
                    row["configurational_heat_capacity_kB_per_atom"] = math.nan
                    row["energy_distortion_correlation"] = math.nan
                    row["covariance_dD_dT_per_K"] = math.nan
            else:
                row.update(
                    {
                        "mean_thermodynamic_energy_eV_per_atom": math.nan,
                        "std_thermodynamic_energy_eV_per_atom": math.nan,
                        "sem_thermodynamic_energy_eV_per_atom": math.nan,
                        "n_energy_frames": 0,
                        "configurational_heat_capacity_kB_per_atom": math.nan,
                        "energy_distortion_correlation": math.nan,
                        "covariance_dD_dT_per_K": math.nan,
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("temperature_K").reset_index(drop=True)


def add_energy_anomalies(
    summary: pd.DataFrame,
    baseline_max_t: float,
    degree: int,
) -> tuple[pd.DataFrame, list[float] | None]:
    if "mean_thermodynamic_energy_eV_per_atom" not in summary:
        return summary, None
    if not np.isfinite(summary["mean_thermodynamic_energy_eV_per_atom"]).any():
        return summary, None

    coeff, predicted = fit_energy_baseline(summary, baseline_max_t, degree)
    output = summary.copy()
    output["baseline_thermodynamic_energy_eV_per_atom"] = predicted
    output["excess_thermodynamic_energy_meV_per_atom"] = 1000.0 * (
        output["mean_thermodynamic_energy_eV_per_atom"] - predicted
    )

    valid = np.isfinite(output["mean_thermodynamic_energy_eV_per_atom"])
    temperatures = output.loc[valid, "temperature_K"].to_numpy(dtype=float)
    energies = output.loc[valid, "mean_thermodynamic_energy_eV_per_atom"].to_numpy(dtype=float)
    derivative = np.gradient(energies, temperatures) / KB_EV_PER_K
    output["derivative_heat_capacity_kB_per_atom"] = np.nan
    output.loc[valid, "derivative_heat_capacity_kB_per_atom"] = derivative
    return output, coeff.tolist()


def detect_onset(
    temperatures: np.ndarray,
    values: np.ndarray,
    baseline_max_t: float,
    sigma_multiplier: float,
    persistence: int,
) -> tuple[float | None, float, float]:
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(temperatures) & np.isfinite(values)
    temperatures = temperatures[finite]
    values = values[finite]
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    values = values[order]

    baseline = values[temperatures <= baseline_max_t]
    if len(baseline) < 2:
        return None, math.nan, math.nan
    center, scale = robust_scale(baseline)
    threshold = center + sigma_multiplier * scale
    above = values > threshold
    persistence = max(1, int(persistence))
    for i in range(len(values) - persistence + 1):
        if temperatures[i] <= baseline_max_t:
            continue
        if np.all(above[i : i + persistence]):
            return float(temperatures[i]), center, threshold
    return None, center, threshold



def candidate_peaks(
    temperatures: np.ndarray, values: np.ndarray, max_peaks: int = 4
) -> list[float]:
    """Return temperatures of simple interior local maxima, ranked by height."""
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(temperatures) & np.isfinite(values)
    temperatures = temperatures[finite]
    values = values[finite]
    if len(values) < 3:
        return []
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    values = values[order]
    maxima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            maxima.append((values[i], temperatures[i]))
    maxima.sort(reverse=True)
    return [float(temp) for _, temp in maxima[:max_peaks]]

def free_energy_profiles(
    frame_data: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    if bins <= 0:
        return pd.DataFrame()
    values = frame_data["mean_degree_of_distortion_fixed"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return pd.DataFrame()
    lower, upper = np.percentile(values, [0.5, 99.5])
    if upper <= lower:
        return pd.DataFrame()
    edges = np.linspace(lower, upper, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows: list[dict] = []

    for temperature, group in frame_data.groupby("temperature_K", sort=True):
        q = group["mean_degree_of_distortion_fixed"].to_numpy(dtype=float)
        finite = np.isfinite(q)
        q = q[finite]
        if len(q) < 5:
            continue
        counts, _ = np.histogram(q, bins=edges)
        probability = counts.astype(float) / max(counts.sum(), 1)
        free_energy = np.full(len(probability), np.nan)
        positive = probability > 0
        free_energy[positive] = -KB_EV_PER_K * float(temperature) * np.log(
            probability[positive]
        )
        if np.any(positive):
            free_energy[positive] -= np.nanmin(free_energy[positive])

        energy_by_bin = np.full(len(probability), np.nan)
        if "thermodynamic_energy_eV_per_atom" in group:
            energies = group["thermodynamic_energy_eV_per_atom"].to_numpy(dtype=float)[finite]
            indices = np.digitize(q, edges) - 1
            for bin_index in range(len(probability)):
                selected = energies[(indices == bin_index) & np.isfinite(energies)]
                if len(selected):
                    energy_by_bin[bin_index] = float(np.mean(selected))

        for i, center in enumerate(centers):
            rows.append(
                {
                    "temperature_K": float(temperature),
                    "degree_of_distortion_bin_center": center,
                    "count": int(counts[i]),
                    "probability": probability[i],
                    "free_energy_meV": 1000.0 * free_energy[i]
                    if math.isfinite(free_energy[i])
                    else math.nan,
                    "mean_thermodynamic_energy_eV_per_atom": energy_by_bin[i],
                }
            )
    return pd.DataFrame(rows)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    feature_models: dict,
    energy_metadata: dict,
    energy_coefficients: list[float] | None,
    distorted_threshold: float,
    onset_report: dict,
    n_atoms: int,
) -> None:
    with path.open("w") as handle:
        handle.write("ENERGY-COUPLED DEGREE OF DISTORTION ANALYSIS\n")
        handle.write("=" * 72 + "\n\n")
        handle.write("Definitions\n")
        handle.write("-----------\n")
        handle.write(
            "D_fixed is the RMS of robustly standardized geometric components "
            "relative to a fixed low-temperature distribution.\n"
        )
        handle.write(
            "D_excess is the RMS residual after subtracting the fitted low-T "
            "thermal trend of every component, followed by removal of the "
            "median low-T fluctuation floor.\n"
        )
        handle.write(
            "The distorted fraction is the fraction of In environments whose "
            f"D_fixed exceeds the {args.distorted_percentile:g}th percentile "
            f"of the low-T distribution ({distorted_threshold:.8g}).\n"
        )
        handle.write(
            "Excess energy is the deviation of the mean thermodynamic energy "
            "from a low-temperature polynomial baseline.\n"
        )
        handle.write(
            "The covariance response is Cov(D_fixed,E)/(k_B T^2). For a "
            "coordinate-based observable in a well-sampled canonical ensemble, "
            "this estimates d<D_fixed>/dT.\n\n"
        )

        handle.write("Parameters\n")
        handle.write("----------\n")
        for key, value in vars(args).items():
            handle.write(f"{key} = {value}\n")
        handle.write(f"n_atoms = {n_atoms}\n\n")

        handle.write("Feature definitions and baseline models\n")
        handle.write("---------------------------------------\n")
        for feature, model in feature_models.items():
            handle.write(f"{feature}: {FEATURE_LABELS[feature]}\n")
            for key, value in model.items():
                handle.write(f"  {key} = {value}\n")
        handle.write("\n")

        handle.write("Energy files\n")
        handle.write("------------\n")
        if energy_metadata:
            for temperature, item in energy_metadata.items():
                handle.write(f"{temperature}: {item}\n")
        else:
            handle.write("No energy data were analyzed.\n")
        handle.write(f"energy_baseline_coefficients = {energy_coefficients}\n\n")

        handle.write("Detected anomaly onsets\n")
        handle.write("-----------------------\n")
        for key, value in onset_report.items():
            handle.write(f"{key} = {value}\n")
        handle.write(
            "\nAn onset is reported only when the value remains above the "
            "low-temperature robust mean plus the requested sigma threshold "
            "for the requested number of consecutive temperatures. It must not "
            "be interpreted without uncertainty, heating/cooling, and trajectory-"
            "length checks.\n"
        )


def make_plots(summary: pd.DataFrame, prefix: str) -> None:
    import matplotlib.pyplot as plt

    # Structural degree of distortion.
    plt.figure(figsize=(6.4, 4.6))
    plt.errorbar(
        summary["temperature_K"],
        summary["mean_degree_of_distortion_excess"],
        yerr=summary["sem_degree_of_distortion_excess"],
        marker="o",
        capsize=3,
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Excess degree of distortion")
    plt.tight_layout()
    plt.savefig(f"{prefix}_degree_of_distortion_vs_temperature.png", dpi=300)
    plt.close()

    # Distorted fraction.
    plt.figure(figsize=(6.4, 4.6))
    plt.errorbar(
        summary["temperature_K"],
        summary["mean_distorted_fraction"],
        yerr=summary["sem_distorted_fraction"],
        marker="o",
        capsize=3,
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Fraction of distorted In environments")
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(f"{prefix}_distorted_fraction_vs_temperature.png", dpi=300)
    plt.close()

    if "excess_thermodynamic_energy_meV_per_atom" in summary:
        plt.figure(figsize=(6.4, 4.6))
        plt.errorbar(
            summary["temperature_K"],
            summary["excess_thermodynamic_energy_meV_per_atom"],
            yerr=1000.0 * summary["sem_thermodynamic_energy_eV_per_atom"],
            marker="o",
            capsize=3,
        )
        plt.xlabel("Temperature (K)")
        plt.ylabel("Excess thermodynamic energy (meV atom$^{-1}$)")
        plt.tight_layout()
        plt.savefig(f"{prefix}_excess_energy_vs_temperature.png", dpi=300)
        plt.close()

    if "configurational_heat_capacity_kB_per_atom" in summary:
        plt.figure(figsize=(6.4, 4.6))
        plt.plot(
            summary["temperature_K"],
            summary["configurational_heat_capacity_kB_per_atom"],
            marker="o",
        )
        plt.xlabel("Temperature (K)")
        plt.ylabel("Configurational heat capacity ($k_B$ atom$^{-1}$)")
        plt.tight_layout()
        plt.savefig(f"{prefix}_configurational_heat_capacity.png", dpi=300)
        plt.close()


def main() -> None:
    args = parse_args()
    site_data = read_structure_data(args.structure_pattern)
    features = construct_dimensionless_features(site_data)

    feature_models, z_data = fit_feature_models(
        site_data,
        features,
        args.baseline_max_temperature,
        args.baseline_polynomial_degree,
    )
    z_fixed = z_data["fixed"]
    z_excess = z_data["excess"]
    fixed, n_fixed = rms_available(z_fixed)
    excess_raw, n_excess = rms_available(z_excess)

    baseline_mask = site_data["temperature_K"] <= args.baseline_max_temperature
    excess_floor = float(np.nanmedian(excess_raw[baseline_mask]))
    excess = np.maximum(excess_raw - excess_floor, 0.0)

    site_data = site_data.copy()
    for column in features.columns:
        site_data[f"component_{column}"] = features[column]
        site_data[f"z_fixed_{column}"] = z_fixed[column]
        site_data[f"z_excess_{column}"] = z_excess[column]
    site_data["degree_of_distortion_fixed"] = fixed
    site_data["degree_of_distortion_excess_raw"] = excess_raw
    site_data["degree_of_distortion_excess"] = excess
    site_data["n_valid_fixed_features"] = n_fixed
    site_data["n_valid_excess_features"] = n_excess

    distorted_threshold = float(
        np.nanpercentile(
            site_data.loc[baseline_mask, "degree_of_distortion_fixed"],
            args.distorted_percentile,
        )
    )
    site_data["distorted_environment"] = (
        site_data["degree_of_distortion_fixed"] > distorted_threshold
    ).astype(float)

    frame_data = aggregate_frames(site_data)
    n_atoms = determine_n_atoms(args)
    site_data, frame_data, energy_metadata = attach_energy(
        site_data, frame_data, args, n_atoms
    )

    summary = temperature_summary(frame_data, n_atoms, args.block_size)
    summary, energy_coefficients = add_energy_anomalies(
        summary,
        args.baseline_max_temperature,
        args.baseline_polynomial_degree,
    )

    onset_report: dict[str, object] = {}
    onset, center, threshold = detect_onset(
        summary["temperature_K"].to_numpy(),
        summary["mean_degree_of_distortion_excess"].to_numpy(),
        args.baseline_max_temperature,
        args.onset_sigma,
        args.onset_persistence,
    )
    onset_report["degree_of_distortion_onset_K"] = onset
    onset_report["degree_of_distortion_baseline_center"] = center
    onset_report["degree_of_distortion_threshold"] = threshold

    onset, center, threshold = detect_onset(
        summary["temperature_K"].to_numpy(),
        summary["mean_distorted_fraction"].to_numpy(),
        args.baseline_max_temperature,
        args.onset_sigma,
        args.onset_persistence,
    )
    onset_report["distorted_fraction_onset_K"] = onset
    onset_report["distorted_fraction_baseline_center"] = center
    onset_report["distorted_fraction_threshold"] = threshold

    if "excess_thermodynamic_energy_meV_per_atom" in summary:
        onset, center, threshold = detect_onset(
            summary["temperature_K"].to_numpy(),
            summary["excess_thermodynamic_energy_meV_per_atom"].to_numpy(),
            args.baseline_max_temperature,
            args.onset_sigma,
            args.onset_persistence,
        )
        onset_report["excess_energy_onset_K"] = onset
        onset_report["excess_energy_baseline_center_meV_per_atom"] = center
        onset_report["excess_energy_threshold_meV_per_atom"] = threshold


    peak_columns = [
        "distortion_fluctuation_NIn_varD",
        "configurational_heat_capacity_kB_per_atom",
        "derivative_heat_capacity_kB_per_atom",
        "covariance_dD_dT_per_K",
    ]
    for column in peak_columns:
        if column in summary:
            onset_report[f"candidate_{column}_peaks_K"] = candidate_peaks(
                summary["temperature_K"].to_numpy(),
                summary[column].to_numpy(),
            )

    prefix = args.output_prefix
    site_path = Path(f"{prefix}_per_In_per_frame.csv")
    frame_path = Path(f"{prefix}_per_frame.csv")
    summary_path = Path(f"{prefix}_vs_temperature.csv")
    profile_path = Path(f"{prefix}_free_energy_profiles.csv")
    metadata_path = Path(f"{prefix}_method_and_onsets.txt")

    site_data.to_csv(site_path, index=False)
    frame_data.to_csv(frame_path, index=False)
    summary.to_csv(summary_path, index=False)

    profiles = free_energy_profiles(frame_data, args.free_energy_bins)
    if not profiles.empty:
        profiles.to_csv(profile_path, index=False)

    write_metadata(
        metadata_path,
        args,
        feature_models,
        energy_metadata,
        energy_coefficients,
        distorted_threshold,
        onset_report,
        n_atoms,
    )

    if not args.no_plots:
        make_plots(summary, prefix)

    print("Analysis completed.")
    print(f"Per-In/per-frame data : {site_path}")
    print(f"Per-frame data        : {frame_path}")
    print(f"Temperature summary   : {summary_path}")
    if not profiles.empty:
        print(f"Free-energy profiles  : {profile_path}")
    print(f"Method/onset report   : {metadata_path}")
    print("Detected onsets:")
    for key, value in onset_report.items():
        if key.endswith("_K"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
