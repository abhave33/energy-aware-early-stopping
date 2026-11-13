import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import FashionCNN

MAX_EPOCHS = 30
PATIENCE = 3
MIN_EAES_EPOCHS = 5
SMOOTHING_WINDOW = 3
EXPECTED_SEEDS = (42, 43, 44)
OUTLIER_MULTIPLIER = 5.0
METHOD_ORDER = ("Fixed", "Standard", "EAES low (1)", "EAES medium (3)", "EAES high (5)")


def _read_run(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "model.pth"
    record: dict[str, object] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": "invalid",
        "exclusion_reason": "",
        "method": "",
        "method_label": "",
        "seed": None,
        "threshold": None,
        "test_accuracy_pct": None,
        "epochs_completed": None,
        "training_time_seconds": None,
        "energy_wh": None,
        "co2_emissions_kg": None,
        "checkpoint_epoch": None,
        "stop_reason": "",
        "min_epoch_seconds": None,
        "median_epoch_seconds": None,
        "max_epoch_seconds": None,
        "min_epoch_energy_wh": None,
        "median_epoch_energy_wh": None,
        "max_epoch_energy_wh": None,
    }

    if not summary_path.is_file() or not history_path.is_file() or not checkpoint_path.is_file():
        record["exclusion_reason"] = "Missing summary, history CSV, or checkpoint."
        return record

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with history_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError("History CSV has no epoch rows.")

        method = summary["method"]
        seed = int(summary["seed"])
        epochs = int(summary["total_epochs_completed"])
        if method not in {"fixed", "standard", "eaes"}:
            raise ValueError(f"Unknown method {method!r}.")
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"Seed {seed} is outside the configured comparison seeds.")
        if len(rows) != epochs:
            raise ValueError("CSV row count does not match summary epoch count.")
        if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
            raise ValueError("Epoch rows are missing, duplicated, or out of order.")
        if any(row["method"] != method for row in rows):
            raise ValueError("CSV method column does not match summary.")
        if any(int(row["total_epochs_completed"]) != epochs for row in rows):
            raise ValueError("CSV completed-epoch count does not match summary.")

        energy = [float(row["energy_wh"]) for row in rows]
        emissions = [float(row["co2_emissions_kg"]) for row in rows]
        durations = [float(row["epoch_seconds"]) for row in rows]
        val_accuracies = [float(row["val_accuracy"]) * 100 for row in rows]
        val_losses = [float(row["val_loss"]) for row in rows]
        if not all(math.isfinite(value) and value > 0 for value in energy):
            raise ValueError("Energy contains missing, non-finite, zero, or negative values.")
        if not all(math.isfinite(value) and value >= 0 for value in emissions):
            raise ValueError("Emissions contain missing, non-finite, or negative values.")
        if not all(math.isfinite(value) and value > 0 for value in durations):
            raise ValueError("Epoch duration contains missing or invalid values.")
        if not all(math.isfinite(value) for value in val_accuracies + val_losses):
            raise ValueError("Validation metrics contain non-finite values.")
        if any(not 0 <= float(row["test_accuracy"]) <= 1 for row in rows):
            raise ValueError("CSV test accuracy is outside [0, 1].")

        test_accuracies = {float(row["test_accuracy"]) for row in rows}
        if len(test_accuracies) != 1:
            raise ValueError("Test accuracy is inconsistent across the CSV rows.")
        test_accuracy = test_accuracies.pop()
        if not math.isclose(sum(energy), float(summary["total_energy_wh"]), rel_tol=1e-8):
            raise ValueError("Epoch energy sum does not match summary.")
        if not math.isclose(
            sum(emissions), float(summary["total_co2_emissions_kg"]), rel_tol=1e-8
        ):
            raise ValueError("Epoch emissions sum does not match summary.")
        if not math.isclose(
            sum(durations), float(summary["total_training_time_seconds"]), rel_tol=1e-8
        ):
            raise ValueError("Epoch duration sum does not match summary.")
        if not math.isclose(test_accuracy, float(summary["test_accuracy"]), rel_tol=1e-8):
            raise ValueError("Test accuracy does not match summary.")

        if method == "fixed":
            if epochs != MAX_EPOCHS or summary["stop_reason"] != "maximum_epochs_reached":
                raise ValueError("Fixed run did not complete the 30-epoch maximum.")
            if int(summary["checkpoint_epoch"]) != epochs:
                raise ValueError("Fixed checkpoint is not from the final epoch.")
            method_label = "Fixed"
            threshold = None
        elif method == "standard":
            if int(summary["patience"]) != PATIENCE:
                raise ValueError("Standard run does not use patience 3.")
            if epochs > MAX_EPOCHS:
                raise ValueError("Standard run exceeds the 30-epoch maximum.")
            if summary["stop_reason"] == f"validation_loss_patience_{PATIENCE}":
                best_loss_epoch = min(range(epochs), key=lambda i: val_losses[i]) + 1
                if epochs != best_loss_epoch + PATIENCE:
                    raise ValueError("Standard stopping does not match patience 3.")
            elif summary["stop_reason"] != "maximum_epochs_reached" or epochs != MAX_EPOCHS:
                raise ValueError("Standard stopping reason is inconsistent.")
            method_label = "Standard"
            threshold = None
        else:
            threshold = float(summary["eaes_threshold_accuracy_points_per_wh"])
            if threshold not in {1.0, 3.0, 5.0}:
                raise ValueError(f"EAES threshold {threshold} is not one of 1, 3, or 5.")
            if epochs > MAX_EPOCHS:
                raise ValueError("EAES run exceeds the 30-epoch maximum.")
            low_count = 0
            expected_stop = None
            for index, row in enumerate(rows):
                epoch = index + 1
                score = None
                if index >= SMOOTHING_WINDOW:
                    current_average = sum(val_accuracies[index - 2:index + 1]) / SMOOTHING_WINDOW
                    previous_average = sum(val_accuracies[index - 3:index]) / SMOOTHING_WINDOW
                    improvement = max(0.0, current_average - previous_average)
                    score = improvement / energy[index]
                    saved_score = row["eaes_score_per_wh"]
                    if epoch >= MIN_EAES_EPOCHS and (
                        not saved_score
                        or not math.isclose(float(saved_score), score, rel_tol=1e-7, abs_tol=1e-7)
                    ):
                        raise ValueError(f"EAES saved score does not match at epoch {epoch}.")
                if epoch >= MIN_EAES_EPOCHS:
                    low_count = low_count + 1 if score is not None and score < threshold else 0
                    if int(row["eaes_low_score_epochs"]) != low_count:
                        raise ValueError(f"EAES low-score counter does not match at epoch {epoch}.")
                    if low_count >= PATIENCE:
                        expected_stop = epoch
                        break
            if expected_stop is not None:
                if epochs != expected_stop or summary["stop_reason"] != "eaes_three_consecutive_low_scores":
                    raise ValueError("EAES stop epoch does not match the three-low-score rule.")
            elif epochs != MAX_EPOCHS or summary["stop_reason"] != "maximum_epochs_reached":
                raise ValueError("EAES run stopped without meeting a documented stop rule.")
            threshold_label = {1.0: "low", 3.0: "medium", 5.0: "high"}.get(threshold, str(threshold))
            method_label = f"EAES {threshold_label} ({threshold:g})"

        checkpoint_epoch = int(summary["checkpoint_epoch"])
        if method != "fixed":
            expected_best_epoch = min(range(epochs), key=lambda i: val_losses[i]) + 1
            if checkpoint_epoch != expected_best_epoch:
                raise ValueError("Checkpoint epoch does not match lowest validation loss.")
        model = FashionCNN()
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )

        record.update({
            "status": "valid",
            "method": method,
            "method_label": method_label,
            "seed": seed,
            "threshold": threshold,
            "test_accuracy_pct": test_accuracy * 100,
            "epochs_completed": epochs,
            "training_time_seconds": float(summary["total_training_time_seconds"]),
            "energy_wh": float(summary["total_energy_wh"]),
            "co2_emissions_kg": float(summary["total_co2_emissions_kg"]),
            "checkpoint_epoch": checkpoint_epoch,
            "stop_reason": summary["stop_reason"],
            "min_epoch_seconds": min(durations),
            "median_epoch_seconds": statistics.median(durations),
            "max_epoch_seconds": max(durations),
            "min_epoch_energy_wh": min(energy),
            "median_epoch_energy_wh": statistics.median(energy),
            "max_epoch_energy_wh": max(energy),
        })
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        record["exclusion_reason"] = str(error)
    return record


def _detect_outliers(records: list[dict[str, object]]) -> None:
    valid_records = [record for record in records if record["status"] == "valid"]
    durations = [
        float(record["median_epoch_seconds"])
        for record in valid_records
        if record["median_epoch_seconds"] is not None
    ]
    energies = [
        float(record["median_epoch_energy_wh"])
        for record in valid_records
        if record["median_epoch_energy_wh"] is not None
    ]
    duration_median = statistics.median(durations) if durations else None
    energy_median = statistics.median(energies) if energies else None

    for record in valid_records:
        flags = []
        if duration_median and float(record["max_epoch_seconds"]) > OUTLIER_MULTIPLIER * duration_median:
            flags.append(
                f"max epoch duration exceeds {OUTLIER_MULTIPLIER:g}x the cross-run median epoch duration"
            )
        if energy_median and float(record["max_epoch_energy_wh"]) > OUTLIER_MULTIPLIER * energy_median:
            flags.append(
                f"max epoch energy exceeds {OUTLIER_MULTIPLIER:g}x the cross-run median epoch energy"
            )
        if flags:
            record["status"] = "outlier_review"
            record["exclusion_reason"] = "; ".join(flags)


def _summarize(
    records: list[dict[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    run_keys: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in records:
        if record["status"] == "valid":
            key = (str(record["method_label"]), int(record["seed"]))
            run_keys.setdefault(key, []).append(record)
    for (method_label, seed), duplicate_runs in run_keys.items():
        if len(duplicate_runs) > 1:
            for record in duplicate_runs:
                record["status"] = "outlier_review"
                record["exclusion_reason"] = (
                    f"Multiple completed {method_label} runs exist for seed {seed}; "
                    "resolve the duplicate before aggregation."
                )

    eligible = [
        record for record in records
        if record["status"] == "valid"
        and not record["run_id"].startswith("smoke_")
        and int(record["seed"]) in EXPECTED_SEEDS
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in eligible:
        grouped.setdefault(str(record["method_label"]), []).append(record)

    fixed_energy = statistics.mean(
        float(record["energy_wh"]) for record in grouped.get("Fixed", [])
    ) if grouped.get("Fixed") else None
    standard_energy = statistics.mean(
        float(record["energy_wh"]) for record in grouped.get("Standard", [])
    ) if grouped.get("Standard") else None

    metrics = (
        "test_accuracy_pct",
        "epochs_completed",
        "training_time_seconds",
        "energy_wh",
        "co2_emissions_kg",
    )
    comparison: list[dict[str, object]] = []
    for method_label in METHOD_ORDER:
        runs = grouped.get(method_label, [])
        row: dict[str, object] = {
            "method": method_label,
            "run_count": len(runs),
            "seed_count": len({record["seed"] for record in runs}),
            "missing_seeds": ",".join(
                str(seed) for seed in EXPECTED_SEEDS
                if seed not in {record["seed"] for record in runs}
            ),
            "comparison_status": (
                "complete"
                if {record["seed"] for record in runs} == set(EXPECTED_SEEDS)
                else "incomplete"
            ),
            "energy_savings_vs_fixed_pct": None,
            "energy_savings_vs_standard_pct": None,
        }
        for metric in metrics:
            values = [float(record[metric]) for record in runs if record[metric] is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        mean_energy = row["energy_wh_mean"]
        if mean_energy is not None and fixed_energy:
            row["energy_savings_vs_fixed_pct"] = (
                fixed_energy - float(mean_energy)
            ) / fixed_energy * 100
        if mean_energy is not None and standard_energy:
            row["energy_savings_vs_standard_pct"] = (
                standard_energy - float(mean_energy)
            ) / standard_energy * 100
        comparison.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(records[0].keys()) if records else []
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    comparison_fields = list(comparison[0].keys()) if comparison else []
    with (output_dir / "method_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


def _plot_comparisons(records: list[dict[str, object]], comparison: list[dict[str, object]], output_dir: Path) -> None:
    plot_methods = [row for row in comparison if row["run_count"]]
    labels = [str(row["method"]) for row in plot_methods]
    x = list(range(len(labels)))

    for metric, std_key, title, ylabel, filename, show_error in (
        ("energy_wh", "energy_wh_std", "Mean Estimated Energy by Method", "Energy (Wh)", "mean_energy_by_method.png", True),
        ("test_accuracy_pct", "test_accuracy_pct_std", "Mean Test Accuracy by Method", "Test accuracy (%)", "mean_test_accuracy_by_method.png", True),
        ("training_time_seconds", "training_time_seconds_std", "Mean Training Time by Method", "Training time (seconds)", "mean_training_time_by_method.png", False),
    ):
        means = [row[f"{metric}_mean"] for row in plot_methods]
        errors = [row[std_key] or 0.0 for row in plot_methods] if show_error else None
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, means, yerr=errors, capsize=5 if show_error else 0)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    seeds = list(EXPECTED_SEEDS)
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(seeds)
    for seed_index, seed in enumerate(seeds):
        seed_values = []
        for method in labels:
            candidates = [
                record for record in records
                if record["status"] == "valid"
                and record["method_label"] == method
                and record["seed"] == seed
            ]
            seed_values.append(
                float(candidates[0]["epochs_completed"]) if len(candidates) == 1 else math.nan
            )
        offsets = [index - 0.4 + width / 2 + seed_index * width for index in x]
        ax.bar(offsets, seed_values, width=width, label=f"Seed {seed}")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Completed Epochs by Method and Seed")
    ax.set_ylabel("Epochs completed")
    ax.set_ylim(bottom=0, top=MAX_EPOCHS + 2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "epochs_completed_by_method.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and compare EAES experiment runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    run_dirs = sorted(
        path for path in args.results_dir.iterdir()
        if path.is_dir() and path != args.output_dir and not path.name.startswith("analysis")
    )
    records = [_read_run(run_dir) for run_dir in run_dirs]
    _detect_outliers(records)
    comparison = _summarize(records, args.output_dir)
    _plot_comparisons(records, comparison, args.output_dir)

    print(f"Analyzed {len(records)} runs. Results written to {args.output_dir}.")
    print(f"Valid runs: {sum(record['status'] == 'valid' for record in records)}")
    print(f"Runs requiring review: {sum(record['status'] != 'valid' for record in records)}")
    for row in comparison:
        print(
            f"{row['method']}: {row['run_count']} valid run(s), "
            f"mean energy={row['energy_wh_mean']}, "
            f"mean test accuracy={row['test_accuracy_pct_mean']}"
        )


if __name__ == "__main__":
    main()
