import argparse
import csv
import json
import math
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
from codecarbon import EmissionsTracker
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from dataset import get_data
from model import FashionCNN

MAX_EPOCHS = 30
LEARNING_RATE = 0.001
SEED = 42
PATIENCE = 3
MIN_EAES_EPOCHS = 5
EAES_SMOOTHING_WINDOW = 3


class ValidationLossEarlyStopping:
    """Stop after patience consecutive epochs without lower validation loss."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = math.inf
        self.bad_epochs = 0

    def update(self, validation_loss: float) -> tuple[bool, bool]:
        improved = validation_loss < self.best_loss
        if improved:
            self.best_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


class EnergyAwareEarlyStopping:
    """Apply the documented accuracy-per-Wh rule to each completed epoch."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.accuracy_history_pp: list[float] = []
        self.low_score_epochs = 0

    def update(
        self, epoch: int, validation_accuracy_pp: float, energy_wh: float | None
    ) -> dict[str, float | int | bool | None]:
        self.accuracy_history_pp.append(validation_accuracy_pp)
        improvement_pp = None
        score = None
        should_stop = False

        if len(self.accuracy_history_pp) >= EAES_SMOOTHING_WINDOW + 1:
            history = self.accuracy_history_pp
            current_average = sum(history[-EAES_SMOOTHING_WINDOW:]) / EAES_SMOOTHING_WINDOW
            previous_average = sum(
                history[-EAES_SMOOTHING_WINDOW - 1:-1]
            ) / EAES_SMOOTHING_WINDOW
            improvement_pp = max(0.0, current_average - previous_average)

        if epoch >= MIN_EAES_EPOCHS:
            if energy_wh is None or not math.isfinite(energy_wh) or energy_wh <= 0:
                self.low_score_epochs = 0
            else:
                score = improvement_pp / energy_wh if improvement_pp is not None else None
                if score is not None and score < self.threshold:
                    self.low_score_epochs += 1
                else:
                    self.low_score_epochs = 0
                should_stop = self.low_score_epochs >= PATIENCE

        return {
            "improvement_pp": improvement_pp,
            "score": score,
            "low_score_epochs": self.low_score_epochs,
            "should_stop": should_stop,
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_tracker(method: str, epoch: int) -> EmissionsTracker:
    return EmissionsTracker(
        project_name=f"eaes_{method}_epoch_{epoch}",
        measure_power_secs=1,
        save_to_file=False,
        save_to_api=False,
        save_to_logger=False,
        allow_multiple_runs=False,
        log_level="error",
    )


def _read_tracker_measurements(
    tracker: EmissionsTracker, returned_emissions_kg: float | None
) -> tuple[float | None, float | None]:
    data = getattr(tracker, "final_emissions_data", None)
    if data is None:
        raise RuntimeError(
            "CodeCarbon did not produce finalized measurement data. "
            "The run is stopping rather than saving unverified energy values."
        )

    energy_kwh = getattr(data, "energy_consumed", None)
    energy_wh = None
    if isinstance(energy_kwh, (int, float)) and math.isfinite(energy_kwh) and energy_kwh >= 0:
        energy_wh = float(energy_kwh) * 1000
    else:
        warnings.warn(
            "CodeCarbon returned a missing or invalid energy value for this epoch; "
            "energy will be blank in the results and EAES will skip its score.",
            RuntimeWarning,
        )

    emissions_kg = returned_emissions_kg
    if not isinstance(emissions_kg, (int, float)) or not math.isfinite(emissions_kg) or emissions_kg < 0:
        warnings.warn(
            "CodeCarbon returned missing or invalid CO2-equivalent emissions; "
            "emissions will be blank in the results.",
            RuntimeWarning,
        )
        emissions_kg = None
    else:
        emissions_kg = float(emissions_kg)

    if energy_wh == 0:
        warnings.warn(
            "CodeCarbon reported zero Wh for this epoch. The value is retained, "
            "but EAES will not divide by zero and will reset its low-score counter.",
            RuntimeWarning,
        )
    return energy_wh, emissions_kg


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_loss += loss.item() * count
        total_correct += (predictions.argmax(1) == labels).sum().item()
        total_examples += count

    if total_examples == 0:
        raise RuntimeError("The training data loader produced no examples.")
    return total_loss / total_examples, total_correct / total_examples


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            count = labels.size(0)
            total_loss += loss.item() * count
            total_correct += (predictions.argmax(1) == labels).sum().item()
            total_examples += count

    if total_examples == 0:
        raise RuntimeError("The evaluation data loader produced no examples.")
    return total_loss / total_examples, total_correct / total_examples


def _new_run_directory(results_root: Path, method: str, seed: int, smoke_test: bool) -> Path:
    prefix = "smoke_" if smoke_test else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = results_root / f"{prefix}{method}_seed{seed}_{timestamp}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def run_experiment(
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    eaes_threshold: float | None = None,
    seed: int = SEED,
    results_root: Path = Path("results"),
    smoke_test: bool = False,
) -> Path:
    if method not in {"fixed", "standard", "eaes"}:
        raise ValueError(f"Unknown experiment method: {method}")
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    if max_epochs > MAX_EPOCHS:
        raise ValueError(f"max_epochs cannot exceed {MAX_EPOCHS}.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if method == "eaes" and eaes_threshold is None:
        raise ValueError("An EAES threshold is required for method='eaes'.")
    if eaes_threshold is not None and (
        not math.isfinite(eaes_threshold) or eaes_threshold < 0
    ):
        raise ValueError("eaes_threshold must be a finite, non-negative number.")

    _set_seed(seed)
    device = torch.device("cpu")
    model = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    run_dir = _new_run_directory(results_root, method, seed, smoke_test)
    checkpoint_path = run_dir / "model.pth"
    history: list[dict[str, object]] = []
    early_stopper = ValidationLossEarlyStopping(patience)
    eaes_stopper = (
        EnergyAwareEarlyStopping(eaes_threshold)
        if method == "eaes" and eaes_threshold is not None
        else None
    )
    best_val_loss = math.inf
    best_epoch = 0
    stop_reason = "maximum_epochs_reached"
    for epoch in range(1, max_epochs + 1):
        tracker = _create_tracker(method, epoch)
        epoch_start = time.perf_counter()
        tracker.start()
        try:
            train_loss, train_accuracy = _train_epoch(
                model, train_loader, optimizer, loss_fn, device
            )
            val_loss, val_accuracy = _evaluate(model, val_loader, loss_fn, device)
        finally:
            returned_emissions_kg = tracker.stop()
        epoch_seconds = time.perf_counter() - epoch_start
        energy_wh, emissions_kg = _read_tracker_measurements(
            tracker, returned_emissions_kg
        )

        row: dict[str, object] = {
            "method": method,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": epoch_seconds,
            "energy_wh": energy_wh,
            "co2_emissions_kg": emissions_kg,
            "energy_status": (
                "unavailable" if energy_wh is None
                else "zero_or_too_small" if energy_wh == 0
                else "measured"
            ),
            "eaes_improvement_pp": None,
            "eaes_score_per_wh": None,
            "eaes_low_score_epochs": None,
        }

        if method == "fixed":
            best_epoch = epoch
        elif method == "standard":
            improved, should_stop = early_stopper.update(val_loss)
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            if should_stop:
                stop_reason = f"validation_loss_patience_{patience}"
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            if eaes_stopper is None:
                raise RuntimeError("EAES stopping logic was not initialized.")
            decision = eaes_stopper.update(epoch, val_accuracy * 100, energy_wh)
            row["eaes_improvement_pp"] = decision["improvement_pp"]
            row["eaes_score_per_wh"] = decision["score"]
            row["eaes_low_score_epochs"] = decision["low_score_epochs"]
            if decision["should_stop"]:
                stop_reason = "eaes_three_consecutive_low_scores"

        history.append(row)
        print(
            f"{method} epoch {epoch}/{max_epochs}: "
            f"train_acc={train_accuracy:.2%}, val_acc={val_accuracy:.2%}, "
            f"val_loss={val_loss:.4f}, energy_wh={energy_wh}, "
            f"co2_kg={emissions_kg}"
        )

        if stop_reason != "maximum_epochs_reached":
            break

    total_epochs = len(history)
    total_training_seconds = sum(float(row["epoch_seconds"]) for row in history)
    if total_epochs == 0:
        raise RuntimeError("Training completed without completing an epoch.")
    if method != "fixed" and not checkpoint_path.exists():
        raise RuntimeError("Training completed without producing a model checkpoint.")

    if method == "fixed":
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_selection = "final epoch"
        best_epoch = total_epochs
    else:
        checkpoint_selection = "lowest validation loss; ties keep the earlier epoch"
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    _, test_accuracy = _evaluate(model, test_loader, loss_fn, device)
    known_energies = [row["energy_wh"] for row in history]
    known_emissions = [row["co2_emissions_kg"] for row in history]
    energy_complete = all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in known_energies
    )
    emissions_complete = all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in known_emissions
    )
    total_energy_wh = (
        sum(float(value) for value in known_energies) if energy_complete else None
    )
    total_emissions_kg = (
        sum(float(value) for value in known_emissions) if emissions_complete else None
    )

    for row in history:
        row["total_epochs_completed"] = total_epochs
        row["test_accuracy"] = test_accuracy
        row["total_energy_wh"] = total_energy_wh
        row["total_training_time_seconds"] = total_training_seconds
        row["total_co2_emissions_kg"] = total_emissions_kg
        row["checkpoint_selection"] = checkpoint_selection
        row["stop_reason"] = stop_reason

    csv_path = run_dir / "history.csv"
    fieldnames = [
        "method",
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_accuracy",
        "val_loss",
        "epoch_seconds",
        "energy_wh",
        "co2_emissions_kg",
        "energy_status",
        "eaes_improvement_pp",
        "eaes_score_per_wh",
        "eaes_low_score_epochs",
        "total_epochs_completed",
        "test_accuracy",
        "total_energy_wh",
        "total_training_time_seconds",
        "total_co2_emissions_kg",
        "checkpoint_selection",
        "stop_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "method": method,
        "seed": seed,
        "eaes_threshold_accuracy_points_per_wh": eaes_threshold,
        "max_epochs": max_epochs,
        "patience": patience if method == "standard" else None,
        "total_epochs_completed": total_epochs,
        "stop_reason": stop_reason,
        "checkpoint_epoch": best_epoch,
        "checkpoint_selection": checkpoint_selection,
        "test_accuracy": test_accuracy,
        "total_training_time_seconds": total_training_seconds,
        "total_energy_wh": total_energy_wh,
        "energy_measurement_complete": energy_complete,
        "total_co2_emissions_kg": total_emissions_kg,
        "emissions_measurement_complete": emissions_complete,
        "smoke_test_synthetic_data": smoke_test,
        "checkpoint_file": checkpoint_path.name,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Completed {method}: {total_epochs} epochs, test accuracy={test_accuracy:.2%}")
    print(f"Results: {run_dir}")
    return run_dir


def _make_smoke_test_loaders(seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(8, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (8,), generator=generator)
    dataset = TensorDataset(images, labels)
    return tuple(
        DataLoader(dataset, batch_size=4, shuffle=False) for _ in range(3)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FashionCNN using fixed, standard, or EAES stopping."
    )
    parser.add_argument("--method", choices=("fixed", "standard", "eaes"), default="fixed")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--eaes-threshold",
        type=float,
        help="Required for EAES: accuracy percentage points gained per Wh.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny synthetic dataset and run only one epoch; not research data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.epochs > MAX_EPOCHS:
        raise SystemExit(f"--epochs cannot exceed {MAX_EPOCHS}.")
    if args.patience < 1:
        raise SystemExit("--patience must be at least 1.")
    if args.method == "eaes" and args.eaes_threshold is None:
        raise SystemExit("--eaes-threshold is required when --method eaes is selected.")
    if args.eaes_threshold is not None and (
        not math.isfinite(args.eaes_threshold) or args.eaes_threshold < 0
    ):
        raise SystemExit("--eaes-threshold must be finite and non-negative.")

    if args.smoke_test:
        train_loader, val_loader, test_loader = _make_smoke_test_loaders(args.seed)
        epochs = 1
        print("Smoke test: using synthetic data for one epoch; results are not research data.")
    else:
        train_loader, val_loader, test_loader = get_data(seed=args.seed)
        epochs = args.epochs

    run_experiment(
        method=args.method,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        max_epochs=epochs,
        patience=args.patience,
        eaes_threshold=args.eaes_threshold,
        seed=args.seed,
        results_root=args.results_dir,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
