
import time
from pathlib import Path

import torch
from codecarbon import EmissionsTracker

# Create results directory
Path("results").mkdir(exist_ok=True)

tracker = EmissionsTracker(
    project_name="eaes_phase1_test",
    output_dir="results",
    output_file="energy_test.csv",
    measure_power_secs=5,
    log_level="error"
)

print("Starting energy measurement...")
tracker.start()

try:
    # Example computational workload
    start = time.perf_counter()

    a = torch.randn(400, 400)
    b = torch.randn(400, 400)

    while time.perf_counter() - start < 30:
        c = torch.mm(a, b)

finally:
    emissions = tracker.stop()

print("Energy measurement completed.")
print("Estimated emissions (kg CO2eq):", emissions)
print("Results saved in results/energy_test.csv")
