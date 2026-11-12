# Energy-Aware Early Stopping for Neural Network Training

## Project Overview

This project investigates whether Energy-Aware Early Stopping (EAES) can reduce unnecessary neural-network training energy consumption without significantly affecting model accuracy.

We are comparing three training methods:

1. Fixed training for 30 epochs
2. Standard early stopping based on validation loss
3. Energy-Aware Early Stopping (EAES)

The project uses a Convolutional Neural Network (CNN) trained on Fashion-MNIST. CodeCarbon is used to estimate energy consumption and CO₂-equivalent emissions.

**Progress Presentation:** November 9, 2026

---

## Team Responsibilities

### Project Partner — CNN and Baseline Training

- Prepare the Fashion-MNIST dataset
- Implement the CNN architecture
- Build the training and validation pipeline
- Implement fixed 30-epoch training
- Implement standard early stopping
- Analyze accuracy, validation loss, and training time

### Aakansha — Energy Tracking and EAES

- Research energy-efficient neural-network training
- Investigate existing early-stopping methods
- Test CodeCarbon energy measurements
- Design and implement the EAES algorithm
- Evaluate EAES thresholds
- Analyze energy consumption and emissions

---

## Phase 1 — Foundation

**Status: Completed.**

### CNN and Dataset Setup

- [x] Set up Python environment and dependencies
- [x] Download and prepare Fashion-MNIST
- [x] Implement CNN architecture (`model.py`)
- [x] Implement training and validation pipeline (`train.py`)
- [x] Run initial 3-epoch CNN training test
- [x] Record training/validation accuracy and loss
- [x] Record epoch training duration
- [x] Generate accuracy and loss graphs

### Research and EAES Preparation

- [x] Document research problem and related work
- [x] Prepare research notes covering Strubell et al., Henderson et al., and Meng and Yao
- [x] Install and test CodeCarbon
- [x] Record preliminary energy and emissions measurements
- [x] Document the EAES formula and measurement units
- [x] Create EAES algorithm pseudocode
- [x] Define minimum epochs, smoothing, and stopping conditions
- [x] Review requirements for integrating EAES with the CNN training pipeline

### Phase 1 Outcome

The CNN training pipeline is working, and initial accuracy and loss measurements have been collected.

CodeCarbon has also been successfully tested on a separate computational workload.

The research background and EAES algorithm design are documented in the `docs/` directory.

The original three-epoch run remains saved as `results/history.csv`. The experiment
runner writes new results into unique subdirectories and does not overwrite that
Phase 1 file.

## Phase 2 — Experiment Runner

The shared training runner supports `fixed`, `standard`, and `eaes` modes. Each
epoch is timed and measured with CodeCarbon over the same boundary: tracker start,
full training pass, validation pass, and tracker stop. Training accuracy,
validation accuracy/loss, duration, energy, and estimated CO2-equivalent emissions
are saved to a run-specific `history.csv`, together with repeated run totals and
test accuracy. The same final totals and checkpoint details are summarized in
`summary.json`.

CodeCarbon estimates are not exact physical measurements. Missing or invalid
energy is marked in the CSV; EAES skips scoring for that epoch and resets its
consecutive-low-score count. If CodeCarbon does not produce finalized measurement
data, the run fails visibly rather than reporting unverified energy. Aggregate
energy is blank unless every epoch has a positive measurement.

Checkpoint selection:

- `fixed`: test the final epoch's weights.
- `standard`: test the checkpoint with the lowest validation loss; ties keep the
  earlier checkpoint.
- `eaes`: test the checkpoint with the lowest validation loss; ties keep the
  earlier checkpoint.

Standard early stopping uses validation-loss patience 3 by default. EAES requires
an explicit threshold in accuracy percentage points per Wh because threshold
values remain to be selected by preliminary experiments.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST |
| Training images | 50,000 |
| Validation images | 10,000 |
| Test images | 10,000 |
| Image dimensions | 28 × 28 grayscale |
| Number of classes | 10 |
| Batch size | 64 |
| CNN architecture | 2 convolutional layers with pooling and a fully connected classifier |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Dataset split seed | 42 |
| Initial Phase 1 training test | 3 epochs |
| Maximum Phase 2 epochs | 30 |
| Standard early-stopping patience | 3 epochs |
| Energy measurement | CodeCarbon |

The experimental settings will remain consistent across comparable training methods.

## Project Structure

```text
energy-aware-early-stopping/
├── .gitignore
├── README.md
├── requirements.txt
├── dataset.py
├── model.py
├── train.py
├── plot_results.py
├── energy_test.py
├── tests/
│   └── test_train.py
├── docs/
│   ├── research_notes.md
│   └── eaes_design.md
├── data/
└── results/
    ├── history.csv                 # preserved Phase 1, three-epoch results
    ├── accuracy.png
    ├── loss.png
    └── <method>_seed<seed>_<time>/
        ├── history.csv
        ├── summary.json
        └── model.pth
```

The downloaded dataset, local virtual environment, raw energy-test data, and trained model weights may be excluded from Git through `.gitignore`.

---

## How to Run the Project

### 1. Install Dependencies

Create and activate a Python virtual environment before installing dependencies.

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare Fashion-MNIST

```bash
python dataset.py
```

### 3. Verify CNN Architecture

```bash
python model.py
```

### 4. Run the Three Experiment Methods

Each command runs up to 30 epochs and creates a new results directory, leaving
existing results untouched:

```bash
python train.py --method fixed
python train.py --method standard --patience 3
python train.py --method eaes --eaes-threshold YOUR_THRESHOLD
```

Replace `YOUR_THRESHOLD` with the threshold selected for the experiment. No EAES
threshold is selected in the project yet; the value is measured in validation
accuracy percentage points per Wh. Use the same seed (default `42`) and hardware
for comparable methods. Change the seed explicitly for repeated runs, for example
`--seed 43`.

To run a one-epoch integration smoke test without downloading Fashion-MNIST or
producing research data:

```bash
python train.py --method fixed --smoke-test
python train.py --method standard --smoke-test
python train.py --method eaes --eaes-threshold 0.1 --smoke-test
```

Smoke-test outputs are labeled `smoke_...` and use synthetic data. They validate
the runner and result-writing path only; do not use them as experimental results.

Run the stopping-rule unit tests with:

```bash
python -m unittest discover -s tests -v
```

### 5. Generate the Original Phase 1 Training Graphs

```bash
python plot_results.py
```

### 6. Test CodeCarbon

```bash
python energy_test.py
```

`plot_results.py` continues to read the preserved Phase 1 `results/history.csv`.
Phase 2 run-specific CSVs can be plotted by selecting a run's CSV in a separate
analysis step.

---

## Phase 1 — CodeCarbon Test Results

**Test Date:** October 26, 2026

**Testing Environment:**

- Computer: Apple M4 Mac
- RAM: 16 GB
- Python: 3.13.1
- CodeCarbon: 3.3.2

### Measurements

| Measurement | Result |
|---|---|
| Test duration | 33.36 seconds |
| Estimated energy consumption | 0.00012836 kWh |
| Estimated energy in Wh | 0.12836 Wh |
| Estimated CO₂-equivalent emissions | 0.00004177 kg |
| Estimated CPU power | 11.85 W |

### Conclusion

CodeCarbon successfully recorded estimated energy consumption and emissions during a short computational workload.

This confirms that energy tracking is functioning on the development computer.

These values are preliminary measurements and do not represent CNN training energy consumption or demonstrate EAES energy savings.

---

## EAES Algorithm Design

The proposed EAES algorithm evaluates validation accuracy improvement relative to energy consumption.

### Formula

S(t) = ΔA(t) / E(t)

Where:

- ΔA(t) = Smoothed validation accuracy improvement, in percentage points
- E(t) = Energy consumed during epoch t, in Wh
- S(t) = Accuracy improvement per Wh

### Accuracy Smoothing

The algorithm uses a three-epoch moving average.

Current smoothed accuracy:

A_avg(t) = [A(t) + A(t-1) + A(t-2)] / 3

Previous smoothed accuracy:

A_avg(t-1) = [A(t-1) + A(t-2) + A(t-3)] / 3

Accuracy improvement:

ΔA(t) = max(0, A_avg(t) - A_avg(t-1))

### Stopping Rules

1. Train for at least 5 epochs.
2. Calculate smoothed validation accuracy.
3. Treat negative accuracy improvements as zero.
4. Divide accuracy improvement by energy consumed in Wh.
5. Compare the score against a selected threshold.
6. Stop after 3 consecutive eligible epochs below the threshold.
7. Otherwise, continue until the maximum of 30 epochs.
8. Handle zero or invalid energy measurements without dividing by zero.

Low, medium, and high threshold values will be selected during preliminary experiments.

Full EAES pseudocode and design details are available in `docs/eaes_design.md`.

---

## Research Methodology

Three training strategies will be compared:

1. Fixed training
2. Standard early stopping
3. Energy-Aware Early Stopping

Each experimental condition will be repeated three times with different random seeds.

Results will be compared using:

- Test accuracy
- Validation accuracy and loss
- Completed epochs
- Training time
- Energy consumption
- Estimated CO₂-equivalent emissions

The primary success criterion is reducing energy consumption without reducing test accuracy by more than one percentage point.

Research background and citations are documented in `docs/research_notes.md`.

---

## Phase 2 — Implementation Status

### Project Partner — Baseline Methods

- [x] Implement fixed training up to 30 epochs
- [x] Implement standard early stopping based on validation loss
- [x] Save accuracy, runtime, and loss measurements

### EAES Development

- [x] Integrate CodeCarbon into the CNN training loop
- [x] Record estimated energy consumption per epoch
- [x] Implement EAES stopping logic
- [x] Test stopping conditions
- [ ] Select initial low, medium, and high thresholds

### Joint Tasks

- [x] Integrate the baseline and EAES implementations
- [x] Verify consistent experimental settings
- [ ] Begin repeated experiments
- [ ] Generate comparison graphs
- [ ] Analyze preliminary results

---

## Project Timeline

| Phase | Tasks | Status |
|---|---|---|
| Phase 1 | Dataset, CNN, research, EAES design, energy-tracking test | Completed |
| Phase 2 | Implement and smoke-test all three training methods | Implementation complete; full runs pending |
| Phase 3 | Repeated experiments and comparison of training methods | Planned |
| Phase 4 | Results analysis and progress presentation | Planned |
| Final Stage | Further experiments, refinements, and final project report | Planned |

**Progress Presentation:** November 9, 2026

**Target:** Approximately 75% project completion with working stopping methods and preliminary experimental results.