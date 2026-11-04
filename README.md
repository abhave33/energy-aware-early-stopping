# Energy-Aware Early Stopping for Neural Network Training

## Project Overview

This project investigates whether Energy-Aware Early Stopping (EAES) can reduce unnecessary neural-network training energy consumption without significantly affecting model accuracy.

We are comparing three training methods:

1. Fixed training for 30 epochs
2. Standard early stopping based on validation loss
3. Energy-Aware Early Stopping (EAES)

The project uses a Convolutional Neural Network (CNN) trained on Fashion-MNIST. CodeCarbon is used to estimate energy consumption and CO₂-equivalent emissions.

**Progress Presentation:** November 9, 2026  
**Presentation Goal:** Approximately 75% of the overall project completed.

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

**Status: Core Phase 1 deliverables completed.**

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

The project is ready to begin Phase 2 implementation.

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
| Initial training test | 3 epochs |
| Fixed-training baseline | 30 epochs |
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
├── docs/
│   ├── research_notes.md
│   └── eaes_design.md
├── data/
└── results/
    ├── history.csv
    ├── accuracy.png
    ├── loss.png
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

### 4. Run Initial CNN Training

```bash
python train.py
```

### 5. Generate Training Graphs

```bash
python plot_results.py
```

### 6. Test CodeCarbon

```bash
python energy_test.py
```

The initial CNN training script runs three epochs. Full 30-epoch experiments will be implemented during Phase 2.

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

## Phase 2 — Implementation

### Project Partner — Baseline Methods

- [ ] Implement fixed 30-epoch training
- [ ] Implement standard early stopping based on validation loss
- [ ] Save baseline accuracy, runtime, and loss measurements
- [ ] Prepare baseline results for comparison

### EAES Development

- [ ] Integrate CodeCarbon into the CNN training loop
- [ ] Record estimated energy consumption per epoch
- [ ] Implement EAES stopping logic
- [ ] Test stopping conditions
- [ ] Select initial low, medium, and high thresholds

### Joint Tasks

- [ ] Integrate the baseline and EAES implementations
- [ ] Verify consistent experimental settings
- [ ] Begin repeated experiments
- [ ] Generate comparison graphs
- [ ] Analyze preliminary results

---

## Project Timeline

| Phase | Tasks | Status |
|---|---|---|
| Phase 1 | Dataset, CNN, research, EAES design, energy-tracking test | Completed |
| Phase 2 | Fixed training, standard early stopping, CodeCarbon integration, EAES implementation | Next |
| Phase 3 | Repeated experiments and comparison of training methods | Planned |
| Phase 4 | Results analysis and progress presentation | Planned |
| Final Stage | Further experiments, refinements, and final project report | Planned |

**Progress Presentation:** November 9, 2026

**Target:** Approximately 75% project completion with working stopping methods and preliminary experimental results.