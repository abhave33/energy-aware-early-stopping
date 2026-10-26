## Phase 1 — Progress and Team Handoff

### Anjali — CNN and Dataset Setup

**Completed Work:**

- [x] Set up Python environment and installed dependencies
- [x] Downloaded and prepared Fashion-MNIST dataset (`dataset.py`)
- [x] Implemented CNN architecture (`model.py`)
- [x] Created training and validation pipeline (`train.py`)
- [x] Successfully ran initial 3-epoch training test
- [x] Recorded training/validation accuracy, loss, and epoch duration
- [x] Generated accuracy and loss graphs (`plot_results.py`)
- [x] Uploaded code and results to GitHub

### Aakansha — Phase 1 Tasks

**To Complete:**

- [ ] Review research papers on energy-efficient neural-network training and early stopping
- [ ] Understand how CodeCarbon measures energy consumption and emissions
- [ ] Test CodeCarbon with a basic Python script
- [ ] Design the EAES algorithm and write pseudocode
- [ ] Document the EAES stopping conditions
- [ ] Review the CNN architecture and training pipeline uploaded to GitHub

### Information Aakansha Needs

**Dataset and Model:**

- Fashion-MNIST (50,000 training / 10,000 validation / 10,000 testing images)
- CNN with 2 convolutional layers and fully connected classifier
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Random seed: 42

**Training Pipeline:**

- `dataset.py` — Dataset loading and splitting
- `model.py` — CNN architecture
- `train.py` — Training and validation
- `plot_results.py` — Performance graphs
- `results/history.csv` — Epoch-by-epoch results

**EAES Design Requirements:**

- Calculate validation accuracy improvement relative to energy consumed per epoch
- Minimum of 5 training epochs before stopping
- Use 3-epoch accuracy smoothing
- Treat negative accuracy improvements as zero
- Stop after 3 consecutive epochs below the energy-efficiency threshold
- Report the threshold in accuracy improvement per watt-hour

### Phase 1 Completion Goal

**Next Phase:** Integrate CodeCarbon, implement fixed 30-epoch training and standard early stopping, and begin implementing EAES.
