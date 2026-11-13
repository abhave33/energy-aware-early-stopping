# Phase 1 Research Notes

## 1. Research Problem

Neural networks can consume unnecessary energy when training continues even though validation accuracy is improving very little.

Traditional neural-network training often uses a fixed number of epochs. However, additional epochs do not always produce meaningful improvements in model performance.

This project investigates whether stopping training based on both accuracy improvement and energy consumption can reduce unnecessary energy use.

## 2. Research Background and Related Work

### 2.1 Strubell et al. (2019) — Energy Costs of Deep Learning

**Paper:** Energy and Policy Considerations for Deep Learning in NLP

Strubell et al. studied the computational, financial, and environmental costs associated with training deep-learning models.

Their research highlights that machine-learning performance should not be evaluated using accuracy alone. The computational resources and energy required to train models are also important considerations.

**Relevance to Our Project:**

This paper supports the motivation for reducing unnecessary training. If a neural network continues training without meaningful accuracy improvements, additional computational resources may be wasted.

Our EAES approach attempts to reduce this waste by considering energy consumption when deciding whether training should stop.

### 2.2 Henderson et al. (2020) — Measuring Energy and Carbon Emissions

**Paper:** Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning

Henderson et al. examined the importance of measuring and reporting energy consumption and carbon emissions associated with machine-learning experiments.

Their research emphasizes that environmental measurements can help researchers better understand the resource costs of model development.

**Relevance to Our Project:**

This paper supports our decision to measure energy consumption and estimated carbon emissions alongside training accuracy and runtime.

Our project uses CodeCarbon to estimate the energy consumed during CNN training.

By recording these measurements, we can compare the environmental and computational costs of different early-stopping methods.

### 2.3 Meng and Yao (2023) — Early Stopping in Deep Learning

**Paper:** Impact of Classification Difficulty on the Weight Matrices Spectra in Deep Learning and Application to Early-Stopping

Meng and Yao investigated relationships between classification difficulty, neural-network weight matrices, and early stopping.

Their research provides background on using training behavior to determine when additional training may no longer be beneficial.

**Relevance to Our Project:**

This paper helps establish the importance of early stopping as a method for reducing unnecessary training.

Our project extends this general idea by investigating whether energy consumption can also help determine an appropriate stopping point.

Rather than relying only on validation performance, EAES considers how much additional accuracy is gained relative to the energy consumed.

## 3. Research Gap

Existing early-stopping approaches commonly focus on validation loss, validation accuracy, or other measures of model performance.

However, these approaches do not necessarily account directly for the energy consumed during each additional training epoch.

A model might continue training even when additional epochs consume energy but produce very little accuracy improvement.

This project investigates whether combining validation accuracy improvement with energy consumption can provide a more energy-efficient stopping decision.

## 4. Research Question

Can Energy-Aware Early Stopping reduce neural-network training energy consumption without lowering test accuracy by more than one percentage point?

## 5. Proposed Approach

The project proposes an Energy-Aware Early Stopping (EAES) algorithm that evaluates whether continued training produces enough accuracy improvement relative to energy consumption.

EAES calculates an energy-efficiency score after each eligible epoch:

**S(t) = Accuracy Improvement / Energy Consumed**

Where:

- Accuracy improvement is measured in percentage points.
- Energy consumption is measured in watt-hours (Wh).
- The score represents accuracy improvement per watt-hour.

The algorithm uses a three-epoch moving average of validation accuracy to reduce fluctuations.

Training must complete at least five epochs before stopping is permitted.

If the energy-efficiency score remains below a selected threshold for three consecutive eligible epochs, training stops.

The complete algorithm design and pseudocode are documented in `docs/eaes_design.md`.

## 6. Dataset and CNN

The project uses the Fashion-MNIST dataset, containing 70,000 grayscale clothing images across 10 categories.

Our initial dataset configuration is:

- Training images: 50,000
- Validation images: 10,000
- Test images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10

The proposed CNN implementation uses:

- Two convolutional layers
- ReLU activation functions
- Max-pooling layers
- Fully connected classification layers
- Adam optimizer
- Learning rate of 0.001
- Batch size of 64

The same CNN architecture and training settings will be used across all comparison methods.

## 7. Planned Experimental Comparison

Three training methods will be compared.

### Method 1: Fixed Training

The CNN trains for a fixed maximum of 30 epochs, regardless of whether validation performance has stopped improving.

This serves as the baseline for comparing training time and energy consumption.

### Method 2: Standard Early Stopping

The CNN stops training when validation loss no longer improves according to the selected patience setting.

This provides a conventional early-stopping baseline.

### Method 3: Energy-Aware Early Stopping

The CNN uses validation accuracy improvement and estimated energy consumption to calculate an energy-efficiency score.

Training stops when the score remains below the selected threshold for three consecutive eligible epochs.

Low, medium, and high EAES thresholds will be evaluated.

## 8. Experimental Measurements

The following values will be recorded and compared:

- Training accuracy
- Validation accuracy
- Validation loss
- Test accuracy
- Number of completed epochs
- Training duration
- Estimated energy consumption (kWh and Wh)
- Estimated CO₂-equivalent emissions

Each experimental condition was repeated three times using random seeds 42, 43,
and 44.

To maintain fairness, the same dataset split, CNN architecture, optimizer, learning rate, epoch limit, and computer will be used across comparable conditions.

The primary success criterion is reduced energy consumption without a test-accuracy decrease greater than one percentage point.

## 9. Phase 1 — CodeCarbon Testing

### Purpose

CodeCarbon was installed and tested to determine whether it could successfully estimate energy consumption and carbon emissions on the development computer.

A short computational workload was executed using Python and PyTorch while CodeCarbon recorded system energy measurements.

### Testing Environment

**Date:** October 26, 2026

- Hardware: Apple M4 Mac
- RAM: 16 GB
- Operating system: macOS
- Python version: 3.13.1
- CodeCarbon version: 3.3.2

### Test Results

| Measurement | Recorded Value |
|---|---|
| Duration | 33.36 seconds |
| Estimated energy consumption | 0.00012836 kWh |
| Energy consumption in Wh | 0.12836 Wh |
| Estimated CO₂-equivalent emissions | 0.00004177 kg |
| Estimated CPU power | 11.85 W |

### Interpretation

The CodeCarbon test successfully generated a CSV file containing estimated energy consumption, emissions, and system measurements.

The test confirms that CodeCarbon can run on the development computer and produce energy estimates.

These results are preliminary and were generated using a short computational workload. They do not represent energy consumption during CNN training or demonstrate energy savings from EAES.

### Phase 2 Integration

CodeCarbon is integrated into the shared CNN training runner for fixed, standard,
and EAES modes. Each epoch uses the same boundary: tracker start, full training
pass, validation pass, and tracker stop. Energy values in kWh are converted to Wh
for storage and EAES scoring. CodeCarbon measurements remain estimates, not exact
physical readings.

The runner records energy and emissions per epoch, total training duration, and
final test accuracy in separate run folders. Zero or unavailable energy readings
are not used in EAES division; such epochs reset the consecutive-low-score count.
Missing finalized tracker data stops the run instead of being reported as valid
energy. The initial comparison experiments for thresholds 1, 3, and 5 have been
completed across three seeds; results and limitations are summarized below.

## Phase 3 — Repeated Experiment Results

The five conditions were run sequentially on CPU using the same Fashion-MNIST
split (split seed 42), CNN, optimizer, learning rate, and maximum of 30 epochs.
Training seeds were 42, 43, and 44. There are 15 valid runs: one run per method
and seed. All three seeds are present for every method in
`results/analysis/method_comparison.csv`.

| Method | Test accuracy (%) | Epochs | Training time (s) | Estimated energy (Wh) | Energy saved vs. fixed | Energy saved vs. standard |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 91.66 ± 0.18 | 30.00 ± 0.00 | 767.96 ± 38.54 | 4.6385 ± 0.4495 | 0.00% | -190.50% |
| Standard | 91.52 ± 0.29 | 10.67 ± 1.53 | 270.63 ± 36.73 | 1.5967 ± 0.2361 | 65.58% | 0.00% |
| EAES low (1) | 91.52 ± 0.29 | 12.67 ± 0.58 | 312.84 ± 16.45 | 1.9074 ± 0.0895 | 58.88% | -19.46% |
| EAES medium (3) | 91.66 ± 0.31 | 8.67 ± 1.15 | 215.83 ± 28.00 | 1.3284 ± 0.1545 | 71.36% | 16.81% |
| EAES high (5) | 91.66 ± 0.31 | 8.00 ± 0.00 | 197.97 ± 6.18 | 1.2090 ± 0.0405 | 73.94% | 24.28% |

Entries are mean ± sample standard deviation over the three seeds. Energy savings
are calculated from method mean energy:
`(fixed mean energy - method mean energy) / fixed mean energy * 100`; savings
relative to standard use the same formula with standard mean energy as the
baseline. A negative value means the method used more energy than that baseline.
Every method's test accuracy was within one percentage point of fixed training
for each matched seed. EAES high had the lowest mean energy; EAES medium and high
used less energy than standard on all three seeds, while EAES low used more
energy than standard on average.

### Anomalous Run Excluded from Aggregates

The first low-threshold seed-42 attempt,
`results/eaes_seed42_20261112_161203`, is retained unchanged for auditability.
One epoch took approximately 24,135 seconds and recorded approximately 67.35 Wh,
more than five times the cross-run median epoch duration and energy. The
analysis marks it `outlier_review` and excludes it from method aggregates.
The separate valid seed-42 low-threshold rerun is included instead. No other run
met the analyzer's outlier rule, and the 15 included runs had positive,
finite per-epoch energy and duration readings. CodeCarbon estimates are not
exact physical measurements.

The analysis outputs are `results/analysis/run_summary.csv`,
`results/analysis/method_comparison.csv`, and four PNG comparisons. The
run-level file retains the anomalous record and its exclusion reason; the
method-level file contains three valid runs and three unique seeds for each
condition. No synthetic smoke-test runs are present in the experiment folders
or included in the comparison.

## Limitations

- Only one dataset (Fashion-MNIST) and one CNN architecture were evaluated.
- CodeCarbon reports estimated energy and CO₂e; this study did not use a physical
  electricity meter, and background activity can affect the estimates.
- Each condition was run with only three random seeds, limiting the strength of
  conclusions about variability and generalization.
- The conclusions may not transfer to different datasets, model architectures,
  hardware, or operating conditions.

## 10. Limitations

### Energy Measurement Accuracy

CodeCarbon estimates energy consumption rather than measuring it directly using a physical electricity meter.

Energy estimates may vary depending on the hardware, background activity, and measurement conditions.

### Small Dataset and Model

Fashion-MNIST is relatively small, and the CNN may train quickly.

Energy differences between training methods could therefore be small and difficult to measure accurately.

### Hardware Differences

Energy consumption varies between computers.

Comparative experiments should use the same machine and consistent conditions to reduce variability.

### Threshold Selection

The selected EAES threshold influences when training stops.

A higher threshold may stop training earlier and reduce energy consumption but could decrease accuracy.

A lower threshold may allow more training and preserve accuracy but reduce potential energy savings.

Multiple threshold values will therefore be tested.

## 11. Phase 1 Conclusion

Phase 1 established the research motivation and experimental approach for studying energy-aware neural-network training.

The Fashion-MNIST dataset and CNN training pipeline were prepared separately as part of the baseline development work.

CodeCarbon was successfully installed and tested, producing estimated energy and carbon-emissions measurements.

The EAES formula, smoothing method, stopping conditions, and experimental comparison plan have been documented.

The next phase will focus on integrating energy tracking into CNN training, implementing EAES, and comparing the three training methods.

## 12. References

[1] E. Strubell, A. Ganesh, and A. McCallum, "Energy and policy considerations for deep learning in NLP," *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 3645–3650, 2019. DOI: 10.18653/v1/P19-1355.

[2] P. Henderson, J. Hu, J. Romoff, E. Brunskill, D. Jurafsky, and J. Pineau, "Towards the systematic reporting of the energy and carbon footprints of machine learning," *Journal of Machine Learning Research*, vol. 21, no. 248, pp. 1–43, 2020.

[3] X. Meng and J. Yao, "Impact of classification difficulty on the weight matrices spectra in deep learning and application to early-stopping," *Journal of Machine Learning Research*, vol. 24, no. 28, pp. 1–40, 2023.