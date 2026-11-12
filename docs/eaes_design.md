
# Energy-Aware Early Stopping (EAES) Design

## Goal
Reduce unnecessary neural-network training by stopping
when validation accuracy improvement becomes too small
relative to energy consumption.

## Formula

S(t) = Accuracy Improvement / Energy Used

Energy is measured in watt-hours (Wh).

Accuracy improvement is measured in percentage points.

### Accuracy Smoothing

EAES uses a three-epoch moving average to reduce fluctuations in validation accuracy.

Current smoothed accuracy:

A_avg(t) = [A(t) + A(t-1) + A(t-2)] / 3

Previous smoothed accuracy:

A_avg(t-1) = [A(t-1) + A(t-2) + A(t-3)] / 3

Accuracy improvement:

Improvement(t) = max(0, A_avg(t) - A_avg(t-1))

Energy-efficiency score:

S(t) = Improvement(t) / Energy_Wh(t)

Validation accuracy is expressed in percentage points, and energy consumption is expressed in watt-hours.

Four epochs of validation accuracy are required to calculate the first smoothed improvement. However, EAES will not make a stopping decision before epoch 5.

Validation accuracies are converted from fractions to percentage points before
smoothing. For example, an accuracy of `0.85` is represented as `85.0`.

At epoch `t`, the runner calculates:

- Current average: mean of validation accuracies at epochs `t-2`, `t-1`, and `t`.
- Previous average: mean of validation accuracies at epochs `t-3`, `t-2`, and `t-1`.
- Improvement: `max(0, current average - previous average)`, in percentage points.

The improvement can be recorded at epoch 4, once four accuracy observations
exist, but score evaluation and stopping decisions begin only at epoch 5.

If energy is zero, missing, non-finite, or invalid, the score is skipped and the
consecutive low-score counter resets. CodeCarbon measurement exceptions or the
absence of finalized measurement data stop the run visibly; the runner does not
substitute a guessed value. A zero reading is retained as zero in the epoch CSV,
but is not used as a divisor.

## Stopping Rules

1. Train for at least 5 epochs.
2. Smooth validation accuracy using the latest 3 epochs.
3. Calculate accuracy improvement after every epoch.
4. Negative improvement is treated as zero.
5. Divide accuracy improvement by epoch energy usage.
6. Compare the score with the selected threshold.
7. If the score is below the threshold for 3
   consecutive eligible epochs, stop training.
8. Otherwise, continue until the maximum of 30 epochs.

## Pseudocode

Initialize low_score_count = 0
Initialize accuracy_history = []

For epoch = 1 to 30:

    Train CNN for one epoch

    Evaluate validation accuracy and loss

    Measure energy consumed during the epoch

    Save accuracy, loss, time, and energy

    Add validation accuracy to accuracy_history

    If fewer than 4 accuracy values are available:
        Continue training

    current_smoothed_accuracy =
        average of latest 3 validation accuracies

    previous_smoothed_accuracy =
        average of previous 3 validation accuracies

    improvement =
        max(0, current_smoothed_accuracy
               - previous_smoothed_accuracy)

    If epoch < 5:
        Continue training

    If energy consumption is too small to measure:
        Record invalid score and reset low_score_count
        Continue training

    score = improvement / energy_in_Wh

    If score < threshold:
        low_score_count += 1
    Else:
        low_score_count = 0

    If low_score_count >= 3:
        Stop training

Save the selected checkpoint and experiment results. Evaluate test accuracy
using the final epoch for fixed training and the lowest-validation-loss
checkpoint for standard early stopping and EAES; ties keep the earlier epoch.

## Thresholds

Three EAES thresholds are intended to be evaluated:

- Low threshold: TBD
- Medium threshold: TBD
- High threshold: TBD

The values are not selected yet. Supply the selected value with
`--eaes-threshold`; the CLI requires it for EAES runs rather than silently using
an arbitrary default. It is measured in validation accuracy percentage points
per Wh.

## Comparison Methods

1. Fixed training for 30 epochs
2. Standard early stopping using validation loss
3. Energy-Aware Early Stopping

## Success Criteria

EAES should reduce training energy consumption
without decreasing test accuracy by more than
one percentage point.

## Implementation Notes

- Use the same CNN and dataset split for all methods.
- Use the same optimizer and learning rate.
- Use consistent hardware across experiments.
- Repeat each experimental condition with 3 seeds.
- Convert CodeCarbon kWh readings to Wh.
- Record emissions, accuracy, energy, and runtime.
- Measure each epoch from CodeCarbon start through one complete training pass,
  validation pass, and CodeCarbon stop. Use the same boundary for all methods.
- Include CodeCarbon initialization/stop in the recorded epoch duration. Test-set
  evaluation and checkpoint writing are outside the epoch energy boundary.
- Save each run in a new results subdirectory to preserve existing results.
- Use the final epoch checkpoint for fixed training. For standard and EAES, save
  the checkpoint with the lowest validation loss (ties keep the earlier epoch)
  and evaluate that checkpoint on the test set.
- The per-run CSV records each epoch and the final test accuracy; `summary.json`
  records totals. Total energy is reported only when every epoch has a positive
  energy reading, preventing a partial sum from appearing complete.
