import unittest

from train import EnergyAwareEarlyStopping, ValidationLossEarlyStopping


class ValidationLossEarlyStoppingTests(unittest.TestCase):
    def test_stops_after_three_consecutive_non_improvements(self):
        stopper = ValidationLossEarlyStopping(patience=3)

        self.assertEqual(stopper.update(0.5), (True, False))
        self.assertEqual(stopper.update(0.4), (True, False))
        self.assertEqual(stopper.update(0.4), (False, False))
        self.assertEqual(stopper.update(0.41), (False, False))
        self.assertEqual(stopper.update(0.42), (False, True))


class EnergyAwareEarlyStoppingTests(unittest.TestCase):
    def test_uses_three_epoch_moving_average_improvement(self):
        stopper = EnergyAwareEarlyStopping(threshold=0.1)
        accuracies = [80.0, 82.0, 84.0, 86.0]
        decisions = [
            stopper.update(epoch, accuracy, 1.0)
            for epoch, accuracy in enumerate(accuracies, start=1)
        ]

        self.assertIsNone(decisions[2]["improvement_pp"])
        self.assertEqual(decisions[3]["improvement_pp"], 2.0)
        self.assertFalse(decisions[3]["should_stop"])

    def test_minimum_epoch_and_three_consecutive_low_scores(self):
        stopper = EnergyAwareEarlyStopping(threshold=0.1)
        decisions = [
            stopper.update(epoch, 80.0, 1.0)
            for epoch in range(1, 8)
        ]

        self.assertEqual(decisions[3]["low_score_epochs"], 0)
        self.assertEqual(decisions[4]["low_score_epochs"], 1)
        self.assertEqual(decisions[5]["low_score_epochs"], 2)
        self.assertTrue(decisions[6]["should_stop"])

    def test_invalid_energy_resets_the_low_score_counter(self):
        stopper = EnergyAwareEarlyStopping(threshold=0.1)
        decisions = []
        for epoch in range(1, 10):
            energy = None if epoch == 6 else 1.0
            decisions.append(stopper.update(epoch, 80.0, energy))

        self.assertEqual(decisions[4]["low_score_epochs"], 1)
        self.assertEqual(decisions[5]["low_score_epochs"], 0)
        self.assertEqual(decisions[8]["low_score_epochs"], 3)
        self.assertTrue(decisions[8]["should_stop"])


if __name__ == "__main__":
    unittest.main()
