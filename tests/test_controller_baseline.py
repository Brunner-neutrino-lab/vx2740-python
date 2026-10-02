import unittest

import numpy as np

from vx2740.controller import VX2740Controller
from vx2740.driver import ADC_MIDSCALE, SAMPLE_RATE_HZ


class PulseBaselineWindowTests(unittest.TestCase):
    def setUp(self):
        self.controller = VX2740Controller(mode="simulation")
        self.controller._thresholds = {0: 1_000}

    def test_long_pretrigger_uses_only_first_two_microseconds(self):
        self.controller.configure_record_window(pre_us=4.0, post_us=1.0)
        pre = self.controller._pre_samples
        post = self.controller._post_samples
        two_us_samples = int(round(2e-6 * SAMPLE_RATE_HZ))

        waveform = np.full(
            (1, pre + post),
            ADC_MIDSCALE,
            dtype=np.uint16,
        )
        # A large late-pre-trigger excursion must not bias the baseline used
        # by online pulse finding.
        waveform[0, two_us_samples:pre] += 1_000
        waveform[0, pre:] += 1_100

        amplitudes, timestamps = self.controller._find_pulses_batch(
            waveform,
            np.array([0.0]),
            channel=0,
        )

        self.assertEqual(amplitudes, [1_100.0])
        self.assertEqual(len(timestamps), 1)

    def test_short_pretrigger_uses_entire_available_region(self):
        self.controller.configure_record_window(pre_us=1.0, post_us=1.0)
        pre = self.controller._pre_samples
        post = self.controller._post_samples

        waveform = np.full(
            (1, pre + post),
            ADC_MIDSCALE + 200,
            dtype=np.uint16,
        )
        waveform[0, pre] = ADC_MIDSCALE + 1_100
        self.controller._thresholds = {0: 900}

        amplitudes, timestamps = self.controller._find_pulses_batch(
            waveform,
            np.array([0.0]),
            channel=0,
        )

        self.assertEqual(amplitudes, [900.0])
        self.assertEqual(len(timestamps), 1)
