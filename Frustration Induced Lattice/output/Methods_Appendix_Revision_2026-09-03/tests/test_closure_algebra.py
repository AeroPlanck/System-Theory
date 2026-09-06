"""Independent residual test for the corrected |m|<=2 closure."""

import unittest

import numpy as np


class ClosureAlgebraTests(unittest.TestCase):
    def test_stationary_second_harmonic_residual(self):
        rng = np.random.default_rng(903)
        for _ in range(200):
            velocity, coupling = rng.uniform(0.2, 5.0, 2)
            angle = rng.uniform(-np.pi, np.pi)
            denominator = rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 10.0)
            real_product, imag_product, grad_real, grad_imag = rng.normal(size=4)
            qx = (
                velocity * grad_imag / 2
                - coupling * real_product * np.sin(angle)
                - coupling * imag_product * np.cos(angle)
            ) / denominator
            qy = (
                -velocity * grad_real / 2
                + coupling * real_product * np.cos(angle)
                - coupling * imag_product * np.sin(angle)
            ) / denominator
            forcing = -velocity / 2 * (grad_real + 1j * grad_imag)
            forcing += coupling * np.exp(1j * angle) * (real_product + 1j * imag_product)
            residual = forcing + 1j * denominator * (qx + 1j * qy)
            self.assertLess(abs(residual), 2e-13)

    def test_historical_counterexample(self):
        self.assertAlmostEqual(abs(1j * (-8) * (1 / 8) + 1j), 0.0)
        self.assertAlmostEqual(abs(1j * (-8) * (-1 / 8) + 1j), 2.0)


if __name__ == "__main__":
    unittest.main()
