import unittest
import numpy as np

from keplertools.fun import calc_RV_from_time


class TestCalcRVFromTime(unittest.TestCase):
    """
    Unit tests for the calc_RV_from_time function, comparing C and Python implementations.
    """

    def setUp(self):
        """
        Initialize common test data for all test cases.
        """
        # Example data: 100 epochs, 2 planets
        self.t = np.linspace(0, 10, 100)  # Epoch times in jd (n=100)
        self.tp = np.array(
            [5.0, 2.0]
        )  # Times of periastron passages for 2 objects (m=2)
        self.per = np.array([10.0, 5.0])  # Orbital periods for 2 objects
        self.e = np.array([0.1, 0.3])  # Eccentricities
        self.w = np.array([np.pi / 4, np.pi / 3])  # Arguments of periapsis in radians
        self.K = np.array([30.0, 40.0])  # Semi-amplitudes in m/s

    def test_basic_consistency(self):
        """
        Test basic consistency between C and Python implementations with typical parameters.
        """
        # Calculate RV with C implementation
        rv_c = calc_RV_from_time(
            self.t, self.tp, self.per, self.e, self.w, self.K, use_c=True
        )
        # Calculate RV with Python implementation
        rv_py = calc_RV_from_time(
            self.t, self.tp, self.per, self.e, self.w, self.K, use_c=False
        )
        # Compare using np.allclose with a tight tolerance
        self.assertTrue(
            np.allclose(rv_c, rv_py, atol=1e-6),
            f"C and Python implementations differ beyond tolerance.\nMax difference: {np.max(np.abs(rv_c - rv_py))} m/s",
        )

    def test_single_planet_circular_orbit(self):
        """
        Test with a single planet in a circular orbit.
        """
        # Define single planet data
        t = np.linspace(0, 20, 200)  # 200 epochs
        tp = np.array([10.0])  # Periastron time
        per = np.array([10.0])  # Period
        e = np.array([0.0])  # Circular orbit
        w = np.array([0.0])  # Argument of periapsis
        K = np.array([50.0])  # Semi-amplitude

        # Calculate RV with C implementation
        rv_c = calc_RV_from_time(t, tp, per, e, w, K, use_c=True)
        # Calculate RV with Python implementation
        rv_py = calc_RV_from_time(t, tp, per, e, w, K, use_c=False)
        # Compare
        self.assertTrue(
            np.allclose(rv_c, rv_py, atol=1e-6),
            f"C and Python implementations differ for circular orbit.\nMax difference: {np.max(np.abs(rv_c - rv_py))} m/s",
        )

    def test_high_eccentricity(self):
        """
        Test with high eccentricity to evaluate numerical stability.
        """
        # Define high eccentricity data
        t = np.linspace(0, 30, 300)  # 300 epochs
        tp = np.array([15.0])  # Periastron time
        per = np.array([15.0])  # Period
        e = np.array([0.99])  # High eccentricity
        w = np.array([np.pi / 2])  # Argument of periapsis
        K = np.array([100.0])  # Semi-amplitude

        # Calculate RV with C implementation
        rv_c = calc_RV_from_time(t, tp, per, e, w, K, use_c=True)
        # Calculate RV with Python implementation
        rv_py = calc_RV_from_time(t, tp, per, e, w, K, use_c=False)
        # Compare using a slightly higher tolerance due to numerical challenges at high eccentricity
        self.assertTrue(
            np.allclose(rv_c, rv_py, atol=1e-5),
            f"C and Python implementations differ for high eccentricity.\nMax difference: {np.max(np.abs(rv_c - rv_py))} m/s",
        )

    def test_multiple_planets(self):
        """
        Test with multiple planets to ensure scalability.
        """
        # Define multiple planets data
        t = np.linspace(0, 50, 500)  # 500 epochs
        tp = np.array([10.0, 20.0, 30.0])  # Periastron times
        per = np.array([10.0, 15.0, 20.0])  # Periods
        e = np.array([0.05, 0.15, 0.25])  # Eccentricities
        w = np.array([np.pi / 6, np.pi / 3, np.pi / 2])  # Arguments of periapsis
        K = np.array([10.0, 20.0, 30.0])  # Semi-amplitudes

        # Calculate RV with C implementation
        rv_c = calc_RV_from_time(t, tp, per, e, w, K, use_c=True)
        # Calculate RV with Python implementation
        rv_py = calc_RV_from_time(t, tp, per, e, w, K, use_c=False)
        # Compare
        self.assertTrue(
            np.allclose(rv_c, rv_py, atol=1e-6),
            f"C and Python implementations differ for multiple planets.\nMax difference: {np.max(np.abs(rv_c - rv_py))} m/s",
        )

    def test_zero_eccentricity(self):
        """
        Test with zero eccentricity (circular orbit) to ensure correct handling.
        """
        # Define zero eccentricity data
        t = np.linspace(0, 10, 100)  # 100 epochs
        tp = np.array([5.0])  # Periastron time
        per = np.array([10.0])  # Period
        e = np.array([0.0])  # Circular orbit
        w = np.array([0.0])  # Argument of periapsis
        K = np.array([50.0])  # Semi-amplitude

        # Calculate RV with C implementation
        rv_c = calc_RV_from_time(t, tp, per, e, w, K, use_c=True)
        # Calculate RV with Python implementation
        rv_py = calc_RV_from_time(t, tp, per, e, w, K, use_c=False)
        # Compare
        self.assertTrue(
            np.allclose(rv_c, rv_py, atol=1e-6),
            f"C and Python implementations differ for zero eccentricity.\nMax difference: {np.max(np.abs(rv_c - rv_py))} m/s",
        )

    def test_no_planets(self):
        """
        Test with no planets to ensure the function handles empty arrays gracefully.
        """
        # Define no planets data
        t = np.linspace(0, 10, 100)  # 100 epochs
        tp = np.array([])  # No periastron times
        per = np.array([])  # No periods
        e = np.array([])  # No eccentricities
        w = np.array([])  # No arguments of periapsis
        K = np.array([])  # No semi-amplitudes

        # Calculate RV with C implementation
        rv_c = calc_RV_from_time(t, tp, per, e, w, K, use_c=True)
        # Calculate RV with Python implementation
        rv_py = calc_RV_from_time(t, tp, per, e, w, K, use_c=False)
        # Expected output is an array of zeros
        expected = np.zeros_like(t, dtype=np.float64)
        # Compare C implementation
        self.assertTrue(
            np.allclose(rv_c, expected, atol=1e-6),
            f"C implementation did not return zero velocities for no planets.\nMax difference: {np.max(np.abs(rv_c - expected))} m/s",
        )
        # Compare Python implementation
        self.assertTrue(
            np.allclose(rv_py, expected, atol=1e-6),
            f"Python implementation did not return zero velocities for no planets.\nMax difference: {np.max(np.abs(rv_py - expected))} m/s",
        )
        # Compare C and Python implementations
        self.assertTrue(
            np.allclose(rv_c, rv_py, atol=1e-6),
            f"C and Python implementations differ for no planets.\nMax difference: {np.max(np.abs(rv_c - rv_py))} m/s",
        )

    def test_invalid_input_sizes(self):
        """
        Test with mismatched input array sizes to ensure proper error handling.
        """
        # Define mismatched data
        t = np.linspace(0, 10, 100)  # 100 epochs
        tp = np.array([5.0, 2.0, 3.0])  # 3 periastron times
        per = np.array([10.0, 5.0])  # 2 periods
        e = np.array([0.1, 0.3, 0.2])  # 3 eccentricities
        w = np.array([np.pi / 4, np.pi / 3, np.pi / 6])  # 3 arguments of periapsis
        K = np.array([30.0, 40.0, 25.0])  # 3 semi-amplitudes

        # Expect the function to raise a ValueError due to mismatched array sizes
        with self.assertRaises(ValueError):
            calc_RV_from_time(t, tp, per, e, w, K, use_c=True)

        with self.assertRaises(ValueError):
            calc_RV_from_time(t, tp, per, e, w, K, use_c=False)

    def test_output_shape(self):
        """
        Test that the output radial velocity array has the correct shape.
        """
        rv_c = calc_RV_from_time(
            self.t, self.tp, self.per, self.e, self.w, self.K, use_c=True
        )
        rv_py = calc_RV_from_time(
            self.t, self.tp, self.per, self.e, self.w, self.K, use_c=False
        )
        # Check shapes
        self.assertEqual(
            rv_c.shape,
            self.t.shape,
            f"C implementation output shape {rv_c.shape} does not match input epochs shape {self.t.shape}.",
        )
        self.assertEqual(
            rv_py.shape,
            self.t.shape,
            f"Python implementation output shape {rv_py.shape} does not match input epochs shape {self.t.shape}.",
        )


if __name__ == "__main__":
    unittest.main()
