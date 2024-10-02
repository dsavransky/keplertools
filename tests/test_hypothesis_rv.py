import unittest
import numpy as np
from keplertools.fun import calc_RV_from_time
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from typing import Tuple

# ==========================
# Strategy Definitions
# ==========================

# Strategy for epoch times: sorted, unique, positive floats
t_strategy = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=1, max_value=1000),  # Number of epochs
    elements=st.floats(
        min_value=0.0, max_value=1e6,
        allow_nan=False, allow_infinity=False
    ),
).map(np.sort)

# Strategy for number of planets
n_planets_strategy = st.integers(min_value=0, max_value=20)

# Strategy for planetary parameters based on number of planets
def planet_parameters_strategy(n_planets):
    if n_planets == 0:
        return st.just((np.array([]), np.array([]), np.array([]), np.array([]), np.array([])))
    else:
        # Set valid ranges for each orbital parameter
        param_ranges = {
            'tp': [0.0, 1e6],
            'per': [1e-2, 1e6],
            # NOTE: The methods diverge too much above 0.99 to be worth testing
            'e': [0.0, 0.99],
            'w': [0.0, 2 * np.pi],
            'K': [0.0, 1000],
        }

        # Generate array strategies based on the parameter ranges
        strategies = {
            param: arrays(
                dtype=np.float64,
                shape=n_planets,
                elements=st.floats(
                    min_value=limits[0],
                    max_value=limits[1],
                    allow_nan=False,
                    allow_infinity=False
                )
            )
            for param, limits in param_ranges.items()
        }

        return st.tuples(strategies['tp'], strategies['per'], strategies['e'], strategies['w'], strategies['K'])


# Composite strategy combining epoch times and planetary parameters
@st.composite
def rv_input_strategy(draw):
    t = draw(t_strategy)
    n_planets = draw(n_planets_strategy)
    tp, per, e, w, K = draw(planet_parameters_strategy(n_planets))
    return t, tp, per, e, w, K

# ==========================
# Test Class
# ==========================

class TestCalcRVFromTimeHypothesis(unittest.TestCase):
    """
    Hypothesis-based unit tests for the calc_RV_from_time function, comparing C
    and Python implementations across a wide range of inputs.
    """

    @settings(deadline=None)
    @given(rv_input=rv_input_strategy())
    def test_consistency_with_hypothesis(self, rv_input: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """
        Test consistency between C and Python implementations.

        Args:
            rv_input (Tuple): A tuple containing:
                - t (np.ndarray): Epoch times.
                - tp (np.ndarray): Times of periastron passages.
                - per (np.ndarray): Orbital periods.
                - e (np.ndarray): Eccentricities.
                - w (np.ndarray): Arguments of periapsis.
                - K (np.ndarray): Semi-amplitudes.
        """
        t, tp, per, e, w, K = rv_input

        if tp.size == 0:
            # No planets case
            with self.assertRaises(ValueError):
                calc_RV_from_time(t, tp, per, e, w, K, noc=False)
            return

        # Calculate RV with C implementation
        rv_c = calc_RV_from_time(t, tp, per, e, w, K, noc=False)

        # Calculate RV with Python implementation
        rv_py = calc_RV_from_time(t, tp, per, e, w, K, noc=True)

        # Adjust atol based on maximum K
        atol = 1e-6 * K.max()

        # Adjust relative tolerance based on eccentricity
        if np.any(e > 0.9):
            rtol = 1e-3
        else:
            rtol = 1e-4

        # Assert that both implementations are close within the specified tolerances
        self.assertTrue(
            np.allclose(rv_c, rv_py, rtol=rtol, atol=atol),
            ("C and Python implementations differ beyond tolerance.\n"
            f"Max difference: {np.max(np.abs(rv_c - rv_py))} m/s")
        )

if __name__ == "__main__":
    unittest.main()
