import unittest
from keplertools.fun import (
    eccanom,
    eccanom_orvara,
    trueanom,
    vec2orbElem,
    vec2orbElem2,
    calcAB,
    orbElem2vec,
    invKepler,
    kepler2orbstate,
    orbstate2kepler,
    universalfg,
)
import numpy as np


class TestKeplerTools(unittest.TestCase):
    """
    Test method: verify invertibility of all transformations using random inputs
    """

    def setUp(self):
        self.n = int(1e3)  # Number of test cases to run
        # general tolerances
        self.etol = np.sqrt(np.spacing(1))
        self.angtol = np.sqrt(np.spacing(2 * np.pi))

    def tearDown(self):
        pass

    def test_eccanom_vecM_scalare(self):
        """Test eccentric anomaly calculation for a scalar eccentricity value

        Use a normal distribution with sigma 2\pi to ensure that some M values are
        outside of (0,2\pi) range
        """

        M = np.random.randn(self.n) * 2 * np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M, e)
        calcM = E - e * np.sin(E)
        self.assertTrue(
            np.max(np.abs(calcM - np.mod(M, 2 * np.pi))) < np.finfo(float).eps * 10.0
        )

        # Test eccanom_orvara against eccanom
        E_orvara, *_ = eccanom_orvara(M, e)
        self.assertTrue(np.max(np.abs(E - E_orvara)) < np.finfo(float).eps * 100.0)

    def test_eccanom_vecM_scalare_noc(self):
        """Test eccentric anomaly calc for a scalar eccentricity value (Python value)

        Use a normal distribution with sigma 2\pi to ensure that some M values are
        outside of (0,2\pi) range
        """

        M = np.random.randn(self.n) * 2 * np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M, e, noc=True)
        calcM = E - e * np.sin(E)
        self.assertTrue(
            np.max(np.abs(calcM - np.mod(M, 2 * np.pi))) < np.finfo(float).eps * 10.0
        )

    def test_eccanom_vecM_vece(self):
        """Test eccentric anomaly calculation for vector of eccentricities

        Use a normal distribution with sigma 2\pi to ensure that some M values are
        outside of (0,2\pi) range
        """

        M = np.random.randn(self.n) * 2 * np.pi
        e = np.random.rand(self.n)

        E = eccanom(M, e)
        calcM = E - e * np.sin(E)
        self.assertTrue(
            np.max(np.abs(calcM - np.mod(M, 2 * np.pi))) < np.finfo(float).eps * 10.0
        )

    def test_eccanom_vecM_vece_noc(self):
        """Test eccentric anomaly calculation for vector of eccentricities (python ver)

        Use a normal distribution with sigma 2\pi to ensure that some M values are
        outside of (0,2\pi) range
        """

        M = np.random.randn(self.n) * 2 * np.pi
        e = np.random.rand(self.n)

        E = eccanom(M, e, noc=True)
        calcM = E - e * np.sin(E)
        self.assertTrue(
            np.max(np.abs(calcM - np.mod(M, 2 * np.pi))) < np.finfo(float).eps * 10.0
        )

    def test_eccanom_scalarM_scalare(self):
        """Test eccentric anomaly calculation for scalar M, e"""

        M = np.random.rand(1)[0] * 2 * np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M, e)
        calcM = E - e * np.sin(E)
        self.assertTrue(
            np.max(np.abs(calcM - np.mod(M, 2 * np.pi))) < np.finfo(float).eps * 10.0
        )

        # Compare orvara and base eccanom versions
        E_orvara, *_ = eccanom_orvara(M, e)
        self.assertTrue(np.max(np.abs(E - E_orvara)) < np.finfo(float).eps * 100.0)

    def test_eccanom_scalarM_scalare_noc(self):
        """Test eccentric anomaly calculation for scalar M, e (python ver)"""

        M = np.random.rand(1)[0] * 2 * np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M, e, noc=True)
        calcM = E - e * np.sin(E)
        self.assertTrue(
            np.max(np.abs(calcM - np.mod(M, 2 * np.pi))) < np.finfo(float).eps * 10.0
        )

    def test_eccanom_badinputs(self):
        """Test incorrect inputs to eccanom"""

        M = np.random.rand(100)
        e = np.random.rand(30)

        with self.assertRaises(AssertionError):
            E = eccanom(M, e, noc=True)

    def test_orbElem2vec_1body(self):
        """Test state vector generation for single body"""

        # generate one body full orbit
        a = np.random.rand(1)[0] * 100
        e = np.random.rand(1)[0] * 0.95
        O = np.random.rand(1)[0] * 2 * np.pi
        I = np.random.rand(1)[0] * np.pi
        w = np.random.rand(1)[0] * 2 * np.pi
        E = np.linspace(
            0.01, 2 * np.pi - 0.01, 100
        )  # bad stuff happens in mod right at boundaries so clip E

        # convert back and forth to vectors
        rs, vs, ABs = orbElem2vec(E, 1.0, orbElem=(a, e, O, I, w), returnAB=True)
        a1, e1, E1, O1, I1, w1, P1, tau1 = vec2orbElem(rs, vs, 1.0)

        # check that you got back what you put in
        self.assertTrue(np.all(np.abs(a - a1) < np.sqrt(np.spacing(a))))
        self.assertTrue(np.max(np.abs(e - e1)) < self.etol)
        self.assertTrue(np.max(np.abs(O - O1)) < self.angtol)
        self.assertTrue(np.max(np.abs(w - w1)) < self.angtol)
        self.assertTrue(np.max(np.abs(I - I1)) < self.angtol)
        self.assertTrue(np.max(np.abs(E - E1)) < self.angtol)

        # convert back and forth again using A,B
        rs, vs = orbElem2vec(E, 1.0, AB=ABs)
        a1, e1, E1, O1, I1, w1, P1, tau1 = vec2orbElem(rs, vs, 1.0)

        self.assertTrue(np.all(np.abs(a - a1) < np.sqrt(np.spacing(a))))
        self.assertTrue(np.max(np.abs(e - e1)) < self.etol)
        self.assertTrue(np.max(np.abs(O - O1)) < self.angtol)
        self.assertTrue(np.max(np.abs(w - w1)) < self.angtol)
        self.assertTrue(np.max(np.abs(I - I1)) < self.angtol)
        self.assertTrue(np.max(np.abs(E - E1)) < self.angtol)

    def test_orbElem2vec_Nbody(self):
        """Test state vector generation for multiple bodies"""

        # generate n bodies at random phases
        n = self.n
        a = np.random.rand(n) * 100
        e = np.random.rand(n) * 0.95
        O = np.random.rand(n) * 2 * np.pi
        I = np.random.rand(n) * np.pi
        w = np.random.rand(n) * 2 * np.pi
        E = np.random.rand(n) * 2 * np.pi
        mus = np.random.rand(n)

        # convert back and forth to vectors
        rs, vs, ABs = orbElem2vec(E, mus, orbElem=(a, e, O, I, w), returnAB=True)
        a1, e1, E1, O1, I1, w1, P1, tau1 = vec2orbElem(rs, vs, mus)

        # check that you got back what you put in
        self.assertTrue(np.all(np.abs(a - a1) < np.sqrt(np.spacing(a))))
        self.assertTrue(np.max(np.abs(e - e1)) < self.etol)
        self.assertTrue(np.max(np.abs(O - O1)) < self.angtol)
        self.assertTrue(np.max(np.abs(w - w1)) < self.angtol)
        self.assertTrue(np.max(np.abs(I - I1)) < self.angtol)
        self.assertTrue(np.max(np.abs(E - E1)) < self.angtol)

        # convert back and forth again using A,B
        rs, vs = orbElem2vec(E, mus, AB=ABs)
        a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(rs, vs, mus)

        self.assertTrue(np.all(np.abs(a - a2) < np.sqrt(np.spacing(a))))
        self.assertTrue(np.max(np.abs(e - e2)) < self.etol)
        self.assertTrue(np.max(np.abs(O - O2)) < self.angtol)
        self.assertTrue(np.max(np.abs(w - w2)) < self.angtol)
        self.assertTrue(np.max(np.abs(I - I2)) < self.angtol)
        self.assertTrue(np.max(np.abs(E - E2)) < self.angtol)

    def test_vec2orbElem2(self):
        """Test state vector generation for multiple bodies Vallado version"""

        # generate n bodies at random phases
        n = self.n
        a = np.random.rand(n) * 100
        e = np.random.rand(n) * 0.95
        O = np.random.rand(n) * 2 * np.pi
        I = np.random.rand(n) * np.pi
        w = np.random.rand(n) * 2 * np.pi
        E = np.random.rand(n) * 2 * np.pi
        mus = np.random.rand(n)

        # convert back and forth to vectors
        rs, vs = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
        a1, e1, E1, O1, I1, w1, P1, tau1 = vec2orbElem2(rs, vs, mus)

        # check that you got back what you put in
        self.assertTrue(np.all(np.abs(a - a1) < np.sqrt(np.spacing(a))))
        self.assertTrue(np.max(np.abs(e - e1)) < self.etol)
        self.assertTrue(np.max(np.abs(O - O1)) < self.angtol)
        self.assertTrue(np.max(np.abs(w - w1)) < self.angtol)
        self.assertTrue(np.max(np.abs(I - I1)) < self.angtol)
        self.assertTrue(np.max(np.abs(E - E1)) < self.angtol)

    def test_trueanom(self):
        """Test True anomaly calculation"""
        n = self.n
        E = np.random.rand(n) * 2 * np.pi
        e = np.random.rand(n)
        nu = trueanom(E, e)

        self.assertTrue(
            np.max(
                np.abs(e + (1 - e**2) / (1 + e * np.cos(nu)) * np.cos(nu) - np.cos(E))
            )
            < 1e-9
        )

    def test_invkepler(self):
        """Test generalized kepler inverse function"""

        # first closed orbits
        M = np.random.rand(self.n) * 2 * np.pi
        e = np.linspace(0, 1 - self.etol, self.n)

        E, _, nu = invKepler(M, e, return_nu=True)

        self.assertTrue(
            np.max(
                np.abs(e + (1 - e**2) / (1 + e * np.cos(nu)) * np.cos(nu) - np.cos(E))
            )
            < self.etol
        )

        # now hyperbolae
        M = np.random.rand(self.n) * 200 - 100
        e = np.linspace(1 + self.etol, 25, self.n)

        H, _, nu = invKepler(M, e, return_nu=True)

        self.assertTrue(
            np.max(
                np.abs(
                    np.sqrt(e**2 - 1) * np.sinh(H) / (e * np.cosh(H) - 1) - np.sin(nu)
                )
            )
            < self.etol
        )

        # and some parabolae
        B, _, nu = invKepler(M, np.ones(self.n), return_nu=True)

        self.assertTrue(np.max(np.abs(B + B**3 / 3 - M)) < self.etol)

    def test_kepler_orbstate_roundtrip(self):
        """Test conversion back and forth between orbital state and Keplerian elements"""

        # semi-major axes uniformly distributed in -1,1
        a0 = np.random.rand(self.n) * 2 - 1
        e0 = np.zeros(self.n)  # assign eccentricities by orbit type
        closed = a0 > 0
        nclosed = len(np.where(closed)[0])
        e0[closed] = np.random.rand(nclosed)
        e0[~closed] = (np.random.rand(self.n - nclosed) + 1) * 5

        # force a few orbits to be parabolic and circular
        e0[np.where(closed)[0][:100]] = 0
        e0[np.where(~closed)[0][:100]] = 1
        a0[np.where(~closed)[0][:100]] *= -1

        mu = 1  # arbitrary mu, same for all orbits
        n = np.sqrt(1 / np.abs(a0) ** 3)  # mean motion
        n[e0 == 1] *= 2  # Treat a as semi-parameter for parabolae
        M0 = np.random.randn(self.n) * np.pi / 2  # randomize initial mean anomaly
        E0, _, nu0 = invKepler(M0, e0, return_nu=True)

        # randomly distribute orientation angles
        O0 = np.random.rand(self.n) * 2 * np.pi
        w0 = np.random.rand(self.n) * 2 * np.pi
        I0 = np.arccos(np.random.rand(self.n) * 2 - 1)
        # ensure a few zero inclination and a few pi:
        I0[np.random.choice(self.n, 50)] = 0
        I0[np.random.choice(self.n, 50)] = np.pi

        # Evaluate initial conditions and invert
        r0, v0 = kepler2orbstate(a0, e0, O0, I0, w0, mu, nu0)
        a1, e1, O1, I1, w1, _ = orbstate2kepler(r0, v0, mu)

        # handle special cases:
        zeroI = I0 == 0
        zeroe = e0 == 0
        zeroeI = zeroI & zeroe
        zeroI[zeroeI] = False
        zeroe[zeroeI] = False

        # e = I = 0: nu, w, Omega indistinguishable.  Put everything in nu and set w=O=0
        if np.any(zeroeI):
            nu0[zeroeI] = np.mod(nu0[zeroeI] + w0[zeroeI] + O0[zeroeI], 2 * np.pi)
            w0[zeroeI] = 0
            O0[zeroeI] = 0

        # I = 0: w, Omega indistinguishable. Put everything in omega and set O = 0
        if np.any(zeroI):
            w0[zeroI] = np.mod(w0[zeroI] + O0[zeroI], 2 * np.pi)
            O0[zeroI] = 0

        # e = 0: w, nu indistinguishable. Put everything in nu and set w = 0
        if np.any(zeroe):
            nu0[zeroe] = np.mod(w0[zeroe] + nu0[zeroe], 2 * np.pi)
            w0[zeroe] = 0

        self.assertTrue(np.all(np.abs(a0 - a1) < np.sqrt(np.spacing(np.abs(a0)))))
        self.assertTrue(np.max(np.abs(e0 - e1)) < self.etol)
        self.assertTrue(np.max(np.abs(O0 - O1)) < self.angtol)
        self.assertTrue(np.max(np.abs(w0 - w1)) < self.angtol)
        self.assertTrue(np.max(np.abs(I0 - I1)) < self.angtol)

    def test_universalfg(self):
        """Test f and g propagation"""

        # semi-major axes uniformly distributed in -1,1
        a0 = np.random.rand(self.n) * 2 - 1
        e0 = np.zeros(self.n)  # assign eccentricities by orbit type
        closed = a0 > 0
        nclosed = len(np.where(closed)[0])
        e0[closed] = np.random.rand(nclosed)
        e0[~closed] = (np.random.rand(self.n - nclosed) + 1) * 5

        # force a few orbits to be parabolic and circular
        e0[np.where(closed)[0][:100]] = 0
        e0[np.where(~closed)[0][:100]] = 1
        a0[np.where(~closed)[0][:100]] *= -1

        mu = 1  # arbitrary mu, same for all orbits
        n = np.sqrt(1 / np.abs(a0) ** 3)  # mean motion
        n[e0 == 1] *= 2  # Treat a as semi-parameter for parabolae
        M0 = np.random.randn(self.n) * np.pi / 2  # randomize initial mean anomaly
        _, _, nu0 = invKepler(M0, e0, return_nu=True, convergence_error=False)
        M1 = M0 + n * 1  # 1 time unit later
        _, _, nu1 = invKepler(M1, e0, return_nu=True, convergence_error=False)

        # randomly distribute orientation angles
        O0 = np.random.rand(self.n) * 2 * np.pi
        w0 = np.random.rand(self.n) * 2 * np.pi
        I0 = np.arccos(np.random.rand(self.n) * 2 - 1)
        # ensure a few zero inclination and a few pi:
        I0[np.random.choice(self.n, 50)] = 0
        I0[np.random.choice(self.n, 50)] = np.pi

        # Evaluate initial and final conditions
        r0, v0 = kepler2orbstate(a0, e0, O0, I0, w0, mu, nu0)
        r1, v1 = kepler2orbstate(a0, e0, O0, I0, w0, mu, nu1)

        # Now propgate via f and g for 1 time unit
        r1fg, v1fg, counter = universalfg(
            r0, v0, mu, 1, return_counter=True, convergence_error=False
        )

        r1mag = np.sqrt(np.sum(r1 * r1, axis=1))

        self.assertTrue(
            np.all(
                np.sqrt(np.sum((r1fg[counter < 100] - r1[counter < 100]) ** 2, axis=1))
                < np.sqrt(np.spacing(r1mag[counter < 100]))
            )
        )
