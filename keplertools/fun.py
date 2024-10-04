import warnings
from typing import Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import keplertools.Cyeccanom  # type: ignore
import keplertools.CyRV  # type: ignore

np.float_ = np.float64  # for numpy 2 compatibility
floatORarray = Union[float, npt.NDArray[np.float_]]


def eccanom(
    M: npt.ArrayLike,
    e: npt.ArrayLike,
    epsmult: float = 4.01,
    maxIter: int = 100,
    returnIter: bool = False,
    noc: bool = False,
    verb: bool = False,
) -> Union[Tuple[npt.NDArray[np.float_], int], npt.NDArray[np.float_]]:
    """Finds eccentric anomaly from mean anomaly and eccentricity

    This method uses Newton-Raphson iteration to find the eccentric
    anomaly from mean anomaly and eccentricity, assuming a closed (0<e<1)
    orbit.

    Args:
        M (float or ndarray):
            mean anomaly (rad)
        e (float or ndarray):
            eccentricity (eccentricity may be a scalar if M is given as
            an array, but otherwise must match the size of M.)
        epsmult (float):
            Precision of convergence (multiplied by precision of floating data type).
            Optional, defaults to 4.01.
        maxIter (int):
            Maximum numbr of iterations.  Optional, defaults to 100.
        returnIter (bool):
            Return number of iterations (defaults false, only available in python
            version, ignored if using C version)
        noc (bool):
            Don't use C version even if it can be loaded.
        verb (bool):
            Print exactly which version (C or Python is being used)

    Returns:
        tuple:
            E (float or ndarray):
                eccentric anomaly (rad)
            numIter (int):
                Number of iterations (returned only if returnIter=True)

    Notes:
        If either M or e are scalar, and the other input is an array, the scalar input
        will be expanded to the same size array as the other input.  So, a scalar M
        and array e will result in the calculation of the eccentric anomaly for one
        mean anomaly at a variety of eccentricities, and a scalar e and array M input
        will result in the calculation of eccentric anomalies for one eccentricity at
        a variety of mean anomalies.  If both inputs are arrays then they are matched
        element by element.

    """

    # make sure M and e are of the correct format.
    # if either is scalar, expand to match sizes
    M = np.array(M, ndmin=1).astype(float).flatten()
    e = np.array(e, ndmin=1).astype(float).flatten()
    if e.size != M.size:
        if e.size == 1:
            e = np.array([e[0]] * len(M))
        if M.size == 1:
            M = np.array([M[0]] * len(e))

    assert e.shape == M.shape, "Incompatible inputs."
    assert np.all((e >= 0) & (e < 1)), "e defined outside [0,1)"

    # force M into [0, 2*pi)
    M = np.mod(M, 2 * np.pi)

    if noc:
        if verb:
            print("Using Python version.")

        # initial values for E
        E = M / (1 - e)
        mask = e * E**2 > 6 * (1 - e)
        E[mask] = (6 * M[mask] / e[mask]) ** (1.0 / 3.0)

        # Newton-Raphson setup
        tolerance = np.finfo(float).eps * epsmult
        numIter = 0
        err = 1.0
        while err > tolerance and numIter < maxIter:
            E = E - (M - E + e * np.sin(E)) / (e * np.cos(E) - 1)
            err = np.max(abs(M - (E - e * np.sin(E))))
            numIter += 1

        if numIter == maxIter:
            raise ValueError("eccanom failed to converge. Final error of %e" % err)
    else:
        if verb:
            print("Using C version.")
        E = keplertools.Cyeccanom.Cyeccanom(M, e, epsmult, maxIter)
        returnIter = False

    if returnIter:
        return E, numIter
    else:
        return E  # type: ignore


def eccanom_orvara(
    M: npt.ArrayLike,
    e: npt.ArrayLike,
) -> Union[
    Tuple[npt.NDArray[np.float_], int],
    Tuple[npt.NDArray[np.float_], int],
    Tuple[npt.NDArray[np.float_], int],
]:
    """Finds eccentric anomaly E, sinE, cosE from mean anomaly and eccentricity

    This uses the method described in the orvara paper which uses a 5th order
    polynomial to approximate E and does a single Newton-Raphson iteration to
    refine it.

    Args:
        M (float or ndarray):
            mean anomaly (rad)
        e (float or ndarray):
            eccentricity (eccentricity may be a scalar if M is given as
            an array, but otherwise must match the size of M.)

    Returns:
        tuple:
            E (float or ndarray):
                eccentric anomaly (rad)
            sinE (float or ndarray):
                Sine of eccentric anomaly (rad)
            cosE (float or ndarray):
                Cosine of eccentric anomaly (rad)

    Notes:
        If either M or e is scalar, and the other input is an array, the scalar input
        will be expanded to the same size array as the other input. If both inputs are
        arrays then they are matched element by element.

    """

    e_is_scalar = np.isscalar(e)
    if np.isscalar(M):
        M = np.array(M, ndmin=1).astype(float).flatten()

    # Force M into [0, 2*pi)
    M = np.mod(M, 2 * np.pi)

    # If there is only one unique eccentricity, process all M values at once
    if e_is_scalar:
        E, sinE, cosE = keplertools.Cyeccanom.Cyeccanom_orvara(M, e)
    else:
        print(
            "NOTE: The orvara method is optimized for one eccentricity and you are providing multiple."
        )
        # Broadcasting to ensure M and e are compatible
        M, e = np.broadcast_arrays(M.ravel(), np.array([e]))

        # Apply the Cyeccanom_orvara function over each pair of M and e using np.vectorize
        vectorized_orvara = np.vectorize(
            orvara_vector_helper,
            otypes=[np.float_, np.float_, np.float_],
        )
        E, sinE, cosE = vectorized_orvara(M, e)

    return E, sinE, cosE


def orvara_vector_helper(M_val, e_val):
    """Wraps the Cyeccanom_orvara function to handle single M value as array."""
    E, sinE, cosE = keplertools.Cyeccanom.Cyeccanom_orvara(np.array([M_val]), e_val)
    return E[0], sinE[0], cosE[0]


def trueanom(E: npt.ArrayLike, e: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Finds true anomaly from eccentric anomaly and eccentricity

    The implemented method corresponds to Eq. 6.28 in Green assuming a closed
    (0<e<1) orbit.


    Args:
        E (float or ndarray):
            eccentric anomaly (rad)
        e (float or ndarray):
            eccentricity (eccentricity may be a scalar if M is given as
            an array, but otherwise must match the size of M.)

    Returns:
        ndarray:
            true anomaly (rad)

    Notes:
        If either E or e are scalar, and the other input is an array, the scalar
        input will be expanded to the same size array as the other input.

    """

    E = np.array(E, ndmin=1).astype(float).flatten()
    e = np.array(e, ndmin=1).astype(float).flatten()
    if e.size != E.size:
        if e.size == 1:
            e = np.array([e[0]] * len(E))
        if E.size == 1:
            E = np.array([E[0]] * len(e))

    assert e.shape == E.shape, "Incompatible inputs."
    assert np.all((e >= 0) & (e < 1)), "e defined outside [0,1)"

    nu = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0))
    nu[nu < 0] += 2 * np.pi

    return nu  # type: ignore


def vec2orbElem2(
    rs: npt.NDArray[np.float_],
    vs: npt.NDArray[np.float_],
    mus: Union[float, npt.NDArray[np.float_]],
) -> Tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
]:
    """Convert position and velocity vectors to Keplerian orbital elements

    Implements the algorithm from Vallado

    Args:
        rs (ndarray):
            3n x 1 stacked initial position vectors:
            [r1(1);r1(2);r1(3);r2(1);r2(2)r2(3);...;rn(1);rn(2);rn(3)]
            or 3 x n or n x 3 matrix of position vectprs.
        vs (ndarray):
            3n x 1 stacked initial velocity vectors or 3 x n or n x3 matrix
        mus (ndarray or float):
            nx1 array of gravitational parameters (G*m_i) where G is the
            gravitational constant and m_i is the mass of the ith body.
            if all vectors represent the same body, mus may be a scalar.

    Returns:
        tuple:
            a (ndarray):
                Semi-major axes
            e (ndarray):
                eccentricities
            E (ndarray):
                eccentric anomalies
            O (ndarray):
                longitudes of ascending nodes (rad)
            I (ndarray):
                inclinations (rad)
            w (ndarray):
                arguments of pericenter (rad)
            P (ndarray):
                orbital periods
            tau (ndarray):
                time of periapsis crossing


    Notes:
        All units must be complementary, i.e., if positions are in AU, and time is in
        days, vs must be in AU/day, mus must be in AU^3/day^2


    """
    assert (np.mod(rs.size, 3) == 0) and (
        vs.size == rs.size
    ), "rs and vs must be of the same size and contain 3n elements."

    nplanets = int(rs.size / 3.0)
    if not (np.isscalar(mus)):
        assert mus.size == nplanets, "mus must be scalar or of size n"  # type: ignore

    assert rs.ndim < 3, "rs cannot have more than two dimensions"
    if rs.ndim == 1:
        rs = np.reshape(rs, (nplanets, 3)).T
    else:
        assert 3 in rs.shape, "rs must be 3xn or nx3"
        if rs.shape[0] != 3:
            rs = rs.T

    assert vs.ndim < 3, "vs cannot have more than two dimensions"
    if vs.ndim == 1:
        vs = np.reshape(vs, (nplanets, 3)).T
    else:
        assert 3 in vs.shape, "vs must be 3xn or nx3"
        if vs.shape[0] != 3:
            vs = vs.T

    v2s = np.sum(vs**2.0, axis=0)  # orbital velocity squared
    rmag = np.sqrt(np.sum(rs**2.0, axis=0))  # orbital radius

    hvec = np.vstack(
        (
            rs[1] * vs[2] - rs[2] * vs[1],
            rs[2] * vs[0] - rs[0] * vs[2],
            rs[0] * vs[1] - rs[1] * vs[0],
        )
    )  # angular momentum vector
    nvec = np.vstack(
        (
            -hvec[1],
            hvec[0],
            np.zeros(len(hvec[2])),
        )
    )  # node-pointing vector
    evec = (
        np.tile((v2s - mus / rmag) / mus, (3, 1)) * rs
        - np.tile(np.sum(rs * vs, axis=0) / mus, (3, 1)) * vs
    )  # eccentricity vector
    nmag = np.sqrt(np.sum(nvec**2.0, axis=0))
    e = np.sqrt(np.sum(evec**2.0, axis=0))

    En = v2s / 2 - mus / rmag
    a = -mus / 2 / En
    ell = a * (1 - e**2)
    if np.any(e == 1):
        tmp = np.sum(hvec**2.0, axis=0) / mus
        ell[e == 1] = tmp[e == 1]

    # angles
    I = np.arccos(hvec[2] / np.sqrt(np.sum(hvec**2.0, axis=0)))
    O = np.mod(np.arctan2(nvec[1], nvec[0]), 2 * np.pi)
    w = np.arccos(np.sum(nvec * evec, axis=0) / e / nmag)
    w[evec[2] < 0] = 2 * np.pi - w[evec[2] < 0]

    # ecentric anomaly
    cosE = (1.0 - rmag / a) / e
    sinE = np.sum(rs * vs, axis=0) / (e * np.sqrt(mus * a))
    E = np.mod(np.arctan2(sinE, cosE), 2 * np.pi)

    # orbital periods
    P = 2 * np.pi * np.sqrt(a**3.0 / mus)

    # time of periapsis crossing
    tau = -(E - e * np.sin(E)) / np.sqrt(mus * a**-3.0)

    return a, e, E, O, I, w, P, tau


def vec2orbElem(
    rs: npt.NDArray[np.float_],
    vs: npt.NDArray[np.float_],
    mus: Union[float, npt.NDArray[np.float_]],
) -> Tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
]:
    """Convert position and velocity vectors to Keplerian orbital elements

    Implements the (corrected) algorithm from Vinti

    Args:
        rs (ndarray):
            3n x 1 stacked initial position vectors:
            [r1(1);r1(2);r1(3);r2(1);r2(2)r2(3);...;rn(1);rn(2);rn(3)]
            or 3 x n or n x 3 matrix of position vectors.
        vs (ndarray):
            3n x 1 stacked initial velocity vectors or 3 x n or n x3 matrix
        mus (ndarray or float):
            nx1 array of gravitational parameters (G*m_i) where G is the
            gravitational constant and m_i is the mass of the ith body.
            if all vectors represent the same body, mus may be a scalar.

    Returns:
        tuple:
            a (ndarray):
                Semi-major axes
            e (ndarray):
                eccentricities
            E (ndarray):
                eccentric anomalies
            O (ndarray):
                longitudes of ascending nodes (rad)
            I (ndarray):
                inclinations (rad)
            w (ndarray):
                arguments of pericenter (rad)
            P (ndarray):
                orbital periods
            tau (ndarray):
                time of periapsis crossing

    Notes:
        All units must be complementary, i.e., if positions are in AU, and time is in
        days, vs must be in AU/day, mus must be in AU^3/day^2


    """
    assert (np.mod(rs.size, 3) == 0) and (
        vs.size == rs.size
    ), "rs and vs must be of the same size and contain 3n elements."

    nplanets = int(rs.size / 3.0)
    if not (np.isscalar(mus)):
        assert mus.size == nplanets, "mus must be scalar or of size n"  # type: ignore

    assert rs.ndim < 3, "rs cannot have more than two dimensions"
    if rs.ndim == 1:
        rs = np.reshape(rs, (nplanets, 3)).T
    else:
        assert 3 in rs.shape, "rs must be 3xn or nx3"
        if rs.shape[0] != 3:
            rs = rs.T

    assert vs.ndim < 3, "vs cannot have more than two dimensions"
    if vs.ndim == 1:
        vs = np.reshape(vs, (nplanets, 3)).T
    else:
        assert 3 in vs.shape, "vs must be 3xn or nx3"
        if vs.shape[0] != 3:
            vs = vs.T

    v2s = np.sum(vs**2.0, axis=0)  # orbital velocity squared
    r = np.sqrt(np.sum(rs**2.0, axis=0))  # orbital radius
    Ws = 0.5 * v2s - mus / r  # Keplerian orbital energy
    a = -mus / 2.0 / Ws
    # semi-major axis

    L = np.vstack(
        (
            rs[1] * vs[2] - rs[2] * vs[1],
            rs[2] * vs[0] - rs[0] * vs[2],
            rs[0] * vs[1] - rs[1] * vs[0],
        )
    )  # angular momentum vector
    L2s = np.sum(L**2.0, axis=0)  # angular momentum squared
    Ls = np.sqrt(L2s)  # angular momentum
    p = L2s / mus  # semi-parameter
    e = np.sqrt(1.0 - p / a)  # eccentricity

    # ecentric anomaly
    cosE = (1.0 - r / a) / e
    sinE = np.sum(rs * vs, axis=0) / (e * np.sqrt(mus * a))
    E = np.mod(np.arctan2(sinE, cosE), 2 * np.pi)

    # inclination (strictly in (0,pi))
    I = np.arccos(L[2] / Ls)
    sinI = np.sqrt(L[0] ** 2 + L[1] ** 2.0) / Ls

    # argument of pericenter
    esinwsinI = (vs[0] * L[1] - vs[1] * L[0]) / mus - rs[2] / r
    ecoswsinI = (Ls * vs[2, :]) / mus - (L[0] * rs[1] - L[1] * rs[0]) / (Ls * r)
    w = np.mod(np.arctan2(esinwsinI, ecoswsinI), 2 * np.pi)

    # longitude of ascending node
    cosO = -L[1] / (Ls * sinI)
    sinO = L[0] / (np.sqrt(L2s) * sinI)
    O = np.mod(np.arctan2(sinO, cosO), 2 * np.pi)

    # orbital periods
    P = 2 * np.pi * np.sqrt(a**3.0 / mus)

    # time of periapsis crossing
    tau = -(E - e * np.sin(E)) / np.sqrt(mus * a**-3.0)

    return a, e, E, O, I, w, P, tau


def calcAB(
    a: npt.NDArray[np.float_],
    e: npt.NDArray[np.float_],
    O: npt.NDArray[np.float_],
    I: npt.NDArray[np.float_],
    w: npt.NDArray[np.float_],
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    r"""Calculate inertial frame components of perifocal frame unit vectors scaled
    by orbit semi-major and semi-minor axes.

    Note that these quantities are closely related to the Thiele-Innes constants

    Args:
        a (ndarray):
            Semi-major axes
        e (ndarray):
            eccentricities
        O (ndarray):
            longitudes of ascending nodes (rad)
        I (ndarray):
            inclinations (rad)
        w (ndarray):
            arguments of pericenter (rad)

    Returns:
        tuple:
            A (ndarray):
                Components of eccentricity vector scaled by a
            B (ndarray):
                Components of q vector (orthogonal to e and h) scaled by
                b (=a\sqrt{1-e^2})

    Notes:
        All inputs must be of same size.  Outputs are 3xn for n input points.
        See Vinti (1998) for details on element/coord sys defintions.

    """

    assert a.size == e.size == O.size == I.size == w.size

    A = np.vstack(
        (
            a * (np.cos(O) * np.cos(w) - np.sin(O) * np.cos(I) * np.sin(w)),
            a * (np.sin(O) * np.cos(w) + np.cos(O) * np.cos(I) * np.sin(w)),
            a * np.sin(I) * np.sin(w),
        )
    )

    B = np.vstack(
        (
            -a
            * np.sqrt(1 - e**2)
            * (np.cos(O) * np.sin(w) + np.sin(O) * np.cos(I) * np.cos(w)),
            a
            * np.sqrt(1 - e**2)
            * (-np.sin(O) * np.sin(w) + np.cos(O) * np.cos(I) * np.cos(w)),
            a * np.sqrt(1 - e**2) * np.sin(I) * np.cos(w),
        )
    )

    return A, B


def orbElem2vec(
    E: npt.NDArray[np.float_],
    mus: Union[float, npt.NDArray[np.float_]],
    orbElem: Optional[
        Tuple[
            npt.NDArray[np.float_],
            npt.NDArray[np.float_],
            npt.NDArray[np.float_],
            npt.NDArray[np.float_],
            npt.NDArray[np.float_],
        ]
    ] = None,
    AB: Optional[Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]] = None,
    returnAB: bool = False,
) -> Union[
    Tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        Tuple[
            npt.NDArray[np.float_],
            npt.NDArray[np.float_],
        ],
    ],
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
]:
    """Convert Keplerian orbital elements to position and velocity vectors

    Args:
        E (ndarray):
            nx1 array of eccentric anomalies (rad)
        mus (ndarray or float):
            nx1 array of gravitational parameters (G*m_i) where G is the gravitational
            constant and m_i is the mass of the ith body. if all vectors represent the
            same body, mus may be a scalar.
        orbElem (tuple):
            (a,e,O,I,w) Exact inputs to calcAB. Either this or AB input must be set
        AB (tuple):
            (A,B) Exact outpus from calcAB
        returnAB (bool):
            Default False. If True, returns (A,B) as thrid output.

    Returns:
        tuple:
            rs (ndarray):
                3 x n stacked position vectors
            vs (ndarray):
                3 x n stacked velocity vectors
            AB (tuple):
                (A,B)

    Notes:
        All units are complementary, i.e., if mus are in AU^3/day^2 then
        positions will be in AU, and velocities will be AU/day.

        Possible combinations or inputs are:

        1. E scalar, mu scalar - single body, single position.
           A, B should be 3x1 (or orbElem should be all scalars).

        2. E vector, mu scalar - single body, many orbital positions.
           A, B should be 3x1 (or orbElem should be all scalars).

        3. E vector, mu vector - multiple bodies at varying orbital positions.
           A, B should be 3xn where E.size==n (or all orbElem should be size n)
           and mus.size must equal E.size.

    """

    assert (orbElem is not None) or (
        AB is not None
    ), "You must supply either orbElem or AB inputs."
    if np.isscalar(E):
        assert np.isscalar(mus), "Scalar E input requires scalar mus input (one body)."
        E = np.array(E, ndmin=1)
    else:
        assert np.isscalar(mus) or (
            mus.size == E.size  # type: ignore
        ), "mus must be of the same size as E or scalar."
    if orbElem is not None:
        assert AB is None, "You can only set orbElem or AB."
        A, B = calcAB(orbElem[0], orbElem[1], orbElem[2], orbElem[3], orbElem[4])
        a = orbElem[0]
        e = orbElem[1]
    if AB is not None:
        assert orbElem is None, "You can only set orbElem or AB."
        A = AB[0]
        B = AB[1]
        a = np.linalg.norm(A, axis=0)
        e = np.sqrt(1 - (np.linalg.norm(B, axis=0) / a) ** 2.0)
    if np.isscalar(E) or np.isscalar(mus):
        assert (A.size == 3) and (
            B.size == 3
        ), "A and B must be 3x1 for scalar E or mu (one body)."
    if not (np.isscalar(E)) and not (np.isscalar(mus)):
        assert (A.size == 3 * E.size) and (
            B.size == 3 * E.size
        ), "A and B must be 3xn for vector E (multiple bodies)."

    if np.isscalar(mus) and not (np.isscalar(E)):
        r = np.matmul(A, np.array((np.cos(E) - e), ndmin=2)) + np.matmul(
            B, np.array(np.sin(E), ndmin=2)
        )
        v = (
            np.matmul(-A, np.array(np.sin(E), ndmin=2))
            + np.matmul(B, np.array(np.cos(E), ndmin=2))
        ) * np.tile(
            np.sqrt(mus * a ** (-3.0)) / (1 - e * np.cos(E)),
            (3, 1),  # type:ignore
        )
    else:
        r = np.matmul(A, np.diag(np.cos(E) - e)) + np.matmul(B, np.diag(np.sin(E)))
        v = np.matmul(
            np.matmul(-A, np.diag(np.sin(E))) + np.matmul(B, np.diag(np.cos(E))),
            np.diag(np.sqrt(mus * a ** (-3.0)) / (1 - e * np.cos(E))),
        )

    if returnAB:
        return r, v, (A, B)
    else:
        return r, v


def forcendarray(x: floatORarray) -> npt.NDArray[np.float_]:
    """Convert any numerical value into 1-D ndarray

    Args:
        x (float or numpy.ndarray):
            Input
    Returns:
        numpy.ndarray:
            Same size as input but in ndarray form
    """

    return np.array(x, ndmin=1).astype(float).flatten()


def validateOrbitalStateInputs(
    r: npt.NDArray[np.float_], v: npt.NDArray[np.float_], mu: floatORarray
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Validate and standardize dimensionality of orbital state vector inputs

    Args:
        r (numpy.ndarray):
            Components of orbital radius. 3n elements in 1D as
            [r1(1);r1(2);r1(3);r2(1);r2(2)r2(3);...;rn(1);rn(2);rn(3)]
            or in 2D as nx3 or 3xn
        v (numpy.ndarray):
            Components of orbital velocity. Same stacking as r
        mu (float or numpy.ndarray):
            Gravitational parameters.  If float, assuming all state vectors belong to
            the same system.

    Returns:
        tuple:
        r (numpy.ndarray):
            Components of orbital radius. (n x 3)
        v (numpy.ndarray):
            Components of orbital velocity. (n x 3)
        mu (numpy.ndarray):
            Gravitational parameters.  (size 1 or n)
    """
    # figure out dimensionality of inputs
    assert len(r.shape) <= 3, "r input must have max dimension 2"
    if (len(r.shape) == 1) or (1 in r.shape):
        r = r.flatten().reshape(r.size // 3, 3)
    else:
        assert 3 in r.shape, "If r is 2D, one dimension must be length 3"
        if r.shape[0] == 3:
            r = r.transpose()

    assert len(v.shape) <= 3, "v input must have max dimension 2"
    if (len(v.shape) == 1) or (1 in v.shape):
        v = v.flatten().reshape(v.size // 3, 3)
    else:
        assert 3 in v.shape, "If v is 2D, one dimension must be length 3"
        if v.shape[0] == 3:
            v = v.transpose()

    mu = forcendarray(mu)

    assert len(r) == len(v), "r and v must have same sizes."
    assert mu.size == 1 or mu.size == len(
        r
    ), "mu must be scalar or same length as r and v"

    return r, v, mu


def unitvector(
    vec: npt.NDArray[np.float_], mag: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Return the unit vectors of an array of vectors

    Args:
        vec(numpy.ndarray):
            Vectors as nx3
        mag (numpy.ndarray):
            Vector magnitudes as nx1

    Returns:
        numpy.ndarray:
            Unit vectors in the same layout as input
    """

    with np.errstate(divide="ignore"):
        out = vec / np.tile(mag, (3, 1)).transpose()

    return out


def invKepler(
    M: floatORarray,
    e: floatORarray,
    tol: Optional[float] = None,
    E0: Optional[floatORarray] = None,
    maxIter: int = 100,
    return_nu: bool = False,
    convergence_error: bool = True,
) -> Tuple[npt.NDArray[np.float_], ...]:
    """Finds eccentric/hyperbolic/parabolic anomaly from mean anomaly and eccentricity

    This method uses Newton-Raphson iteration to find the eccentric
    anomaly from mean anomaly and eccentricity, assuming a closed (0<e<1)
    orbit.

    Args:
        M (float or ndarray):
            mean anomaly (rad)
        e (float or ndarray):
            eccentricity (eccentricity may be a scalar if M is given as
            an array, but otherwise must match the size of M.)
        tol (float):
            Convergence of tolerance. Defaults to eps(2*pi)
        E0 (float or ndarray):
            Initial guess for iteration.  Defaults to Taylor-expansion based value for
            closed orbits and Vallado-derived heuristic for open orbits. If set, must
            match size of M.
        maxIter (int):
            Maximum numbr of iterations.  Optional, defaults to 100.
        return_nu (bool):
            Return true anomaly (defaults false)
        convergence_error (bool):
            Raise error on convergence failure. Defaults True.  If false, throws a
            warning.

    Returns:
        tuple:
            E (ndarray):
                eccentric/parabolic/hyperbolic anomaly (rad)
            numIter (ndarray):
                Number of iterations
            nu (ndarray):
                True anomaly (returned only if return_nu=True)

    Notes:
        If either M or e are scalar, and the other input is an array, the scalar input
        will be expanded to the same size array as the other input.  So, a scalar M
        and array e will result in the calculation of the eccentric anomaly for one
        mean anomaly at a variety of eccentricities, and a scalar e and array M input
        will result in the calculation of eccentric anomalies for one eccentricity at
        a variety of mean anomalies.  If both inputs are arrays then they are matched
        element by element.

    """

    # make sure M and e are of the correct format.
    # if either is scalar, expand to match sizes
    M = forcendarray(M)
    e = forcendarray(e)
    if e.size != M.size:
        if e.size == 1:
            e = np.array([e[0]] * len(M))
        if M.size == 1:
            M = np.array([M[0]] * len(e))

    assert e.shape == M.shape, "Incompatible inputs."
    assert np.all((e >= 0)), "e values below zero"

    if E0 is not None:
        E0 = forcendarray(E0)
        assert E0.shape == M.shape, "Incompatible inputs."

    # define output
    Eout = np.zeros(M.size)
    numIter = np.zeros(M.size, dtype=int)

    # circles
    cinds = e == 0
    Eout[cinds] = M[cinds]

    # ellipses
    einds = (e > 0) & (e < 1)
    if any(einds):
        Me = np.mod(M[einds], 2 * np.pi)
        ee = e[einds]

        # initial values for E
        if E0 is None:
            E = Me / (1 - ee)
            mask = ee * E**2 > 6 * (1 - ee)
            E[mask] = np.cbrt(6 * Me[mask] / ee[mask])
        else:
            E = E0[einds]

        # Newton-Raphson setup
        counter = np.ones(E.shape)
        err = np.ones(E.shape)

        # set tolerance is none provided
        if tol is None:
            etol = np.spacing(2 * np.pi)
        else:
            etol = tol

        while (np.max(err) > etol) and (np.max(counter) < maxIter):
            inds = err > etol
            E[inds] = E[inds] - (Me[inds] - E[inds] + ee[inds] * np.sin(E[inds])) / (
                ee[inds] * np.cos(E[inds]) - 1
            )
            err[inds] = np.abs(Me[inds] - (E[inds] - ee[inds] * np.sin(E[inds])))
            counter[inds] += 1

        if np.max(counter) == maxIter:
            if convergence_error:
                raise ValueError("Maximum number of iterations exceeded")
            else:
                warnings.warn("Maximum number of iterations exceeded")

        Eout[einds] = E
        numIter[einds] = counter

    # parabolae
    pinds = e == 1
    if np.any(pinds):
        q = 9 * M[pinds] / 6
        B = (q + np.sqrt(q**2 + 1)) ** (1.0 / 3.0) - (np.sqrt(q**2 + 1) - q) ** (
            1.0 / 3.0
        )
        Eout[pinds] = B

    # hyperbolae
    hinds = e > 1
    if np.any(hinds):
        Mh = M[hinds]
        eh = e[hinds]

        # initialize H
        if E0 is None:
            H = Mh / (eh - 1)
            mask = eh * H**2 > 6 * (eh - 1)
            H[mask] = np.cbrt(6 * Mh[mask] / eh[mask])
        else:
            H = E0[hinds]

        # Newton-Raphson setup
        counter = np.ones(H.shape)

        # set tolerance is none provided
        if tol is None:
            htol = 4 * np.spacing(np.abs(H))
        else:
            htol = np.ones(H.shape) * tol

        Hup = np.ones(len(H))
        # we will only update things that haven't hit their tolerance:
        inds = np.abs(Hup) > htol

        while np.any(inds) and np.all(counter < maxIter):
            Hup[inds] = (Mh[inds] - eh[inds] * np.sinh(H[inds]) + H[inds]) / (
                eh[inds] * np.cosh(H[inds]) - 1
            )
            H[inds] = H[inds] + Hup[inds]
            if tol is None:
                htol[inds] = 4 * np.spacing(np.abs(H[inds]))

            counter[inds] += 1
            inds = np.abs(Hup) > htol

        if np.max(counter) == maxIter:
            if convergence_error:
                raise ValueError("Maximum number of iterations exceeded")
            else:
                warnings.warn("Maximum number of iterations exceeded")

        Eout[hinds] = H
        numIter[hinds] = counter

    out: Tuple[npt.NDArray[np.float_], ...] = (Eout, numIter)
    if return_nu:
        nuout = np.zeros(M.size)

        # circles
        nuout[cinds] = M[cinds]

        # ellipses
        if np.any(einds):
            nuout[einds] = np.mod(
                2 * np.arctan(np.sqrt((1 + e[einds]) / (1 - e[einds])) * np.tan(E / 2)),
                2 * np.pi,
            )

        # parabolae
        if np.any(pinds):
            nuout[pinds] = 2 * np.arctan(B)

        # hyperbolae
        if np.any(hinds):
            nuout[hinds] = 2 * np.arctan(
                np.sqrt((e[hinds] + 1) / (e[hinds] - 1)) * np.tanh(H / 2)
            )

        out += (nuout,)

    return out


def kepler2orbstate(
    a: floatORarray,
    e: floatORarray,
    O: floatORarray,
    I: floatORarray,
    w: floatORarray,
    mu: floatORarray,
    nu: floatORarray,
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Calculate orbital state vectors from Keplerian elements

    Args:
        a (float or numpy.ndarray):
            Semi-major axis (or semi-parameter is e = 1)
        e (float or numpy.ndarray):
            eccentricity
        O (float or numpy.ndarray):
            longitude of ascending node (rad)
        I (float or numpy.ndarray):
            inclination (rad)
        w (float or numpy.ndarray):
            arguments of periapsis (rad)
        mu (float or numpy.ndarray):
            Gravitational parameters.  If float, assuming all state vectors belong to
            the same system.
        nu (float or numpy.ndarray):
            True anomaly (rad)

    Returns:
        tuple:
            r (numpy.ndarray):
                Components of orbital radius (n x 3)
            v (numpy.ndarray):
                Components of orbital velocity (n x 3)

    Notes:
        r.flatten() and v.flatten() will automatically stack elements in the proper
        order in a 1D array

    """

    # force all inputs to ndarrays
    a = forcendarray(a)
    e = forcendarray(e)
    O = forcendarray(O)
    I = forcendarray(I)
    w = forcendarray(w)
    mu = forcendarray(mu)
    nu = forcendarray(nu)

    # semi-parameter
    ell = a * (1 - e**2)
    ell[e == 1] = a[e == 1]

    # specific angular momentum
    h = np.sqrt(mu * ell)

    # orbital radius
    r = ell / (1 + e * np.cos(nu))

    r = np.vstack(
        [
            r * (-np.sin(O) * np.sin(nu + w) * np.cos(I) + np.cos(O) * np.cos(nu + w)),
            r * (np.sin(O) * np.cos(nu + w) + np.sin(nu + w) * np.cos(I) * np.cos(O)),
            r * np.sin(I) * np.sin(nu + w),
        ]
    ).transpose()

    v = np.vstack(
        [
            -mu
            * (
                e * np.sin(O) * np.cos(I) * np.cos(w)
                + e * np.sin(w) * np.cos(O)
                + np.sin(O) * np.cos(I) * np.cos(nu + w)
                + np.sin(nu + w) * np.cos(O)
            )
            / h,
            mu
            * (
                -e * np.sin(O) * np.sin(w)
                + e * np.cos(I) * np.cos(O) * np.cos(w)
                - np.sin(O) * np.sin(nu + w)
                + np.cos(I) * np.cos(O) * np.cos(nu + w)
            )
            / h,
            mu * (e * np.cos(w) + np.cos(nu + w)) * np.sin(I) / h,
        ]
    ).transpose()

    return r, v


def orbstate2kepler(
    r: npt.NDArray[np.float_], v: npt.NDArray[np.float_], mu: floatORarray
) -> Tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
]:
    """Calculate  Keplerian elements given orbital state vectors

    Args:
        r (numpy.ndarray):
            Components of orbital radius. 3n elements in 1D as
            [r1(1);r1(2);r1(3);r2(1);r2(2)r2(3);...;rn(1);rn(2);rn(3)]
            or in 2D as nx3 or 3xn
        v (numpy.ndarray):
            Components of orbital velocity. Same stacking as r
        mu (float or numpy.ndarray):
            Gravitational parameters.  If float, assuming all state vectors belong to
            the same system.

    Returns:
        tuple:
            a (ndarray):
                Semi-major axis (or semi-parameter where e = 1)
            e (ndarray):
                eccentricity
            O (ndarray):
                longitude of ascending node (rad)
            I (ndarray):
                inclination (rad)
            w (ndarray):
                arguments of periapsis (rad)
            tp (ndarray):
                time of periapsis passage
    """
    r, v, mu = validateOrbitalStateInputs(r, v, mu)

    v2 = np.sum(v * v, axis=1)  # velocity magnitude squared
    rmag = np.sqrt(np.sum(r * r, axis=1))  # orbital radius magnitude

    hvec = np.cross(r, v)  # specific angular momentum vector
    h2 = np.sum(hvec * hvec, axis=1)  # magnitude squared
    hmag = np.sqrt(h2)  # momentum magnitude
    hhat = unitvector(hvec, hmag)  # unit vector

    # line of nodes:
    nvec = np.vstack([-hvec[:, 1], hvec[:, 0], np.zeros(len(hvec))]).transpose()
    nmag = np.sqrt(np.sum(nvec * nvec, axis=1))  # magnitude
    nhat = unitvector(nvec, nmag)  # unit vector

    # eccentricity vector:
    evec = (
        np.tile(v2 / mu - 1 / rmag, (3, 1)).transpose() * r
        - np.tile(np.sum(r * v, axis=1), (3, 1)).transpose() * v
    )
    e = np.sqrt(np.sum(evec * evec, axis=1))  # eccentricity magnitude
    ehat = unitvector(evec, e)

    En = v2 / 2 - mu / rmag  # specific energy
    with np.errstate(divide="ignore"):
        a = -mu / 2 / En  # semi-major axis

    # handle parabolas (a var is redefined as ell for these cases)
    e[np.abs(e - 1) < 100 * np.spacing(1)] = 1  # grab all cases where e~1
    pinds = e == 1
    if np.any(pinds):
        if mu.size == 1:
            a[pinds] = h2[pinds] / mu
        else:
            a[pinds] = h2[pinds] / mu[pinds]

    # inclination
    sinI = np.sqrt(np.sum(hhat[:, [0, 1]] ** 2, axis=1))
    cosI = hhat[:, 2]
    I = np.arctan2(sinI, cosI)

    # longitude of the ascending node
    O = np.mod(np.arctan2(hvec[:, 0], -hvec[:, 1]), 2 * np.pi)

    # argument of periapsis
    sinw = np.sum(np.cross(nhat, ehat) * hhat, axis=1)
    cosw = np.sum(ehat * nhat, axis=1)
    w = np.mod(np.arctan2(sinw, cosw), 2 * np.pi)

    # true anomaly
    cosnu = np.sum(ehat * r, axis=1) / rmag
    sinnu = np.sum(np.cross(ehat, r) * hhat, axis=1) / rmag
    nu = np.arctan2(sinnu, cosnu)

    # special cases:
    zeroI = I == 0
    zeroe = e < np.spacing(10)
    zeroeI = zeroI & zeroe
    zeroI[zeroeI] = False
    zeroe[zeroeI] = False

    # e = I = 0: nu, w, Omega indistinguishable.  Put everything in nu and set w=O=0
    if np.any(zeroeI):
        w[zeroeI] = 0
        O[zeroeI] = 0
        nu[zeroeI] = np.mod(np.arctan2(r[zeroeI, 1], r[zeroeI, 0]), 2 * np.pi)

    # I = 0: w, Omega indistinguishable. Put everything in omega and set O = 0
    if np.any(zeroI):
        O[zeroI] = 0
        w[zeroI] = np.mod(np.arctan2(evec[zeroI, 1], evec[zeroI, 0]), 2 * np.pi)

    # e = 0: w, nu indistinguishable. Put everything in nu and set w = 0
    if np.any(zeroe):
        cosn = np.sum(nhat[zeroe, :] * r[zeroe, :], axis=1) / rmag[zeroe]
        sinn = (
            np.sum(np.cross(nhat[zeroe, :], r[zeroe, :]) * hhat[zeroe, :], axis=1)
            / rmag[zeroe]
        )
        nu[zeroe] = np.mod(np.arctan2(sinn, cosn), 2 * np.pi)
        w[zeroe] = 0

    # finally, periapsis time is orbit-type specific
    E = np.zeros(len(r))  # eccentric/parabolic/hyperbolic anomaly

    # elliptic case:
    einds = e < 1
    E[einds] = np.mod(
        2 * np.arctan(np.sqrt((1 - e[einds]) / (1 + e[einds])) * np.tan(nu[einds] / 2)),
        2 * np.pi,
    )

    # parabolic case:
    E[pinds] = np.tan(nu[pinds] / 2)

    # hyperbolic case:
    hinds = e > 1
    E[hinds] = 2 * np.arctanh(
        np.sqrt((e[hinds] - 1) / (e[hinds] + 1)) * np.tan(nu[hinds] / 2)
    )

    # mean motion
    n = np.sqrt(mu / np.abs(a) ** 3)
    # handle parabolas
    if np.any(pinds):
        n[pinds] *= 2

    # periapse passage time
    tp = np.zeros(len(r))

    # elliptic
    tp[einds] = -(E[einds] - e[einds] * np.sin(E[einds])) / n[einds]

    # hyperbolic
    tp[hinds] = -(e[hinds] * np.sinh(E[hinds]) - E[hinds]) / n[hinds]

    # parabolic
    tp[pinds] = -(E[pinds] + E[pinds] ** 3 / 3) / n[pinds]

    return a, e, O, I, w, tp


def c2c3(psi: npt.ArrayLike) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Calculate the c2, c3 coefficients for the universal variable

    Args:
        psi (iterable or float):
            psi = chi^2/a for universal variable chi and semi-major axis a

    Returns:
        tuple:
            c2 (numpy.ndarray):
                c2 coefficients (same size as input)
            c3 (numpy.ndarray):
                c3 coefficients (same size as input)
    """

    # force input into 1D ndarray
    psi = np.array(psi, ndmin=1).astype(float).flatten()
    c2 = np.zeros(psi.size)
    c3 = np.zeros(psi.size)

    zeropsi = psi == 0
    pospsi = psi > 0
    negpsi = psi < 0

    c2[zeropsi] = 1 / 2
    c3[zeropsi] = 1 / 6

    c2[pospsi] = (1 - np.cos(np.sqrt(psi[pospsi]))) / psi[pospsi]
    c3[pospsi] = (np.sqrt(psi[pospsi]) - np.sin(np.sqrt(psi[pospsi]))) / np.sqrt(
        psi[pospsi] ** 3
    )

    c2[negpsi] = (1 - np.cosh(np.sqrt(-psi[negpsi]))) / psi[negpsi]
    c3[negpsi] = (np.sinh(np.sqrt(-psi[negpsi])) - np.sqrt(-psi[negpsi])) / np.sqrt(
        -(psi[negpsi] ** 3)
    )

    return c2, c3


def universalfg(
    r0: npt.NDArray[np.float_],
    v0: npt.NDArray[np.float_],
    mu: floatORarray,
    dt: floatORarray,
    maxIter: int = 100,
    return_counter: bool = False,
    convergence_error: bool = True,
) -> Tuple[npt.NDArray[np.float_], ...]:
    """Propagate orbital state vectors by delta t via universal variable-based f and g

    Args:
        r0 (numpy.ndarray):
            Components of orbital radius. 3n elements in 1D as
            [r1(1);r1(2);r1(3);r2(1);r2(2)r2(3);...;rn(1);rn(2);rn(3)]
            or in 2D as nx3 or 3xn
        v0 (numpy.ndarray):
            Components of orbital velocity. Same stacking as r
        mu (float or numpy.ndarray):
            Gravitational parameters.  If float, assuming all state vectors belong to
            the same system.
        dt (float or numpy.ndarray):
            Propagation time.  If float, assuming all states are propagated for the same
            time
        maxIter (int):
            Maximum numbr of iterations.  Optional, defaults to 100.
        return_counter (bool):
            If True, returns the number of iterations for each input state. Defaults
            False.
        convergence_error (bool):
            Raise error on convergence failure if True. Defaults True.

    Returns:
        tuple:
            r (numpy.ndarray):
                Components of orbital radius (n x 3)
            v (numpy.ndarray):
                Components of orbital velocity (n x 3)
            counter(numpy.ndarray):
                Number of required iterations (size n).  Only returned if return_counter
                is True

    .. note::

        r.flatten() and v.flatten() will automatically stack elements in the proper
        order in a 1D array

    """

    r0, v0, mu = validateOrbitalStateInputs(r0, v0, mu)
    dt = forcendarray(dt)
    assert dt.size == 1 or dt.size == len(
        r0
    ), "dt must be scalar or same size as r0 and v0"

    r0mag = np.sqrt(np.sum(r0 * r0, axis=1))  # orbital radius magnitude
    v02 = np.sum(v0 * v0, axis=1)  # velocity magnitude squared
    r0dotv0 = np.sum(r0 * v0, axis=1)  # r_0 \cdot v_0
    alpha = -v02 / mu + 2 / r0mag  # 1/a
    fac0 = r0dotv0 / np.sqrt(mu)  # r_0 \cdot v_0 / \sqrt{mu}

    # classify by orbit type
    epsval = 1000 * np.spacing(1)
    eorbs = alpha >= epsval
    porbs = np.abs(alpha) < epsval
    horbs = alpha <= -epsval

    # utility subfunction to grab desired mu and dt values
    def filtermudt(
        mu: npt.NDArray[np.float_],
        dt: npt.NDArray[np.float_],
        inds: npt.NDArray[np.bool_],
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        if mu.size == 1:
            fmu = mu
        else:
            fmu = mu[inds]

        if dt.size == 1:
            fdt = dt
        else:
            fdt = dt[inds]

        return fmu, fdt

    # Evaluate initial chi values
    chi = np.zeros(len(r0))
    if np.any(eorbs):
        emu, edt = filtermudt(mu, dt, eorbs)
        chi[eorbs] = np.sqrt(emu) * edt * alpha[eorbs]

    if np.any(porbs):
        alpha[porbs] = 0
        pmu, pdt = filtermudt(mu, dt, porbs)

        a1 = 1 / 6
        b1 = fac0[porbs] / 2
        c1 = r0mag[porbs]
        d1 = -np.sqrt(pmu) * pdt
        k0 = b1**2 - 3 * a1 * c1
        k1 = 2 * b1**3 - 9 * a1 * b1 * c1 + 27 * a1**2 * d1
        k2 = np.cbrt((k1 + np.sqrt(k1**2 - 4 * k0**3)) / 2)
        chi[porbs] = -(b1 + k2 + k0 / k2) / 3 / a1

    if np.any(horbs):
        hmu, hdt = filtermudt(mu, dt, horbs)
        chi[horbs] = (
            np.sign(hdt)
            * np.sqrt(-1.0 / alpha[horbs])
            * np.log(
                -2
                * hmu
                * alpha[horbs]
                * hdt
                / (
                    r0dotv0[horbs]
                    + np.sign(hdt)
                    * np.sqrt(-hmu / alpha[horbs])
                    * (1.0 - r0mag[horbs] * alpha[horbs])
                )
            )
        )

    # iteration setup
    counter = np.zeros(len(r0))
    r: npt.NDArray[np.float_] = r0mag.copy()  # type here to prevent ambiguity later
    chiup = np.ones(len(r0))
    # the tolerance is set by the current magnitudes of chi and r
    currtol = 10 * np.spacing(np.max(np.abs(np.vstack((chi, r))), axis=0))
    # we will only update things that haven't hit their tolerance:
    inds = np.abs(chiup) > currtol

    # iterate!
    while np.any(inds) and np.all(counter < maxIter):
        psi = chi[inds] ** 2.0 * alpha[inds]
        c2, c3 = c2c3(psi)
        r[inds] = (
            chi[inds] ** 2.0 * c2
            + fac0[inds] * chi[inds] * (1 - psi * c3)
            + r0mag[inds] * (1 - psi * c2)
        )

        nmu, ndt = filtermudt(mu, dt, inds)

        chiup[inds] = (
            np.sqrt(nmu) * ndt
            - chi[inds] ** 3.0 * c3
            - fac0[inds] * chi[inds] ** 2 * c2
            - r0mag[inds] * chi[inds] * (1 - psi * c3)
        ) / r[inds]
        chi[inds] += chiup[inds]

        currtol[inds] = 10 * np.spacing(
            np.max(np.abs(np.vstack((chi[inds], r[inds]))), axis=0)
        )
        currtol[currtol > 1] = 1  # prevent runaway
        counter[inds] += 1
        inds = np.abs(chiup) > currtol

    if np.any(counter == maxIter):
        if convergence_error:
            raise ValueError("Failed to converge on chi")
        else:
            warnings.warn("Failed to converge on chi")

    # Evaluate f and g functions
    psi = chi**2.0 * alpha
    c2, c3 = c2c3(psi)
    r = chi**2.0 * c2 + fac0 * chi * (1 - psi * c3) + r0mag * (1 - psi * c2)

    f = 1.0 - chi**2.0 / r0mag * c2
    g = dt - chi**3.0 / np.sqrt(mu) * c3
    fdot = np.sqrt(mu) / r / r0mag * chi * (psi * c3 - 1.0)
    gdot = 1.0 - chi**2.0 / r * c2

    # r = f*r0 + g*v0; v = fdot*r0 + gdot*v0
    r = np.diag(f).dot(r0) + np.diag(g).dot(v0)
    v: npt.NDArray[np.float_] = np.diag(fdot).dot(r0) + np.diag(gdot).dot(v0)

    out: Tuple[npt.NDArray[np.float_], ...] = (r, v)
    if return_counter:
        out += (counter,)

    return out


def calc_RV_from_M(
    M: npt.ArrayLike,
    e: npt.ArrayLike,
    w: npt.ArrayLike,
    K: npt.ArrayLike,
):
    """Calculate the combined radial velocity of a system of n objects at m epochs.

    Args:
        M (numpy.ndarray):
            Mean anomalies of the objects at desired epochs (n x m) (rad)
        e (numpy.ndarray):
            Eccentricities of the objects (n x 1)
        w (numpy.ndarray):
            Argument of periapsis of the objects (n x 1) (rad)
        K (numpy.ndarray):
            Semi-amplitudes of the objects (n x 1) (m/s)

    Returns:
        numpy.ndarray:
            System radial velocities at desired epochs

    """

    rv = np.zeros(M.shape[1])
    for nplanet in range(M.shape[0]):
        E, sinE, cosE = eccanom_orvara(M[nplanet, :], e[nplanet])

        # Get the object's rv added to the array
        rv = keplertools.CyRV.CyRV_from_E(
            rv, E, sinE, cosE, e[nplanet], w[nplanet], K[nplanet]
        )
    return rv


def RV_from_time_py(
    t: npt.ArrayLike,
    tp: npt.ArrayLike,
    per: npt.ArrayLike,
    e: npt.ArrayLike,
    w: npt.ArrayLike,
    K: npt.ArrayLike,
) -> npt.ArrayLike:
    """
    Calculate radial velocities using the standard method.

    Args:
        t (numpy.ndarray):
            Epoch times in jd (n,).
        tp (numpy.ndarray):
            Times of periastron passages for each object (m,).
        per (numpy.ndarray):
            Orbital periods for each object (m,).
        e (numpy.ndarray):
            Eccentricities for each object (m,).
        w (numpy.ndarray):
            Arguments of periapsis for each object (m,) in radians.
        K (numpy.ndarray):
            Semi-amplitudes for each object (m,) in m/s.

    Returns:
        rv (numpy.ndarray):
            Array of radial velocities at each epoch (n,).
    """
    # Initialize the radial velocity array
    rv = np.zeros_like(t)
    # Iterate over each planet in the system
    for j in range(tp.size):
        _tp, _per, _e, _w, _K = tp[j], per[j], e[j], w[j], K[j]

        # Calculate mean anomaly
        phase = (t - tp[j]) / per[j]
        M = 2.0 * np.pi * (phase - np.floor(phase))

        # Calculate eccentric anomaly and true anomaly
        E = eccanom(M, e[j])
        nu = trueanom(E, e[j])

        # Update radial velocities
        rv += K[j] * (e[j] * np.cos(w[j]) + np.cos(w[j] + nu))
    return rv


def calc_RV_from_time(
    t: npt.ArrayLike,
    tp: npt.ArrayLike,
    per: npt.ArrayLike,
    e: npt.ArrayLike,
    w: npt.ArrayLike,
    K: npt.ArrayLike,
    noc: bool = True,
) -> npt.ArrayLike:
    """Calculate the combined radial velocity of a system of n objects at m epochs.

    Args:
        t (numpy.ndarray):
            Epochs in jd (m x 1)
        tp (numpy.ndarray):
            Time of periastrons of the objects (n x 1)
        per (numpy.ndarray):
            Period
        e (numpy.ndarray):
            Eccentricities of the objects (n x 1)
        w (numpy.ndarray):
            Argument of periapsis of the objects (n x 1) (rad)
        K (numpy.ndarray):
            Semi-amplitudes of the objects (n x 1) (m/s)
        noc (bool):
            Use the Cython implementation if True, otherwise use the pure
            Python implementation.

    Returns:
        numpy.ndarray:
            System radial velocities at desired epochs

    """

    t = forcendarray(t)
    tp = forcendarray(tp)
    per = forcendarray(per)
    e = forcendarray(e)
    w = forcendarray(w)
    K = forcendarray(K)

    # Make sure all inputs are the same size
    size_match = tp.size == per.size == e.size == w.size == K.size
    if not size_match:
        raise ValueError("Inputs must be the same size")

    # Make sure there is at least one planet
    if tp.size == 0:
        raise ValueError("You must give at least one planet.")

    if noc:
        rv = np.zeros(len(t), dtype=np.double)
        rv = keplertools.CyRV.CyRV_from_time(rv, t, tp, per, e, w, K)
    else:
        rv = RV_from_time_py(t, tp, per, e, w, K)
    return rv
