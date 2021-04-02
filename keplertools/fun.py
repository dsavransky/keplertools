import numpy as np

try:
    import keplertools.Cyeccanom

    haveCyeccanom = True
except ImportError:
    haveCyeccanom = False
    pass


def eccanom(M, e, epsmult=4.01, maxIter=100, returnIter=False, noc=False):
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
        maxiter (int):
            Maximum numbr of iterations.  Optional, defaults to 100.
        returnIter (bool):
            Return number of iterations (defaults false, only available in python version)
        noc (bool):
            Don't use C version even if it can be loaded.

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

    noc = noc and haveCyeccanom

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

        # initial values for E
        E = M / (1 - e)
        mask = e * E ** 2 > 6 * (1 - e)
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
            raise Exception("eccanom failed to converge. Final error of %e" % err)
    else:
        E = keplertools.Cyeccanom.Cyeccanom(M, e, epsmult, maxIter)
        returnIter = False

    if returnIter:
        return E, numIter
    else:
        return E


def trueanom(E, e):
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

    return nu


def vec2orbElem2(rs, vs, mus):
    """Convert position and velocity vectors to Keplerian orbital elements

    Implements the algorithm from Vallado

    Args:
        rs (ndarray):
            3n x 1 stacked initial position vectors:
              [r1(1);r1(2);r1(3);r2(1);r2(2)r2(3);...;rn(1);rn(2);rn(3)]
            or 3 x n or n x 3 matrix of position vecotrs.
        vs (ndarray):
            3n x 1 stacked initial velocity vectors or 3 x n or n x3 matrix
        mus (ndarray or float)
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

    nplanets = rs.size / 3.0
    assert np.isscalar(mus) or mus.size == nplanets, "mus must be scalar or of size n"

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

    v2s = np.sum(vs ** 2.0, axis=0)  # orbital velocity squared
    rmag = np.sqrt(np.sum(rs ** 2.0, axis=0))  # orbital radius

    hvec = np.vstack(
        (
            rs[1] * vs[2] - rs[2] * vs[1],
            rs[2] * vs[0] - rs[0] * vs[2],
            rs[0] * vs[1] - rs[1] * vs[0],
        )
    )  # angular momentum vector
    nvec = np.vstack(
        (-hvec[1], hvec[0], np.zeros(len(hvec[2])))
    )  # node-pointing vector
    evec = (
        np.tile((v2s - mus / rmag) / mus, (3, 1)) * rs
        - np.tile(np.sum(rs * vs, axis=0) / mus, (3, 1)) * vs
    )  # eccentricity vector
    nmag = np.sqrt(np.sum(nvec ** 2.0, axis=0))
    e = np.sqrt(np.sum(evec ** 2.0, axis=0))

    En = v2s / 2 - mus / rmag
    a = -mus / 2 / En
    ell = a * (1 - e ** 2)
    if np.any(e == 1):
        tmp = np.sum(hvec ** 2.0, axis=0) / mus
        ell[e == 1] = tmp[e == 1]

    # angles
    I = np.arccos(hvec[2] / np.sqrt(np.sum(hvec ** 2.0, axis=0)))
    O = np.arccos(nvec[0] / nmag)
    O[nvec[2] < 0] = 2 * np.pi - O[nvec[2] < 0]
    w = np.arccos(np.sum(nvec * evec, axis=0) / e / nmag)
    w[evec[2] < 0] = 2 * np.pi - w[evec[2] < 0]

    # ecentric anomaly
    cosE = (1.0 - rmag / a) / e
    sinE = np.sum(rs * vs, axis=0) / (e * np.sqrt(mus * a))
    E = np.mod(np.arctan2(sinE, cosE), 2 * np.pi)

    # orbital periods
    P = 2 * np.pi * np.sqrt(a ** 3.0 / mus)

    # time of periapsis crossing
    tau = -(E - e * np.sin(E)) / np.sqrt(mus * a ** -3.0)

    return a, e, E, O, I, w, P, tau


def vec2orbElem(rs, vs, mus):
    """Convert position and velocity vectors to Keplerian orbital elements

    Implements the (corrected) algorithm from Vinti

    Args:
        rs (ndarray):
            3n x 1 stacked initial position vectors:
              [r1(1);r1(2);r1(3);r2(1);r2(2)r2(3);...;rn(1);rn(2);rn(3)]
            or 3 x n or n x 3 matrix of position vecotrs.
        vs (ndarray):
            3n x 1 stacked initial velocity vectors or 3 x n or n x3 matrix
        mus (ndarray or float)
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

    nplanets = rs.size / 3.0
    assert np.isscalar(mus) or mus.size == nplanets, "mus must be scalar or of size n"

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

    v2s = np.sum(vs ** 2.0, axis=0)  # orbital velocity squared
    r = np.sqrt(np.sum(rs ** 2.0, axis=0))  # orbital radius
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
    L2s = np.sum(L ** 2.0, axis=0)  # angular momentum squared
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
    P = 2 * np.pi * np.sqrt(a ** 3.0 / mus)

    # time of periapsis crossing
    tau = -(E - e * np.sin(E)) / np.sqrt(mus * a ** -3.0)

    return a, e, E, O, I, w, P, tau


def calcAB(a, e, O, I, w):
    """Calculate inertial frame components of perifocal frame unit vectors scaled
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
                Components of q vector (orthogonal to e and h) scaled by b (=a\sqrt{1-e^2})

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
            * np.sqrt(1 - e ** 2)
            * (np.cos(O) * np.sin(w) + np.sin(O) * np.cos(I) * np.cos(w)),
            a
            * np.sqrt(1 - e ** 2)
            * (-np.sin(O) * np.sin(w) + np.cos(O) * np.cos(I) * np.cos(w)),
            a * np.sqrt(1 - e ** 2) * np.sin(I) * np.cos(w),
        )
    )

    return A, B


def orbElem2vec(E, mus, orbElem=None, AB=None, returnAB=False):
    """Convert Keplerian orbital elements to position and velocity vectors

    Args:
        E (ndarray)
            nx1 array of eccentric anomalies (rad)
        mus (ndarray or float)
            nx1 array of gravitational parameters (G*m_i) where G is the
            gravitational constant and m_i is the mass of the ith body.
            if all vectors represent the same body, mus may be a scalar.
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
            mus.size == E.size
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
        ) * np.tile(np.sqrt(mus * a ** (-3.0)) / (1 - e * np.cos(E)), (3, 1))

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
