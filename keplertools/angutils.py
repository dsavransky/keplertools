import numpy as np
import numpy.typing as npt

np.float_ = np.float64  # for numpy 2 compatibility


def rotMat(axis: int, angle: float) -> npt.NDArray[np.float_]:
    """Returns the DCM ({}^B C^A) for a frame rotation of angle about
    the specified axis

    Args:
        axis (int):
            Body axis to rotate about (1, 2, or 3 only)
        angle (float):
            Angle of rotation

    Returns:
        numpy.ndarray:
            3x3 rotation matrix

    """

    assert axis in [1, 2, 3], "Axis must be one of 1, 2, or 3, only."

    if axis == 1:
        return np.array(
            (
                [1, 0, 0],
                [0, np.cos(angle), np.sin(angle)],
                [0, -np.sin(angle), np.cos(angle)],
            )
        )
    elif axis == 2:
        return np.array(
            (
                [np.cos(angle), 0, -np.sin(angle)],
                [0, 1, 0],
                [np.sin(angle), 0, np.cos(angle)],
            )
        )
    elif axis == 3:
        return np.array(
            (
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            )
        )


def skew(v: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Given 3x1 vector v, return skew-symmetric matrix

    Args:
        v (iterable):
            Component representation of vector.  Must have 3 elements


    Returns:
        numpy.ndarray:
            3x3 skew-symmetric matrix

    """

    assert hasattr(v, "__iter__") and len(v) == 3, "v must be an iterable of length 3."

    if isinstance(v, np.ndarray):
        v = v.flatten()

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def colVec(n: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Turn any 3-element iterable into a 3x1 column vector

    Args:
        n (iterable):
            3 element iterable

    Returns:
        numpy.ndarray:
            3x1 component representation of the vector
    """

    assert np.size(n) == 3
    n = np.array(n, ndmin=2)
    if len(n) == 1:
        n = n.T

    return n


def calcDCM(n: npt.ArrayLike, th: float) -> npt.NDArray[np.float_]:
    """Rodrigues formula: Calculates the DCM ({}^A C^B) for a rotation of angle th about
    an axis n

    Args:
        n (iterable):
            3 element vector representing rotation axis
        th (float):
            Angle of rotation

    Returns:
        numpy.ndarray:
            3x3 rotation matrix

    """

    n = vnorm(colVec(n))

    DCM = np.eye(3) * np.cos(th) + (1 - np.cos(th)) * n * n.T + skew(n) * np.sin(th)

    return DCM


def vnorm(v: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Return components of unit vector of input vector

    Args:
        v (numpy.ndarray):
            Components of vector

    Return
        numpy.ndarray:
            Components of unit vector
    """

    return v / np.linalg.norm(v)


def calcang(x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike) -> float:
    """Compute the angle between vectors x and y when rotating counter-clockwise about
    vector z

    Args:
        x (iterable):
            3 components of x vector
        y (iterable):
            3 components of y vector
        z (iterable):
            3 components of z vector

    Returns:
        float:
            Angle in radians
    """

    x = vnorm(colVec(x))
    y = vnorm(colVec(y))
    z = vnorm(colVec(z))

    t1 = np.linalg.norm(np.matmul(skew(x), y)) * np.sign(
        np.linalg.det(np.hstack((x, y, z)))
    )
    t2 = np.matmul(x.T, y)[0][0]

    return np.arctan2(t1, t2)


def projplane(v: npt.ArrayLike, nv: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Project vectors v onto a plane normal to nv

    Args:
        v (numpy.ndarray):
            3xn vectors to be projected
        nv (numpy.ndarray):
            3x1 or 1x3 components of vector orthogonal to plane of projection

    Returns:
        numpy.ndarray:
            Output has equivalent size to v and contains the projected vectors

    """
    nv = vnorm(nv.flatten())

    projv = v - np.vstack([np.dot(x, nv.flatten()) * nv for x in v.T]).T

    return projv
