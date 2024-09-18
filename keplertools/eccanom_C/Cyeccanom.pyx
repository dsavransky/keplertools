import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.int64_t ITYPE_t

cdef extern from "eccanom_C.h":
    int eccanom_C(double* E, double* M, double* e, double epsmult, int maxIter, int n)
    void eccanom_orvara(double* E, double* sinE, double* cosE, double* M, double e, int n)

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def Cyeccanom(np.ndarray[DTYPE_t, ndim=1] M, np.ndarray[DTYPE_t, ndim=1] e, DTYPE_t epsmult, ITYPE_t maxIter):
    """Finds eccentric anomaly from mean anomaly and eccentricity

    This method uses Newton-Raphson iteration to find the eccentric
    anomaly from mean anomaly and eccentricity, assuming a closed (0<e<1)
    orbit.

    Args:
        M (ndarray):
            mean anomaly (rad)
        e (ndarray):
            eccentricity
        epsmult (float):
            Precision of convergence (multiplied by precision of floating data type).
        maxiter (int):
            Maximum numbr of iterations.

    Returns:
        E (float or ndarray):
            eccentric anomaly (rad)

    Notes:
        M and e must be the same size.
    """

    cdef int n = M.size

    assert (M.dtype == DTYPE) and (e.dtype == DTYPE) and (e.size == n), "Incompatible inputs."

    #initialize output array
    cdef np.ndarray[DTYPE_t, ndim=1] E =  np.zeros(n, dtype=DTYPE)

    cdef int numIter = eccanom_C(<double*> E.data, <double*> M.data, <double*> e.data, epsmult, maxIter, n)

    assert (numIter < maxIter), "eccanom_C failed to converge."

    return E

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def Cyeccanom_orvara(np.ndarray[DTYPE_t, ndim=1] M, DTYPE_t e):
    """Finds eccentric anomaly E, sinE, and cosE from mean anomaly and
    eccentricity

    This uses the method described the orvara paper which uses a 5th order
    polynomial to approximate E and does a single Newton-Raphson iteration to
    refine it.

    Args:
        M (numpy.ndarray):
            mean anomaly (rad)
        e (float):
            eccentricity

    Returns:
        tuple:
            E (numpy.ndarray):
                eccentric anomaly (rad)
            sinE (numpy.ndarray):
                Sine of eccentric anomaly
            cosE (numpy.ndarray):
                Cosine of eccentric anomaly

    .. note::

        Currently only works for a single orbit since it relies on creating a
        lookup table for a single orbit. A loop can be added.

    """

    cdef int n = M.size

    #initialize output array
    cdef np.ndarray[DTYPE_t, ndim=1] E =  np.zeros(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] sinE =  np.zeros(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] cosE =  np.zeros(n, dtype=DTYPE)

    eccanom_orvara(<double*> E.data,<double*> sinE.data, <double*> cosE.data, <double*> M.data, e, n)

    return E, sinE, cosE
