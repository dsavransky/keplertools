import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.int_t ITYPE_t

cdef extern from "eccanom_C.h":
    int eccanom_C(double* E, double* M, double* e, double epsmult, int maxIter, int n)

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
