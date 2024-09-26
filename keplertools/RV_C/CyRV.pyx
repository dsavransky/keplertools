import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.int64_t ITYPE_t

cdef extern from "RV_C.h":
    void meananom(double* M, double* t, double tp, double per, double twopi, int n)
    void RV_from_time(double* rv, double* t, double* tp, double* per, double* e, double* w, double* K, int n, int m)

cdef extern from "../eccanom_C/eccanom_C.h":
    void eccanom_orvara(double* E, double* sinE, double* cosE, double* M, double e, int n)

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def CyRV_from_E(np.ndarray[DTYPE_t, ndim=1] rv,
        np.ndarray[DTYPE_t, ndim=1] E,
        np.ndarray[DTYPE_t, ndim=1] sinE,
        np.ndarray[DTYPE_t, ndim=1] cosE,
        DTYPE_t e,
        DTYPE_t w,
        DTYPE_t K):
    """Finds radial velocity for a single object at the desired epochs, uses
    method from orvara that relies on eccentric anomaly and sine and cosine of
    the eccentric anomaly.

    Args:
        rv (ndarray):
            Preexisting radial velocities, can also be zeros (rad)
        E (ndarray):
            Eccentric anomaly (rad)
        sinE (ndarray):
            Sine of eccentric anomaly (rad)
        cosE (ndarray):
            Cosine of eccentric anomaly (rad)
        e (float):
            Eccentricity
        I (float):
            Inclination (rad)
        w (float):
            Argument of periapsis (rad)
        K (float):
            RV semi-amplitude (m/s)

    Returns:
        rv (ndarray):
            Radial velocities

    """
    cdef extern from "math.h" nogil:
        double sin(double _x)
        double cos(double _x)
        double fabs(double _x)
        double sqrt(double _x)

    cdef double pi = 3.14159265358979323846264338327950288
    cdef double twopi = 2*pi
    cdef double pi_d_2 = pi/2.

    cdef double sqrt1pe = sqrt(1 + e)
    cdef double sqrt1me = sqrt(1 - e)

    cdef double cosarg = cos(w)
    cdef double sinarg = sin(w)
    cdef double ecccosarg = e*cosarg
    cdef double sqrt1pe_div_sqrt1me = sqrt1pe/sqrt1me
    cdef double TA, ratio, fac, tanEAd2

    cdef int i

    ##################################################################
    # Trickery with trig identities.  The code below is mathematically
    # identical to the use of the true anomaly.  If sin(EA) is small
    # and cos(EA) is close to -1, no problem as long as sin(EA) is not
    # precisely zero (set tan(EA/2)=1e100 in this case).  If sin(EA)
    # is small and EA is close to zero, use the fifth-order Taylor
    # expansion for tangent.  This is good to ~1e-15 for EA within
    # ~0.015 of 0.  Assume eccentricity is not precisely unity (this
    # should be forbidden by the priors).  Very, very high
    # eccentricities (significantly above 0.9999) may be problematic.
    # This routine assumes range reduction of the eccentric anomaly to
    # (-pi, pi] and will throw an error if this is violated.
    ##################################################################

    cdef double one_d_24 = 1./24
    cdef double one_d_240 = 1./240

    for i, _ in enumerate(rv):
        _E = E[i]
        # Convert to correct range
        if _E > pi:
            _E = twopi - _E

        if fabs(sinE[i]) > 1.5e-2:
            tanEAd2 = (1 - cosE[i])/sinE[i]
        elif _E < -pi or _E > pi:
            raise ValueError("EA input to calc_RV must be betwen -pi and pi.")
        elif fabs(_E) < pi_d_2:
            tanEAd2 = _E*(0.5 + _E**2*(one_d_24 + one_d_240*_E**2))
        elif sinE[i] != 0:
            tanEAd2 = (1 - cosE[i])/sinE[i]
        else:
            tanEAd2 = 1e100

        ratio = sqrt1pe_div_sqrt1me*tanEAd2
        fac = 2/(1 + ratio**2)
        rv[i] += K*(cosarg*(fac - 1) - sinarg*ratio*fac + ecccosarg)

    return rv


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def CyRV_from_time(np.ndarray[DTYPE_t, ndim=1] rv,
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[DTYPE_t, ndim=1] tp,
        np.ndarray[DTYPE_t, ndim=1] per,
        np.ndarray[DTYPE_t, ndim=1] e,
        np.ndarray[DTYPE_t, ndim=1] w,
        np.ndarray[DTYPE_t, ndim=1] K):
    """Finds radial velocity for a single object at the desired epochs

    Args:
        rv (ndarray):
            Preexisting radial velocities, can also be zeros (rad)
        t (ndarray):
            Times of to calculate RV at (jd)
        tp (float):
            Time of periastron
        per (float):
            Period
        e (float):
            Eccentricity
        w (float):
            Argument of periapsis (rad)
        K (float):
            RV semi-amplitude (m/s)

    Returns:
        rv (ndarray):
            Radial velocities

    """

    cdef int n = t.size
    cdef int m = tp.size
    # Call the C function
    RV_from_time(<double*> rv.data, <double*> t.data, <double*> tp.data,
                 <double*> per.data, <double*> e.data, <double*> w.data,
                 <double*> K.data, n, m)
    return rv
