
/*=================================================================
* eccanom_C     Invert Kepler's time equation for elliptical orbits
*               using Newton-Raphson iteration.
* 
* eccanom_C(E, M, e, epsmult, maxIter, n)
*   E       nx1     Output array - Eccentric anomalies (rad)
*   M       nx1     Mean anomalies (rad)
*   e       nx1     eccentricities
*   epsmul  1x1     Floating type precision multiplier
*   maxIter 1x1     Maximum number of allowed iterations
*   n       1x1     Lengths of E,M,e
*=================================================================*/


int eccanom_C(double E[], double M[], double e[], double epsmult, int maxIter, int n);
