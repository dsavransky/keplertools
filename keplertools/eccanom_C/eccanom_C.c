#include <math.h>
#include <stdio.h>
#include <string.h>

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

int eccanom_C(double E[], double M[], double e[], double epsmult, int maxIter, int n){

    int j;
    double tmp;

    /*initialize E*/
    for (j = 0; j < n; j++){
        tmp = M[j]/(1 - e[j]);
        if (e[j]*pow(tmp,2) > 6*(1 - e[j])){
            E[j] = pow(6*M[j]/e[j],1.0/3.0);
        } else{
            E[j] = tmp;
        }
    }
    
    double tolerance = pow(2,log(1)/log(2) - 52.0)*epsmult;
    int numIter = 0;
    int maxnumIter = 0;
    double err;

    for (j = 0; j < n; j++){
        err = tolerance*2.0;
        numIter = 0;
        while ((err > tolerance) && (numIter < maxIter)) {
            E[j] = E[j] - (M[j] - E[j] + e[j]*sin(E[j]))/(e[j]*cos(E[j])-1);
            err = fabs(M[j] - (E[j] - e[j]*sin(E[j])));
            numIter += 1;
        }
        if(numIter > maxnumIter){maxnumIter = numIter;}
    }

    return maxnumIter;

}
