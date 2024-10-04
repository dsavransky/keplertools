#include "../eccanom_C/eccanom_C.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define pi 3.14159265358979323846264338327950288

const double twopi = 2 * pi;
const double pi_d_2 = pi / 2.;
const double one_d_24 = 1. / 24;
const double one_d_240 = 1. / 240;

/*=================================================================
 * meananom_C     Calculate the mean anomaly for a set of times, ensuring
 *                time units match. This function computes the mean anomaly
 *                based on the given times, time of periastron, and period.
 *
 * meananom_C(M, t, tp, per, n)
 *   M       nx1     Output array - Mean anomalies (rad)
 *   t       nx1     Times to calculate mean anomaly at
 *   tp      1x1     Time of periastron
 *   per     1x1     Period
 *   n       1x1     Number of epochs
 *=================================================================*/

void meananom(double M[], double t[], double tp, double per, int n) {
  const double one_over_per = 1 / per;

  double phase;
  for (int i = 0; i < n; i++) {
    phase = (t[i] - tp) * one_over_per;
    M[i] = twopi * (phase - floor(phase));
  }
  return;
}

/*=================================================================
 * RV_from_time     Calculates the radial velocity of a star based on
 *                  a set of planets using the method (and code) from orvara.
 *
 * RV_from_time(rv, t, tp, per, e, w, K, n, m)
 *   rv       nx1     Output array for radial velocities, can also be zeros
 *                    (m/s)
 *   t        nx1     Times to calculate RV at (jd)
 *   tp       mx1     Time of periastron for each planet
 *   per      mx1     Periods for each planet
 *   e        mx1     Eccentricities for each planet
 *   w        mx1     Arguments of periapsis for each planet (rad)
 *   K        mx1     RV semi-amplitude for each planet (m/s)
 *   n        1x1     Number of epochs to calculate RV at
 *   m        1x1     Number of planets
 *=================================================================*/
void RV_from_time(double rv[], double t[], double tp[], double per[],
                  double e[], double w[], double K[], int n, int m) {

  double _tp, _per, _e, _w, _K;

  int arrSize = n * sizeof(double);
  double *M, *E, *sinE, *cosE;
  M = (double *)malloc(arrSize);
  memset(M, 0, arrSize);
  E = (double *)malloc(arrSize);
  memset(E, 0, arrSize);
  sinE = (double *)malloc(arrSize);
  memset(sinE, 0, arrSize);
  cosE = (double *)malloc(arrSize);
  memset(cosE, 0, arrSize);

  double sqrt1pe, sqrt1me, cosarg, sinarg, ecccosarg, sqrt1pe_div_sqrt1me;
  double TA, ratio, fac, tanEAd2;

  double _E;
  for (int j = m; j--;) {
    _tp = tp[j];
    _per = per[j];
    _e = e[j];
    _w = w[j];
    _K = K[j];

    // Calculate mean anomaly
    meananom(M, t, _tp, _per, n);

    // #Calculating E, sinE, and cosE from M
    eccanom_orvara(E, sinE, cosE, M, _e, n);

    sqrt1pe = sqrt(1. + _e);
    sqrt1me = sqrt(1. - _e);

    cosarg = cos(_w);
    sinarg = sin(_w);
    ecccosarg = _e * cosarg;
    sqrt1pe_div_sqrt1me = sqrt1pe / sqrt1me;

    // ##################################################################
    // #Trickery with trig identities.The code below is mathematically
    // #identical to the use of the true anomaly.If sin(EA) is small
    // #and cos(EA) is close to - 1, no problem as long as sin(EA) is not
    // #precisely zero(set tan(EA / 2) = 1e100 in this case).If sin(EA)
    // #is small and EA is close to zero, use the fifth - order Taylor
    // #expansion for tangent.This is good to ~1e-15 for EA within
    // #~0.015 of 0. Assume eccentricity is not precisely unity(this
    // #should be forbidden by the priors).Very, very high
    // #eccentricities(significantly above 0.9999) may be problematic.
    // #This routine assumes range reduction of the eccentric anomaly to
    // #(- pi, pi] and will throw an error if this is violated.
    // ##################################################################

    for (int i = n; i--;) {
      _E = E[i];
      if (_E > pi) {
        _E = twopi - _E;
      }
      if (fabs(sinE[i]) > 1.5e-6) {
        tanEAd2 = (1 - cosE[i]) / sinE[i];
      } else if (fabs(_E) < pi_d_2) {
        tanEAd2 = _E * (0.5 + _E * _E * (one_d_24 + one_d_240 * _E * _E));
      } else if (sinE[i] != 0) {
        tanEAd2 = (1 - cosE[i]) / sinE[i];
      } else {
        tanEAd2 = 1e100;
      }
      ratio = sqrt1pe_div_sqrt1me * tanEAd2;
      fac = 2 / (1 + ratio * ratio);
      rv[i] += _K * (cosarg * (fac - 1.) - sinarg * ratio * fac + ecccosarg);
    }
  }

  // cleanup
  free(M);
  free(E);
  free(sinE);
  free(cosE);

  return;
}
