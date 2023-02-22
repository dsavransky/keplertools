#include "../eccanom_C/eccanom_C.h"
#include <math.h>

void meananom(double M[], double t[], double tp, double per, double twopi,
              int n) {
  double phase;
  for (int i = n; i--;) {
    phase = (t[i] - tp) / per;
    M[i] = twopi * (phase - floorf(phase));
  }
  return;
}

void RV_from_time(double rv[], double t[], double tp[], double per[],
                  double e[], double w[], double K[], int n, int m) {
  /*Finds radial velocity for a single object at the desired epochs

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
  */

  double pi = 3.14159265358979323846264338327950288;
  double twopi = 2 * pi;
  double _tp, _per, _e, _w, _K;

  double M[n], E[n], sinE[n], cosE[n];
  double pi_d_2 = pi / 2.;
  double sqrt1pe, sqrt1me, cosarg, sinarg, ecccosarg, sqrt1pe_div_sqrt1me;
  double TA, ratio, fac, tanEAd2;

  double one_d_24 = 1. / 24;
  double one_d_240 = 1. / 240;

  double _E;
  for (int j = m; j--;) {
    _tp = tp[j];
    _per = per[j];
    _e = e[j];
    _w = w[j];
    _K = K[j];

    // Calculate mean anomaly
    meananom(M, t, _tp, _per, twopi, n);

    // #Calculating E, sinE, and cosE from M
    eccanom_orvara(E, sinE, cosE, M, _e, n);

    sqrt1pe = sqrt(1 + _e);
    sqrt1me = sqrt(1 - _e);

    cosarg = cos(_w);
    sinarg = sin(_w);
    ecccosarg = _e * cosarg;
    sqrt1pe_div_sqrt1me = sqrt1pe / sqrt1me;

    //     ##################################################################
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
    //     ##################################################################

    for (int i = n; i--;) {
      _E = E[i];
      if (_E > pi) {
        _E = twopi - _E;
      }
      if (fabs(sinE[i]) > 1.5e-2) {
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
      rv[i] += _K * (cosarg * (fac - 1) - sinarg * ratio * fac + ecccosarg);
    }
  }
  return;
}
