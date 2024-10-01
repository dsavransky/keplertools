#include <math.h>
#include <stdio.h>
#include <string.h>

#define pi 3.14159265358979323846264338327950288

static const double one_sixth = 1. / 6;
static const double if3 = 1. / 6;
static const double if5 = 1. / (6. * 20);
static const double if7 = 1. / (6. * 20 * 42);
static const double if9 = 1. / (6. * 20 * 42 * 72);
static const double if11 = 1. / (6. * 20 * 42 * 72 * 110);
static const double if13 = 1. / (6. * 20 * 42 * 72 * 110 * 156);
static const double if15 = 1. / (6. * 20 * 42 * 72 * 110 * 156 * 210);
static const double two_pi = 2. * pi;
static const double pi_d_12 = pi / 12;
static const double pi_d_6 = pi / 6;
static const double pi_d_4 = pi / 4;
static const double pi_d_3 = pi / 3;
static const double fivepi_d_12 = pi * 5. / 12;
static const double pi_d_2 = pi / 2;
static const double sevenpi_d_12 = pi * 7. / 12;
static const double twopi_d_3 = pi * 2. / 3;
static const double threepi_d_4 = pi * 3. / 4;
static const double fivepi_d_6 = pi * 5. / 6;
static const double elevenpi_d_12 = pi * 11. / 12;

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

int eccanom_C(double E[], double M[], double e[], double epsmult, int maxIter,
              int n) {

  int j;
  double tmp;

  /*initialize E*/
  for (j = 0; j < n; j++) {
    tmp = M[j] / (1 - e[j]);
    if (e[j] * pow(tmp, 2) > 6 * (1 - e[j])) {
      E[j] = pow(6 * M[j] / e[j], 1.0 / 3.0);
    } else {
      E[j] = tmp;
    }
  }

  double tolerance = pow(2, log(1) / log(2) - 52.0) * epsmult;
  int numIter = 0;
  int maxnumIter = 0;
  double err;

  for (j = 0; j < n; j++) {
    err = tolerance * 2.0;
    numIter = 0;
    while ((err > tolerance) && (numIter < maxIter)) {
      E[j] = E[j] - (M[j] - E[j] + e[j] * sin(E[j])) / (e[j] * cos(E[j]) - 1);
      err = fabs(M[j] - (E[j] - e[j] * sin(E[j])));
      numIter += 1;
    }
    if (numIter > maxnumIter) {
      maxnumIter = numIter;
    }
  }

  return maxnumIter;
}

/*=================================================================
 * getbounds    Creates the lookup table for the bounds of the eccentric anomaly
 *              and the coefficients for the Taylor series expansion of the
 *              eccentric anomaly, based on a given eccentricity.
 *              of calculating positions and velocities in elliptical orbits.
 *
 * getbounds(bounds, E_tab, e)
 *   bounds    13x1    Output array containing the bounds for the eccentric
 *                     anomaly intervals.
 *   E_tab     78x1    Output array containing the lookup table of polynomial
 *                     coefficients used to calculate the eccentric anomaly
 *                     within the 13 different intervals defined by bounds.
 *   e         1x1     Eccentricity of the orbit.
 *=================================================================*/

void getbounds(double bounds[], double E_tab[], double e) {
  // Taken from https://github.com/t-brandt/orvara
  // Creates the lookup table for the bounds of the eccentric anomaly
  // and the coefficients for the Taylor series expansion of the
  // eccentric anomaly.

  const double g2s_e = 0.2588190451025207623489 * e;
  const double g3s_e = 0.5 * e;
  const double g4s_e = 0.7071067811865475244008 * e;
  const double g5s_e = 0.8660254037844386467637 * e;
  const double g6s_e = 0.9659258262890682867497 * e;
  const double g2c_e = g6s_e;
  const double g3c_e = g5s_e;
  const double g4c_e = g4s_e;
  const double g5c_e = g3s_e;
  const double g6c_e = g2s_e;

  bounds[0] = 0;
  bounds[1] = pi_d_12 - g2s_e;
  bounds[2] = pi_d_6 - g3s_e;
  bounds[3] = pi_d_4 - g4s_e;
  bounds[4] = pi_d_3 - g5s_e;
  bounds[5] = fivepi_d_12 - g6s_e;
  bounds[6] = pi_d_2 - e;
  bounds[7] = sevenpi_d_12 - g6s_e;
  bounds[8] = twopi_d_3 - g5s_e;
  bounds[9] = threepi_d_4 - g4s_e;
  bounds[10] = fivepi_d_6 - g3s_e;
  bounds[11] = elevenpi_d_12 - g2s_e;
  bounds[12] = pi;

  double x;

  E_tab[1] = 1 / (1. - e);
  E_tab[2] = 0;

  x = 1. / (1 - g2c_e);
  E_tab[7] = x;
  E_tab[8] = -0.5 * g2s_e * x * x * x;
  x = 1. / (1 - g3c_e);
  E_tab[13] = x;
  E_tab[14] = -0.5 * g3s_e * x * x * x;
  x = 1. / (1 - g4c_e);
  E_tab[19] = x;
  E_tab[20] = -0.5 * g4s_e * x * x * x;
  x = 1. / (1 - g5c_e);
  E_tab[25] = x;
  E_tab[26] = -0.5 * g5s_e * x * x * x;
  x = 1. / (1 - g6c_e);
  E_tab[31] = x;
  E_tab[32] = -0.5 * g6s_e * x * x * x;

  E_tab[37] = 1;
  E_tab[38] = -0.5 * e;

  x = 1. / (1 + g6c_e);
  E_tab[43] = x;
  E_tab[44] = -0.5 * g6s_e * x * x * x;
  x = 1. / (1 + g5c_e);
  E_tab[49] = x;
  E_tab[50] = -0.5 * g5s_e * x * x * x;
  x = 1. / (1 + g4c_e);
  E_tab[55] = x;
  E_tab[56] = -0.5 * g4s_e * x * x * x;
  x = 1. / (1 + g3c_e);
  E_tab[61] = x;
  E_tab[62] = -0.5 * g3s_e * x * x * x;
  x = 1. / (1 + g2c_e);
  E_tab[67] = x;
  E_tab[68] = -0.5 * g2s_e * x * x * x;

  E_tab[73] = 1. / (1 + e);
  E_tab[74] = 0;

  double B0, B1, B2, idx;
  int i, k;
  for (i = 0; i < 12; i++) {
    // For each interval, calculate the coefficients that have not already been
    // calculated and add to the lookup table E_tab
    idx = 1. / (bounds[i + 1] - bounds[i]);
    k = 6 * i;
    E_tab[k] = i * pi_d_12;

    B0 = idx * (-E_tab[k + 2] - idx * (E_tab[k + 1] - idx * pi_d_12));
    B1 = idx * (-2 * E_tab[k + 2] - idx * (E_tab[k + 1] - E_tab[k + 7]));
    B2 = idx * (E_tab[k + 8] - E_tab[k + 2]);

    E_tab[k + 3] = B2 - 4 * B1 + 10 * B0;
    E_tab[k + 4] = (-2 * B2 + 7 * B1 - 15 * B0) * idx;
    E_tab[k + 5] = (B2 - 3 * B1 + 6 * B0) * idx * idx;
  }

  return;
}

inline double shortsin(double x) {
  // Taken from https://github.com/t-brandt/orvara
  double x2 = x * x;
  return x *
         (1 - x2 * (if3 -
                    x2 * (if5 -
                          x2 * (if7 -
                                x2 * (if9 - x2 * (if11 -
                                                  x2 * (if13 - x2 * if15)))))));
}

/*=================================================================
 * Estart    Calculates the initial guess for the eccentric anomaly
 *           based on the mean anomaly and the eccentricity of the orbit.
 *           Code taken from https://github.com/t-brandt/orvara.
 *
 * Estart(M, e)
 *   M       1x1     Mean anomaly (rad)
 *   e       1x1     Eccentricity of the orbit
 *
 *   Returns the initial estimate of the eccentric anomaly (rad).
 *=================================================================*/
inline double Estart(double M, double e) {

  const double ome = 1. - e;
  const double sqrt_ome = sqrt(ome);
  const double chi = M / (sqrt_ome * ome);
  const double Lam = sqrt(8 + 9 * chi * chi);
  const double S = cbrt(Lam + 3 * chi);
  const double S_squared = S * S;
  const double sigma = 6 * chi / (2 + S_squared + 4. / (S_squared));
  const double s2 = sigma * sigma;
  const double denom = s2 + 2;
  const double E =
      sigma *
      (1 + s2 * ome *
               ((s2 + 20) / (60. * denom) +
                s2 * ome * (s2 * s2 * s2 + 25 * s2 * s2 + 340 * s2 + 840) /
                    (1400 * denom * denom * denom)));
  return E * sqrt_ome;
}

/*=================================================================
 * eccanom_orvara   Invert Kepler's time equation for elliptical orbits
 *                  using orvara's method.
 *
 * eccanom_orvara(E, sinE, cosE, M, e, n)
 *   E       nx1     Output array - Eccentric anomalies (rad)
 *   sinE    nx1     Output array - Sine of eccentric anomalies (rad)
 *   cosE    nx1     Output array - Cosine of eccentric anomalies (rad)
 *   M       nx1     Mean anomalies (rad)
 *   e       1       Eccentricity
 *   n       1       Lengths of E, sinE, cosE, M
 *=================================================================*/
void eccanom_orvara(double E[], double sinE[], double cosE[], double M[],
                    double e, int n) {
  double E_tab[6 * 13];
  double bounds[13];
  getbounds(bounds, E_tab, e);
  int i, j, k;
  double dx;

  const double one_over_ecc = 1.0 / fmax(1e-17, e);
  double _M, _E, _sinE, _cosE, dE, dEsq_d6;
  int Esign;
  double num, denom;

  if (e < 0.78) {
    for (i = 0; i < n; i++) {
      _M = M[i];

      // Cut mean anomaly between 0 and pi to use shorter Taylor series
      if (_M > pi) {
        Esign = -1;
        _M = two_pi - _M;
      } else {
        Esign = 1;
      }

      // Find the relevant interval, searching backwards
      for (j = 11;; j--) {
        if (_M >= bounds[j - 1]) {
          break;
        }
      }
      k = 6 * j;
      dx = _M - bounds[j];

      // Initial guess from lookup table
      _E = E_tab[k] +
           dx * (E_tab[k + 1] +
                 dx * (E_tab[k + 2] +
                       dx * (E_tab[k + 3] +
                             dx * (E_tab[k + 4] + dx * E_tab[k + 5]))));

      // Calculate _sinE and _cosE using the short sin function and sqrt call
      if (!(_E > pi_d_4)) {
        _sinE = shortsin(_E);
        _cosE = sqrt(1. - _sinE * _sinE);
      } else if (_E < threepi_d_4) {
        _cosE = shortsin(pi_d_2 - _E);
        _sinE = sqrt(1 - _cosE * _cosE);
      } else {
        _sinE = shortsin(pi - _E);
        _cosE = -sqrt(1 - _sinE * _sinE);
      }

      num = (_M - _E) * one_over_ecc + _sinE;
      denom = one_over_ecc - _cosE;

      // Get the second order approximation of dE
      dE = num * denom / (denom * denom + 0.5 * _sinE * num);

      // Apply correction to E, sinE, and _cosE with second order approximation
      E[i] = fmod(Esign * (_E + dE) + two_pi, two_pi);
      sinE[i] = Esign * (_sinE * (1 - 0.5 * dE * dE) + dE * _cosE);
      cosE[i] = _cosE * (1 - 0.5 * dE * dE) - dE * _sinE;
    }
  }
  // For higher eccentricities we need to go to third order
  else {
    for (i = 0; i < n; i++) {
      _M = M[i];
      // Cut mean anomaly between 0 and pi to use shorter Taylor series
      if (_M > pi) {
        Esign = -1;
        _M = two_pi - _M;
      } else {
        Esign = 1;
      }
      if ((2 * _M + (1 - e)) > 0.2) {
        for (j = 11;; j--) {
          if (_M >= bounds[j - 1]) {
            break;
          }
        }
        k = 6 * j;
        dx = _M - bounds[j];

        // Initial guess from lookup table
        _E = E_tab[k] +
             dx * (E_tab[k + 1] +
                   dx * (E_tab[k + 2] +
                         dx * (E_tab[k + 3] +
                               dx * (E_tab[k + 4] + dx * E_tab[k + 5]))));

      } else {
        _E = Estart(_M, e);
      }

      // Calculate _sinE and _cosE using the short sin function and sqrt call
      if (!(_E > pi_d_4)) {
        _sinE = shortsin(_E);
        _cosE = sqrt(1. - _sinE * _sinE);
      } else if (_E < threepi_d_4) {
        _cosE = shortsin(pi_d_2 - _E);
        _sinE = sqrt(1 - _cosE * _cosE);
      } else {
        _sinE = shortsin(pi - _E);
        _cosE = -sqrt(1 - _sinE * _sinE);
      }

      num = (_M - _E) * one_over_ecc + _sinE;
      denom = one_over_ecc - _cosE;

      if (_M > 0.4) {
        dE = num * denom / (denom * denom + 0.5 * _sinE * num);
      } else {
        dE = num * (denom * denom + 0.5 * num * _sinE);
        dE /= denom * denom * denom +
              num * (denom * _sinE + one_sixth * num * _cosE);
      }
      dEsq_d6 = dE * dE * one_sixth;

      E[i] = fmod(Esign * (_E + dE) + two_pi, two_pi);
      sinE[i] =
          Esign * (_sinE * (1 - 3 * dEsq_d6) + dE * (1 - dEsq_d6) * _cosE);
      cosE[i] = _cosE * (1 - 3 * dEsq_d6) - dE * (1 - dEsq_d6) * _sinE;
    }
  }
}
