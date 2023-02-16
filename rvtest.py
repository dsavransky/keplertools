import astropy.constants as const
import astropy.units as u
import numpy as np

import keplertools.fun as fun

# Creating planetary system
a = np.array([1]) * u.AU
e = np.array([0])
O = np.array([0]) * u.rad
I = np.array([0]) * u.rad
w = np.array([0]) * u.rad
mp = np.array([1]) * u.M_earth
ms = 1 * u.M_sun
mu = np.array([(const.G * ms).decompose().value for _ in a]) * (u.m**3 / u.s**2)
