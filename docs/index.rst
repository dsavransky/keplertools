.. keplertools documentation master file, created by
   sphinx-quickstart on Mon Sep 26 11:47:31 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

keplertools
=======================================
``keplertools`` provides a variety of methods for the propagation of two-body orbits.


Installation
----------------

To install from PyPI: ::

    pip install keplertools


To install from source, clone or download the code repository, and, in the top level directory run: ::

    pip install .


Installing from source requires a C compiler.  See here for details: https://cython.readthedocs.io/en/latest/src/quickstart/install.html

Closed-Orbit Methods
----------------------
The following methods are for use with closed orbits:

* :py:meth:`~keplertools.fun.eccanom` - Finds eccentric anomaly from mean anomaly and eccentricity.  Can handle multiple different eccentricity values for different mean anomalies. 
* :py:meth:`~keplertools.fun.eccanom_orvara` - Finds eccentric anomaly E, sinE, cosE from mean anomaly and eccentricity using methodologies from the orvara code.  Can only handle one eccentricity value for all mean anomalies. 
* :py:meth:`~keplertools.fun.trueanom` - Finds true anomaly from eccentric anomaly and eccentricity.
* :py:meth:`~keplertools.fun.vec2orbElem` - Convert position and velocity vectors to Keplerian orbital elements using (corrected) algorithm from  [Vinti1998]_.
* :py:meth:`~keplertools.fun.vec2orbElem2` - Convert position and velocity vectors to Keplerian orbital elements using algorithm from [Vallado2013]_.
* :py:meth:`~keplertools.fun.orbElem2vec` - Convert Keplerian orbital elements to position and velocity vectors.

All-Orbit Methods
------------------
The following methods support all orbits (open and closed):

* :py:meth:`~keplertools.fun.invKepler` - Finds eccentric/hyperbolic/parabolic anomaly from mean anomaly and eccentricity. Can also return true anomaly.  Works for single or multiple eccentricity values. 
* :py:meth:`~keplertools.fun.kepler2orbstate` - Calculate orbital state vectors from Keplerian elements.
* :py:meth:`~keplertools.fun.orbstate2kepler`- Calculate  Keplerian elements given orbital state vectors.
* :py:meth:`~keplertools.fun.universalfg` - Propagate orbital state vectors by delta t via universal variable-based f and g. 
* :py:class:`~keplertools.keplerSTM.planSys` - Propagate orbital state vectors using (exact) Kepler state transition matrix. Implements algorithms from [Shepperd1985]_ and [Vallado2013]_. 

Radial Velocity Methods
-------------------------
* :py:meth:`~keplertools.fun.calc_RV_from_M` - Calculate the combined radial velocity of a system of n objects at m epochs given mean anomalies.
* :py:meth:`~keplertools.fun.calc_RV_from_time` - Calculate the combined radial velocity of a system of n objects at m epochs given times.

.. toctree::
   :maxdepth: 4
   :caption: Contents:


   modules
   refs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
