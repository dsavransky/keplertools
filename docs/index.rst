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


To also compile the Cython versions (compiler required, for details see: https://cython.readthedocs.io/en/latest/src/quickstart/install.html): ::

    pip install --no-binary keplertools keplertools[C]


If using a zsh shell (or depending on your specific shell setup), you may need to escape the square brackets (i.e., the last bit of the previous command would be ``keplertools\[C\]``.

Closed-Orbit Methods
----------------------
The following methods are for use with closed orbits:

* :py:meth:`~keplertools.fun.eccanom`
* :py:meth:`~keplertools.fun.trueanom`
* :py:meth:`~keplertools.fun.vec2orbElem`
* :py:meth:`~keplertools.fun.vec2orbElem2`
* :py:meth:`~keplertools.fun.orbElem2vec`

All-Orbit Methods
------------------
The following methods support all orbits (open and closed):

* :py:meth:`~keplertools.fun.invKepler`
* :py:meth:`~keplertools.fun.kepler2orbstate`
* :py:meth:`~keplertools.fun.orbstate2kepler`
* :py:meth:`~keplertools.fun.universalfg`
* :py:class:`~keplertools.keplerSTM.planSys`

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
