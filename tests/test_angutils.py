import unittest
import numpy as np
from keplertools.angutils import (
    calcDCM,
    rotMat,
    skew,
    colVec,
    vnorm,
    calcang,
    projplane,
)


class TestAngUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calcDCM(self):
        """Test for expected outputs for the three simple DCMs about body axes"""

        ns = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        th = 0.5
        for j, n in enumerate(ns):
            self.assertTrue(
                np.max(np.abs(calcDCM(n, th).T - rotMat(j + 1, th))) < 1e-15
            )

    def test_skew(self):
        """Test skew-symmetric property for random inputs"""

        tmp = skew(np.random.rand(3, 1))

        should_be_zeros = tmp + tmp.T

        self.assertTrue(np.all(should_be_zeros == 0))

    def test_colVec(self):
        """Test output dimensionality for random input"""

        self.assertTrue(colVec(np.random.rand(1, 3)).shape == (3, 1))
        self.assertTrue(colVec(np.random.rand(3, 1)).shape == (3, 1))
        self.assertTrue(colVec(np.random.rand(3).tolist()).shape == (3, 1))

    def test_vnorm(self):
        """Test norm output for random inputs"""

        self.assertTrue(np.abs(np.linalg.norm(vnorm(np.random.rand(1, 3))) - 1) < 1e-15)

    def test_calcang(self):
        """Test computed angles for random inputs"""

        x = vnorm(colVec([1, 0, 0]))
        z = vnorm(colVec([0, 0, 1]))
        th = np.pi / 6
        y = np.matmul(calcDCM(z, th), x)

        self.assertTrue(np.abs(calcang(x, y, z) - th) < 1e-15)

        for j in range(int(1e4)):
            x = vnorm(colVec(np.random.randn(3)))
            z = np.random.randn(2)
            z = vnorm(colVec(np.hstack([z, -np.matmul(z, x[:2]) / x[2]])))
            self.assertTrue(np.abs(np.matmul(x.T, z)) < 1e-15)
            th = (np.random.rand(1) * 2 - 1) * np.pi
            y = np.matmul(calcDCM(z, th), x)

            self.assertTrue(np.abs(calcang(x, y, z) - th) < 1e-15)

    def test_projplane(self):
        """Check dot product of output vector with orthogonal direction (should be
        zero)"""

        v = np.random.rand(3, 100)
        nv = np.random.rand(3, 1)
        tmp = projplane(v, nv)
        self.assertTrue(
            np.all(
                np.abs(np.vstack([np.dot(x, nv.flatten()) * nv for x in tmp.T]).T)
                < 1e-15
            )
        )
