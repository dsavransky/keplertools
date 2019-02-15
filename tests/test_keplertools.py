from __future__ import print_function
import unittest
from keplertools.fun import *
import numpy as np

class TestKeplerTools(unittest.TestCase):
    """
    Test method: verify invertibility of all transformations using random inputs
    """

    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_eccanom_vecM_scalare(self):
        #use a normal distribution with sigma 2\pi to ensure that some M values are outside of
        #(0,2\pi) range
        M = np.random.randn(int(1e6))*2*np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M,e)   
        calcM = E - e*np.sin(E)
        self.assertTrue(np.max(np.abs(calcM - np.mod(M, 2*np.pi))) < np.finfo(float).eps*10.)
    
    def test_eccanom_vecM_scalare_noc(self):
        #use a normal distribution with sigma 2\pi to ensure that some M values are outside of
        #(0,2\pi) range
        M = np.random.randn(int(1e6))*2*np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M,e,noc=True)   
        calcM = E - e*np.sin(E)
        self.assertTrue(np.max(np.abs(calcM - np.mod(M, 2*np.pi))) < np.finfo(float).eps*10.)

    def test_eccanom_vecM_vece(self):
        #use a normal distribution with sigma 2\pi to ensure that some M values are outside of
        #(0,2\pi) range
        M = np.random.randn(int(1e6))*2*np.pi
        e = np.random.rand(int(1e6))

        E = eccanom(M,e)   
        calcM = E - e*np.sin(E)
        self.assertTrue(np.max(np.abs(calcM - np.mod(M, 2*np.pi))) < np.finfo(float).eps*10.)

    def test_eccanom_vecM_vece_noc(self):
        #use a normal distribution with sigma 2\pi to ensure that some M values are outside of
        #(0,2\pi) range
        M = np.random.randn(int(1e6))*2*np.pi
        e = np.random.rand(int(1e6))

        E = eccanom(M,e,noc=True)   
        calcM = E - e*np.sin(E)
        self.assertTrue(np.max(np.abs(calcM - np.mod(M, 2*np.pi))) < np.finfo(float).eps*10.)

    def test_eccanom_scalarM_scalare(self):
        M = np.random.rand(1)[0]*2*np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M,e)   
        calcM = E - e*np.sin(E)
        self.assertTrue(np.max(np.abs(calcM - np.mod(M, 2*np.pi))) < np.finfo(float).eps*10.)

    def test_eccanom_scalarM_scalare_noc(self):
        M = np.random.rand(1)[0]*2*np.pi
        e = np.random.rand(1)[0]

        E = eccanom(M,e,noc=True)   
        calcM = E - e*np.sin(E)
        self.assertTrue(np.max(np.abs(calcM - np.mod(M, 2*np.pi))) < np.finfo(float).eps*10.)

    def test_eccanom_badinputs(self):
         M = np.random.rand(100)
         e = np.random.rand(30)

         with self.assertRaises(AssertionError):
            E = eccanom(M,e,noc=True) 

    def test_orbElem2vec_1body(self):
        #generate one body full orbit
        a = np.random.rand(1)[0]*100
        e = np.random.rand(1)[0]
        O = np.random.rand(1)[0]*2*np.pi
        I = np.random.rand(1)[0]*np.pi
        w = np.random.rand(1)[0]*2*np.pi
        E = np.linspace(0.01,2*np.pi-0.01,100) #bad stuff happens in mod right at boundaries so clip E

        #convert back and forth to vectors
        rs,vs,ABs = orbElem2vec(E,1.0,orbElem=(a,e,O,I,w),returnAB=True)
        a1,e1,E1,O1,I1,w1,P1,tau1 = vec2orbElem(rs,vs,1.0)

        tol = np.spacing(np.max(np.abs(np.linalg.norm(rs,axis=0))))*10000

        #check that you got back what you put in
        self.assertTrue(np.max(np.abs(a - a1))<tol)
        self.assertTrue(np.max(np.abs(e - e1))<tol)
        self.assertTrue(np.max(np.abs(O - O1))<tol)
        self.assertTrue(np.max(np.abs(w - w1))<tol)
        self.assertTrue(np.max(np.abs(I - I1))<tol)
        self.assertTrue(np.max(np.abs(E - E1))<tol)

        #convert back and forth again using A,B
        rs,vs = orbElem2vec(E,1.0,AB=ABs)
        a1,e1,E1,O1,I1,w1,P1,tau1 = vec2orbElem(rs,vs,1.0)

        self.assertTrue(np.max(np.abs(a - a1))<tol)
        self.assertTrue(np.max(np.abs(e - e1))<tol)
        self.assertTrue(np.max(np.abs(O - O1))<tol)
        self.assertTrue(np.max(np.abs(w - w1))<tol)
        self.assertTrue(np.max(np.abs(I - I1))<tol)
        self.assertTrue(np.max(np.abs(E - E1))<tol)


    def test_orbElem2vec_Nbody(self):
        #generate n bodies at random phases
        n = int(1e4)
        a = np.random.rand(n)*100
        e = np.random.rand(n)
        O = np.random.rand(n)*2*np.pi
        I = np.random.rand(n)*np.pi
        w = np.random.rand(n)*2*np.pi
        E = np.random.rand(n)*2*np.pi
        mus = np.random.rand(n)

        #convert back and forth to vectors
        rs,vs,ABs = orbElem2vec(E,mus,orbElem=(a,e,O,I,w),returnAB=True)
        a1,e1,E1,O1,I1,w1,P1,tau1 = vec2orbElem(rs,vs,mus)

        tol = np.spacing(np.max(np.abs(np.linalg.norm(rs,axis=0))))*100000

        #check that you got back what you put in
        self.assertTrue(np.max(np.abs(a - a1))<tol)
        self.assertTrue(np.max(np.abs(e - e1))<tol)
        self.assertTrue(np.max(np.abs(O - O1))<tol)
        self.assertTrue(np.max(np.abs(w - w1))<tol)
        self.assertTrue(np.max(np.abs(I - I1))<tol)
        self.assertTrue(np.max(np.abs(E - E1))<tol)

        #convert back and forth again using A,B
        rs,vs = orbElem2vec(E,mus,AB=ABs)
        a1,e1,E1,O1,I1,w1,P1,tau1 = vec2orbElem(rs,vs,mus)

        self.assertTrue(np.max(np.abs(a - a1))<tol)
        self.assertTrue(np.max(np.abs(e - e1))<tol)
        self.assertTrue(np.max(np.abs(O - O1))<tol)
        self.assertTrue(np.max(np.abs(w - w1))<tol)
        self.assertTrue(np.max(np.abs(I - I1))<tol)
        self.assertTrue(np.max(np.abs(E - E1))<tol)

    def test_trueanom(self):
        n = int(1e6)
        E = np.random.rand(n)*2*np.pi
        e = np.random.rand(n)
        nu = trueanom(E,e)

        self.assertTrue(np.max(np.abs(e + (1-e**2)/(1+e*np.cos(nu))*np.cos(nu) - np.cos(E))) < 1e-9)

