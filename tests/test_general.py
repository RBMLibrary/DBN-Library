import unittest
from dbnlib.dbn import DBN
from dbnlib.rbm import RBM

class TestDBN(unittest.TestCase):

    # General tests
    def test_logistic_zero(self):
        r = RBM(10, 10)
        self.assertEquals(r._RBM__logistic(0), 0.5)


