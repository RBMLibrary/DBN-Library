import unittest
from dbnlib.dbn import DBN
from dbnlib.rbm import RBM

class TestDBN(unittest.TestCase):
    # RBM tests
    def test_negative_hidden_constructor(self):
        with self.assertRaises(ValueError):
            r = RBM(10, -10)

    def test_negative_visible_constructor(self):
        with self.assertRaises(ValueError):
            r = RBM(-10, 10)

    # General tests
    def test_logistic_zero(self):
        r = RBM(10, 10)
        self.assertEquals(r._RBM__logistic(0), 0.5)


