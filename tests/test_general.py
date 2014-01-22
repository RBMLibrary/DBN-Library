import unittest
from dbnlib.rbm import RBM

# Tests the general functions
class TestGeneral(unittest.TestCase):

    # Tests the logistic function computes the correct value
    def test_logistic_zero(self):
        r = RBM(10, 10)
        self.assertEquals(r._RBM__logistic(0), 0.5)