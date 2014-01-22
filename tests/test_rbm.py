import unittest
from dbnlib.rbm import RBM

# Tests the Restricted Boltzmann Machine class
class TestRBM(unittest.TestCase):

    # Tests the default construction of an RBM object
    def test_rbm(self):
        r = RBM(10, 10)

    # Tests that an error is thrown if a bad hidden value is passed
    def test_negative_hidden_constructor(self):
        with self.assertRaises(ValueError):
            r = RBM(10, -10)

    # Tests that an error is thrown if a bad visible value is passed
    def test_negative_visible_constructor(self):
        with self.assertRaises(ValueError):
            r = RBM(-10, 10)

    # Tests that an RBM can be trained on good data
    def train_rbm(self):
        r = RBM(5, 10)
        data = [[1,1,1,1,1],[0,0,0,0,0]]
        r.train(data,1,1)

    # Tests that an RBM throws an error on badly shaped training data
    def test_train_rbm_wrong_shape(self):
        r = RBM(5, 10)
        data = [[0,0,0,0]]
        with self.assertRaises(ValueError):
            r.train(data,1,1)

    # Tests that an RBM throws an error on negative batch number
    def test_train_rbm_negative_batches(self):
        r = RBM(5, 10)
        data = [[1,1,1,1,1],[0,0,0,0,0]]
        with self.assertRaises(ValueError):
            r.train(data,1,-1)

    # Tests that an RBM can regenerate good data
    def test_regenerate(self):
        r = RBM(5, 10)
        data = [[1,1,1,1,1]]
        v,h = r.regenerate(data, 1)

    # Tests that an RBM throws an error on badly shaped regeneration data
    def test_regenerate_wrong_shape(self):
        r = RBM(5, 10)
        data = [[1,1,1,1]]
        with self.assertRaises(ValueError):
            v,h = r.regenerate(data, 1)

    # Tests that an RBM throws an error on a negative sample value
    def test_regenerate_negative_samples(self):
        r = RBM(5, 10)
        data = [[1,1,1,1,1]]
        with self.assertRaises(UnboundLocalError):
            v,h = r.regenerate(data, -1)