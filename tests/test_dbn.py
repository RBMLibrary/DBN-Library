import unittest
from dbnlib.dbn import DBN

class TestDBN(unittest.TestCase):
    def test_number_layers_setup_correct(self):
        # Test that the number of layers corrosponds to the number of RBMs given
        # in the initial config. In this example there are 3 layers / RBMs because
        # the stacked RBMs share indivdual layers, so the first RBM is 144, 50. 
        # The second is 50, 50 etc.
        layer_sizes = [144, 50, 50, 2000]
        d = DBN(layer_sizes, 0, 0)
        self.assertEquals(d.number_layers, 3)

    def test_number_inputs_setup_correct(self):
        # The first layer in the layer sizes corrosponds to the inputs.
        layer_sizes = [144, 50, 50, 2000]
        d = DBN(layer_sizes, 0, 0)
        self.assertEquals(d.number_inputs, 144)

    def test_rbms_list_setup_correct(self):
        layer_sizes = [144, 50, 50, 2000]
        d = DBN(layer_sizes, 0, 0)
        self.assertEquals(len(d._rbms), d.number_layers)



