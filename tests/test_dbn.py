import unittest
from dbnlib.dbn import DBN

# Tests the Deep Belief Network class
class TestDBN(unittest.TestCase):

    # Tests the DBN can simply be constructed
    def test_dbn_constructor(self):
        layer_sizes = [10, 20]
        d = DBN(layer_sizes, 1)

    # Tests the DBN can be constructed with one layer only
    def test_one_layer_constructor(self):
        layer_sizes = [10]
        d = DBN(layer_sizes, 1)

    # Tests it throws an error if a negative layer size is given
    def test_negative_dimension_constructor(self):
        layer_sizes = [-10, 500]
        with self.assertRaises(ValueError):
            d = DBN(layer_sizes, 1)

    # Tests it throws an error if no layers are given
    def test_empty_config_constructor(self):
        layer_sizes = []
        with self.assertRaises(IndexError):
            d = DBN(layer_sizes, 1)

    # Tests the number of layers corresponds to the actual layer count (3)
    def test_number_layers_setup_correct(self):
        layer_sizes = [144, 50, 50, 2000]
        d = DBN(layer_sizes, 1)
        self.assertEquals(d.number_layers, 3)

    # Tests the number of inputs is created successfully
    def test_number_inputs_setup_correct(self):
        layer_sizes = [144, 50, 50, 2000]
        d = DBN(layer_sizes, 1)
        self.assertEquals(d.number_inputs, 144)

    # Tests the RBMs are built correctly
    def test_rbms_list_setup_correct(self):
        layer_sizes = [144, 50, 50, 2000]
        d = DBN(layer_sizes, 1)
        self.assertEquals(len(d._rbms), d.number_layers)

    # Tests the topology given and the topology created are the same
    def test_get_topology_constructs(self):
        layer_sizes = [144, 50, 50, 2000]
        number_labels = 1
        d = DBN(layer_sizes, number_labels)
        layer_sizes.append(number_labels)
        self.assertListEqual(d.get_topology(), layer_sizes)

    # Tests you can pre train on a correct data size
    def test_pre_train_correct_data_size(self):
        layer_sizes = [5, 20, 10]
        d = DBN(layer_sizes, 10)
        data = [[1,1,1,1,1], [1,1,1,1,1]]
        d.pre_train(data, 1 , 1)

    # Tests a badly shaped training set throws an error
    def test_pre_train_wrong_data_size(self):
        layer_sizes = [5, 20]
        d = DBN(layer_sizes, 10)
        data = [[1,1,1,1,1,1,1]]
        with self.assertRaises(ValueError):
            d.pre_train(data, 1, 1)

    # Tests a negative batch value throws an error
    def test_pre_train_negative_batch(self):
        layer_sizes = [5, 20, 10]
        d = DBN(layer_sizes, 10)
        data = [[1,1,1,1,1], [1,1,1,1,1]]
        with self.assertRaises(ValueError):
            d.pre_train(data, 1 , -1)

    # Tests that a DBN can be trained with labels
    def test_train_labels(self):
        layer_sizes = [5, 20, 10]
        d = DBN(layer_sizes, 2)
        data = [[1,1,1,1,1], [1,1,1,1,1]]
        labels = [[1,0], [0,1]]
        d.train_labels(data, labels, 1, 1)

    # Tests a badly shaped label training set throws an error
    def test_train_labels_wrong_label_size(self):
        layer_sizes = [5, 20, 10]
        d = DBN(layer_sizes, 2)
        data = [[1,1,1,1,1], [1,1,1,1,1]]
        labels = [[1,0], [0,1,1]]
        with self.assertRaises(ValueError):
            d.train_labels(data, labels, 1, 1)

    # Tests that mismatched labels and data throws an error
    def test_train_labels_no_labels(self):
        layer_sizes = [5, 20, 10]
        d = DBN(layer_sizes, 2)
        data = [[1,1,1,1,1], [1,1,1,1,1]]
        labels = []
        with self.assertRaises(ValueError):
            d.train_labels(data, labels, 1, 1)

    # Tests the DBN can classify data
    def test_classify(self):
        layer_sizes = [5, 20, 10]
        d = DBN(layer_sizes, 2)
        data = [[1,1,1,1,1]]
        probs = d.classify(data)

    # Tests the DBN throws an error when classifying badly shaped data
    def test_classify_wrong_shape(self):
        layer_sizes = [5, 20, 10]
        d = DBN(layer_sizes, 2)
        data = [[1,1,1,1]]
        with self.assertRaises(ValueError):
            probs = d.classify(data)