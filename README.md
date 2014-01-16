Deep Belief Network Library
===========

This our implementation of a Deep Belief Network and Restricted Boltzmann Machine library. It also contains a file to extract images from the MNIST database as well as an example of how to run the DBN. Note: Numpy is required

rbm.py
-----------
Implements a Restricted Boltzmann Machine in python. Can be used on its own but is most effective when used in a DBN.

dbn.py
-----------
Implements a Deep Belief Network in python. Contains a list of RBMs that form a network with a classifier on the top. Image Data should be passed as a list of NUMPY arrays. Image labels should be in the same order as the Image data and should be a list of NUMPY arrays with a 1 in the corresponding index for that label. E.g. A label for class 3 of 5 would be [0 0 1 0 0].

data.py
-----------
Looks for a MNIST file in the current directory and extracts the appropriate labels and images and returns them as NUMPY arrays. Look at this for an idea of how to organise the data

example.py
-----------
An example of how to use the DBN. Creating a training set. Bulding a DBN. Pre-training, fine tuning and then outputting the error message. 

tasks.py
-----------
A simplified version of the processing task we use to train our DBNs. Note that this is just for reference and will not run. You need to create a function that retrieves images from a directory.


