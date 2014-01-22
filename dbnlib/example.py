from data import MNIST
from dbn import DBN
import numpy as np

# An example of how to use the library

# Creates a DBN
d = DBN([28*28, 500,500, 2000], 10, 0.1)

# Loads the Images and Labels from the MNIST database
images = MNIST.load_images('images')
labels = MNIST.load_labels('labels')

# Gets the training set of images
img = images[0:60000]
lbl = labels[0:60000]

# Permutes the data to ensure random batches
train_tuples  = zip(img, lbl)
train_tuples = np.random.permutation(train_tuples)
(img, lbl) = zip(*train_tuples)

# Loads the test images from the MNIST database
tst_img = MNIST.load_images('test_images')
tst_lbl = MNIST.load_labels('test_labels')

# Pre trains the DBN
d.pre_train(img,5,50)

# Executes a 100 iterations of label training
# Attempts to classify and prints the error rate
for i in xrange(0, 100):
  d.train_labels(img, lbl, 50, 50)
  tst_class = d.classify(tst_img,10)
  print 'Error over test data: {0}'.format(1 - (tst_class*tst_lbl).mean() * dbn.number_labels)


# Tests the DBN on test images
tst_class = d.classify(tst_img,20)
# Calculates the error
err_test = 1 - ((tst_class * tst_lbl).mean() * 10)

# Prints the error rate
print err_test

