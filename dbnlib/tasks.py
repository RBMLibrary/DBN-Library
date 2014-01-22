# Another file showing how the library may be used

# Trains the DBN
def train_dbn(dbn, label_list, pre_epoch, train_epoch, train_loop):

    train_tuples  = zip(*retrieve_images(train_path, dbn.height, dbn.width, label_list))
    (test_images, test_labels) = retrieve_images(test_path, dbn.height, dbn.width, label_list)
    train_tuples = np.random.permutation(train_tuples)
    (train_images, train_labels) = zip(*train_tuples)

    training_method(dbn.dbn, train_images, train_labels, test_images, test_labels, pre_epoch, 50, train_epoch, 50, train_loop, 1)

    print"---------------------------------------------------------------"
    print"------------FINISHED--------------------TRAINING---------------"
    print"---------------------------------------------------------------"

# The training method used
def training_method(dbn, train_img, train_lbl, test_img, test_lbl, pre_epoch=5,
    pre_batch=50, train_epoch = 50, train_batch = 50, train_loop=20, samples=1 ):
    dbn.pre_train(train_img, pre_epoch, pre_batch)
    for i in xrange(0, train_loop):
        dbn.train_labels(train_img, train_lbl, train_epoch, train_batch)
        test_class = dbn.classify(test_img, samples)
        print 'Iteration {0}: Error over test data: {1}'.format(i, 1 - (test_class*test_lbl).mean() * dbn.number_labels)

# Retrieves the images
def retrieve_images(path, height, width, label_list):
    (images, labels) = imgpr.retrieve_all_images(path, height, width)
    trim_labels(labels)
    np_labels = convert_labels(labels, label_list)
    np_images = np.array(images)
    return (np_images, np_labels)

# Trims the labels
def trim_labels(labels):
    for index, item in enumerate(labels):
        item = item.split(".")[0]
        labels[index] = item

# Converts the labels into a list
def convert_labels(labels, label_list):
    np_labels = np.zeros((len(labels), (len(label_list))))
    for index, item in enumerate(labels):
        i = label_list.index(item)
        label_data = np.zeros(len(label_list))
        label_data[i] = 1
        np_labels[index] = label_data
    return np_labels
