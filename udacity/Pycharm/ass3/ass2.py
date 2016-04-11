# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# First reload the data we generated in 1_notmnist.ipynb.
pickle_file = '../../udacity/notMNIST.pickle'
#with is just a safe way of dealing with resources. handles correct closing if exceptions, etc
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

#Reformat into a shape that's more adapted to the models we're going to train:
#    data as a flat matrix,
#    labels as float 1-hot encodings.
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    #-1 in reshape means 'use whatever makes sense', either flatten the whole thing or keep previous dimensions.
    # here it keeps the previous dimension
    print(labels.shape)
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    #the first part of this, np.arange(num_labels), creates an array that is [0,1,2...9]
    #we then check, for all rows of labels, if that arange is equal to the content of the row
    #so if labels[3] = 1, we get something like [false,true,false...], and this is converted to [0.0,1.0,0.0...]
    #None is actually optional, it just means don't bother about this dimension or something.
    #or actually, we need None if labels has more than 1 column
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset,  test_labels  = reformat(test_dataset,  test_labels)

print('Training set',   train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set',       test_dataset.shape,  test_labels.shape)


#================================================== PROBLEM ================================================
#Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units
# (nn.relu()) and 1024 hidden nodes. This model should improve your validation / test accuracy.

def accuracy(predictions, labels):
    #argmax returns the indices of the maximum values across dimension 1 ie colums
    #predictions  has shape (10000,10). this == tests if the max from predictions matches the max (ie the only non-null) label
    sum_all_correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    sum_all = predictions.shape[0]
    return 100.0 * sum_all_correct / sum_all

# import math

batch_size = 128 #this random number of training patterns will be used
graph = tf.Graph()
hidden1_units = 1024

#initialize everything
with graph.as_default():
    # Input data. The training data is currently empty, but a random minibatch will be fed in the placeholder during training
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset  = tf.constant(test_dataset)


    # Input layer
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden1_units]))
    biases  = tf.Variable(tf.zeros([hidden1_units]))
    InputLayerOutput = tf.matmul(tf_train_dataset, weights) + biases

    # 1st hidden layer
    weights1= tf.Variable(tf.truncated_normal([hidden1_units, num_labels]))
    biases1 = tf.Variable(tf.zeros([num_labels]))

    hidden1 = tf.nn.relu(InputLayerOutput)


    # Training computation.
    # logits = tf.matmul(tf_train_dataset, weights) + biases
    logits = tf.matmul(hidden1, weights1) + biases1

    loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases), weights1) + biases1)
    test_prediction  = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases), weights1) + biases1)

#another person's code from the forum, also works. basically is the same.
# num_nodes= 1024
# batch_size = 128

# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)

#     # Variables.
#     weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
#     biases_1 = tf.Variable(tf.zeros([num_nodes]))
#     weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
#     biases_2 = tf.Variable(tf.zeros([num_labels]))

#     # Training computation.
#     relu_layer=tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
#     logits = tf.matmul(relu_layer, weights_2) + biases_2
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
#     test_prediction =  tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)


#train the thing
num_steps = 3001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Generate a minibatch by pick an offset within the (randomized) training data. Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data   = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels [offset:(offset + batch_size), :]
        # Dictionary telling the session where to feed the minibatch. Keys are the placeholder nodes and the value are the numpy arrays.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        # Run the thing
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%"  % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


