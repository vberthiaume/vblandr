# These are all the modules we'll be using later. Make sure you can import them before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

#First reload the data we generated in notmist.ipynb.
pickle_file = '../../udacity/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels  = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels  = save['valid_labels']
    test_dataset  = save['test_dataset']
    test_labels   = save['test_labels']
    del save  # hint to help gc free up memory
  
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
#Reformat into a shape that's more adapted to the models we're going to train:
#    data as a flat matrix,
#    labels as float 1-hot encodings.
image_size = 28
num_labels = 10

#this is just like in previous ass
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset,  test_labels  = reformat(test_dataset, test_labels)

print('Training set',   train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set',       test_dataset.shape,  test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
  
  
#================================================== Problem 1 ==================================================
#Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding
# a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor t using
# nn.l2_loss(t). The right amount of regularization should improve your validation / test accuracy.

import time
flat_img_size   = image_size * image_size

batch_size = 128 #this random number of training patterns will be used

#======================= OVERFITTING FOR PROBLEM 2 ==================
# training_size = 15 * batch_size
# train_dataset = train_dataset[:training_size]
# train_labels  = train_labels[:training_size]
# print ("training contains ", train_dataset.shape, " patterns")

graph = tf.Graph()
hidden_layer_units = 1024

#initialize everything
with graph.as_default():
    # Input data. The training data is currently empty, but a random minibatch will be fed in the placeholder during training
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, flat_img_size))
    tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset  = tf.constant(test_dataset)

    # DROPOUT PROBLEM 3
    keep_prob = tf.placeholder(tf.float32)

    # ------------ DEFINING LAYERS ------------
    # Input layer                                                                               # input is 1 x 784
    weights          = tf.Variable(tf.truncated_normal([flat_img_size, hidden_layer_units])) # Weights is 784 x 1024
    biases           = tf.Variable(tf.zeros([hidden_layer_units]))                           # Bias is 1 x 1024
    InputLayerOutput = tf.matmul(tf_train_dataset, weights) + biases                         # Output of input layer is (inputs X weights) + bias = 1 x 1024

    # 1st hidden layer
    hidden1_input    = tf.nn.dropout(tf.nn.relu(InputLayerOutput), keep_prob)                                  # Input is nn.relu(inputLayerOutput), so 1 x 1024
    weights1         = tf.Variable(tf.truncated_normal([hidden_layer_units, num_labels]))    # Weights is 1024 x 10
    biases1          = tf.Variable(tf.zeros([num_labels]))                                   # Bias is 1 x 10
    logits = tf.matmul(hidden1_input, weights1) + biases1                             # logits are (inputs X weights) + bias = 1 x 10
    # logits = tf.nn.dropout(tf.matmul(hidden1_input, weights1) + biases1, keep_prob)     # logits are (inputs X weights) + bias = 1 x 10

    # Training computations
    loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    beta_L2 = .05  #I believe this is called beta
    regularizers = (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases) + tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases), weights1) + biases1)
    test_prediction  = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases), weights1) + biases1)

#train the thing
num_steps = 3001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    start_time = time.clock()

    print("Initialized")
    for step in range(num_steps):
        # Generate a minibatch by pick an offset within the (randomized) training data. Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data   = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels [offset:(offset + batch_size), :]
        # Dictionary telling the session where to feed the minibatch. Keys are the placeholder nodes and the value are the numpy arrays.
        train_feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
        # Run the thing
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=train_feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%"  % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(feed_dict={keep_prob: 1.0}), valid_labels))
            # print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels))
    # print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    end_time = time.clock()
    print("Whole thing took: ", (end_time - start_time)/60, " minutes")

#================================================== Problem 2 ==================================================
#Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
# SEE AROUND LINES 60 TO 63

#================================================== Problem 3 ==================================================
#Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during
# training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides
# nn.dropout() for that, but you have to make sure it's only inserted during training.
#



# What happens to our extreme overfitting case?


#================================================== Problem 4 ==================================================
#Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep
# network is 97.1%.

#One avenue you can explore is to add multiple layers.

#Another one is to use learning rate decay:
#	global_step = tf.Variable(0)  # count the number of steps taken.
#	learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
