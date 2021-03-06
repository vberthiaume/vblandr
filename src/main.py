# All edits to original document Copyright 2016 Vincent Berthiaume. 
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

######################################## IMPORTS ##########################################
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#tensorflow stuff
import time
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#all functions related to datasets
import dataSet

######################################## GLOBAL VARIABLES ##########################################
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float  ('learning_rate', 0.01,   'Initial learning rate.')
flags.DEFINE_integer('max_steps',     2000,   'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1',       128,    'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2',       32,     'Number of units in hidden layer 2.')
# flags.DEFINE_integer('batch_size',    100,    'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('batch_size',    2,      'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string ('train_dir',     'data', 'Directory to put the training data.')


######################################## ACTUAL CODE ##########################################
def main(_):
    
    data_sets = dataSet.getAllDataSets(FLAGS.train_dir)

    with tf.Graph().as_default():                                                       #using default graph
        #MEMBER FUNCTIONS 
        songs_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)    # Generate placeholders for the songs and labels.
        logits          = inference(songs_placeholder, FLAGS.hidden1, FLAGS.hidden2)    # Build a Graph that computes predictions from the inference model.
        loss            = loss_funct(logits, labels_placeholder)                        # Add to the Graph the Ops for loss calculation.
        train_op        = training(loss, FLAGS.learning_rate)                           # Add to the Graph the Ops that calculate and apply gradients.
        eval_correct    = evaluation(logits, labels_placeholder)                        # Add the Op to compare the logits to the labels during evaluation.
        #TF FUNCTION
        summary_op      = tf.merge_all_summaries()                                      # Build the summary operation based on the TF collection of Summaries.
        init            = tf.initialize_all_variables()                                 # Add the variable initializer Op.
        #saver           = tf.train.Saver()                                              # Create a saver for writing training checkpoints.the saver creates a bunch of files, so commented for now.
        sess            = tf.Session()                                                  # Create a session for running Ops on the Graph.
        summary_writer  = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)           # Instantiate a SummaryWriter to output summaries and the Graph.
        sess.run(init)                                                                  # Run the Op to initialize the variables.

         
        start_time = time.time()
        # training loop.
        for step in xrange(FLAGS.max_steps):
            # Fill a feed dictionary with the  data for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train, songs_placeholder, labels_placeholder)
            # Run one step of the model.  The return values are the activations from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them in the list passed to sess.run() and the value tensors will be returned in the tuple from the call.
            _, loss_value   = sess.run([train_op, loss], feed_dict=feed_dict)
            
            if step % 100 == 0:
                duration = time.time() - start_time
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                start_time = time.time()
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                #saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_sets.test)
    #ENDOF MAIN()

def placeholder_inputs(batch_size):
    songs_placeholder  = tf.placeholder(tf.float32, shape=(batch_size, dataSet.TOTAL_INPUTS))
    if dataSet.ONE_HOT:
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, dataSet.NUM_CLASSES))
    else:
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return songs_placeholder, labels_placeholder

def inference(images, hidden1_units, hidden2_units):
    #Build the MNIST model up to where it may be used for inference.

    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([dataSet.TOTAL_INPUTS, hidden1_units], stddev=1.0 / math.sqrt(float(dataSet.TOTAL_INPUTS))), name='weights')
        biases  = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
        biases  = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, dataSet.NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
        biases  = tf.Variable(tf.zeros([dataSet.NUM_CLASSES]), name='biases')
        logits  = tf.matmul(hidden2, weights) + biases
    return logits

def loss_funct(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def fill_feed_dict(data_set, images_pl, labels_pl):
    """Create the feed_dict for the placeholders filled with the next `batch size` examples."""
    images_feed, labels_feed    = data_set.next_batch(FLAGS.batch_size)
    feed_dict                   = { images_pl: images_feed, labels_pl: labels_feed}
    return feed_dict

def do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        songs_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from getAllDataSets().
    """
    # And run one epoch of eval.
    true_count      = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples    = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict   = fill_feed_dict(data_set, songs_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

def training(loss, learning_rate):
    """Sets up the training Ops. Creates a summarizer to track the loss over time in TensorBoard. Creates an optimizer and applies the gradients 
    to all trainable variables. The Op returned by this function is what must be passed to the`sess.run()` call to cause the model to train.
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, dataSet.NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, dataSet.NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op. It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1) of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))



if __name__ == '__main__':
    tf.app.run()
