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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import time
import math
import os
import tempfile
import collections

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import cPickle as pickle
#tensorflow stuff
import tensorflow as tf
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import dtypes
#sms-tools stuff
import sys, os, os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sms-tools/software/models/'))
import utilFunctions as UF
#ffmpeg stuff
import subprocess as sp
import scikits.audiolab

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float  ('learning_rate', 0.01,   'Initial learning rate.')
flags.DEFINE_integer('max_steps',     2000,   'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1',       128,    'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2',       32,     'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size',    100,    'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string ('train_dir',     'data', 'Directory to put the training data.')

# we have 7 genres
NUM_CLASSES = 7

LIBRARY_PATH = '/media/kxstudio/LUSSIER/music/'
# LIBRARY_PATH = '/Volumes/Untitled/music/'

SAMPLE_COUNT = 10 * 44100   # first 10 secs of audio

TOTAL_INPUTS = SAMPLE_COUNT

FORCE_PICKLING = False

overall_song_id = 0

def main(_):
    run_training()

def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = read_data_sets(FLAGS.train_dir)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        songs_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        # Build a Graph that computes predictions from the inference model.
        logits = inference(songs_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # Add to the Graph the Ops for loss calculation.
        loss = loss_funct(logits, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, FLAGS.learning_rate)
        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        # Add the variable initializer Op.
        init = tf.initialize_all_variables()
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        # Run the Op to initialize the variables.
        sess.run(init)
        # training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train, songs_placeholder, labels_placeholder)
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_sets.test)

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def read_data_sets(train_dir, dtype=dtypes.float32):

    #build train, valid and test datasets
    global overall_song_id
    overall_song_id = 0
    pickle_file = buildDataSets()

    #train_images should be a 4D uint8 numpy array [index, y, x, depth]."""

    #with is just a safe way of dealing with resources. handles correct closing if exceptions, etc
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset   = save['wholeTrainDataset']
        train_labels    = save['wholeTrainLabels']
        valid_dataset   = save['wholeValidDataset']
        valid_labels    = save['wholeValidLabels']
        test_dataset    = save['wholeTestDataset']
        test_labels     = save['wholeTestLabels']
        del save  # hint to help gc free up memory
        print('after piclking, Training set',   train_dataset.shape, train_labels.shape)
        print('after piclking, Validation set', valid_dataset.shape, valid_labels.shape)
        print('after piclking, Test set',       test_dataset.shape,  test_labels.shape)

        #TODO: train_dataset ETC MIGHT STILL BE PICKLED AT THAT POINT, EXPLAINING WHY AUDIO IS GARBLED? 

    train       = DataSet(train_dataset, train_labels,  dtype=dtype)
    validation  = DataSet(valid_dataset, valid_labels,  dtype=dtype)
    test        = DataSet(test_dataset,  test_labels,   dtype=dtype)

    return Datasets(train=train, validation=validation, test=test)

def write_test_wav(cur_song_samples, str_id = ""):
    filename = LIBRARY_PATH +'test'+ str_id +'.wav'
    print ("writing", filename)
    scikits.audiolab.wavwrite(cur_song_samples, filename, fs=44100, enc='pcm16')

class DataSet(object):
    def __init__(self, songs, labels, dtype=dtypes.float32):
        global overall_song_id
        """Construct a DataSet. `dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`."""
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        #check that we have the same number of songs and labels
        assert songs.shape[0] == labels.shape[0], ('songs.shape: %s labels.shape: %s' % (songs.shape, labels.shape))
        self._num_examples = songs.shape[0]

        # Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns] (assuming depth == 1)
        # this is not necessary for songs now, because songs is already of the shape [num_song=14, num_samples=44100]
        # assert songs.shape[3] == 1
        # songs = songs.reshape(songs.shape[0], songs.shape[1] * songs.shape[2])
        # if dtype == dtypes.float32:
        #     # Convert from [0, 255] -> [0.0, 1.0].
        #     songs = songs.astype(numpy.float32)
        #     songs = numpy.multiply(songs, 1.0 / 255.0)

        #TODO: NEED TO FIGURE OUT WHY UNPICKLED SOUNDS ARE SHIT OR EMPTY
        
        #we do need to check if we need to normalize it though... or not? not sure. 
        # for cur_song, cur_song_samples in enumerate(songs):
        #     # print (cur_song, np.amax(cur_song_samples))
        #     # print (cur_song, np.amin(cur_song_samples))
        #     print (cur_song, np.mean(cur_song_samples))

        #     #export this to a wav file, to test it
        #     # if cur_song == 0:
        #     write_test_wav(cur_song_samples, str(overall_song_id))
        #     overall_song_id += 1

        self._songs             = songs
        self._labels            = labels
        self._epochs_completed  = 0
        self._index_in_epoch    = 0

    @property
    def songs(self):
        return self._songs

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._songs = self._songs[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._songs[start:end], self._labels[start:end]
    # ENDOF DataSet 

def buildDataSets():
    # this algorithm will use the same number of train, valid, and test patterns for each genre/class.
    s_iTrainSize  = 2*NUM_CLASSES # 200000
    s_iValid_size = NUM_CLASSES  # 10000
    s_iTestSize   = NUM_CLASSES  # 10000

    # get a list of genres for training and testing
    # using test for now to test training

    trainGenreNames, trainGenrePaths = listGenres(LIBRARY_PATH + 'train_small/')
    testGenreNames, testGenrePaths  = listGenres( LIBRARY_PATH + 'test_small/')
    pickle_file =                                 LIBRARY_PATH + 'allData.pickle'
        
    allPickledTrainFilenames = maybe_pickle(trainGenrePaths, FORCE_PICKLING)
    allPickledTestFilenames  = maybe_pickle(testGenrePaths, FORCE_PICKLING)

    #call merge_dataset on data_sets and labels
    wholeValidDataset, wholeValidLabels, wholeTrainDataset, wholeTrainLabels = merge_dataset(allPickledTrainFilenames, s_iTrainSize, s_iValid_size)
    _,                                _, wholeTestDataset,  wholeTestLabels  = merge_dataset(allPickledTestFilenames,  s_iTestSize)

    wholeTrainDataset, wholeTrainLabels = randomize(wholeTrainDataset, wholeTrainLabels)
    wholeTestDataset,  wholeTestLabels  = randomize(wholeTestDataset,  wholeTestLabels)
    wholeValidDataset, wholeValidLabels = randomize(wholeValidDataset, wholeValidLabels)

    # Finally, let's save the data for later reuse: 
    try:
        f = open(pickle_file, 'wb')
        save = {'wholeTrainDataset':    wholeTrainDataset,
                'wholeTrainLabels':     wholeTrainLabels,
                'wholeValidDataset':    wholeValidDataset,
                'wholeValidLabels':     wholeValidLabels,
                'wholeTestDataset':     wholeTestDataset,
                'wholeTestLabels':      wholeTestLabels}
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


    statinfo = os.stat(pickle_file)
    # print('Compressed pickle size:', statinfo.st_size/1000000, "Mb")

    print ('================== DATASETS BUILT ================')
    return pickle_file
    # ENDOF BUILDDATASETS

def listGenres(music_dir):
    dirs = os.listdir(music_dir)
    allAudioGenrePaths = []
    allAudioGenres = []
    for cur_dir in dirs:
        if not cur_dir.startswith('.') and not cur_dir.endswith('pickle') :
            allAudioGenrePaths.append(music_dir+cur_dir)
            allAudioGenres.append(cur_dir)
    return allAudioGenres, allAudioGenrePaths

def maybe_pickle(p_strDataFolderNames, p_bForce=False):
    dataset_all_genres = []
    #data_folders are either the train or test set. folders within those are A, B, etc
    for strCurFolderName in p_strDataFolderNames:
        #we will serialize those subfolders (A, B, etc), that's what pickling is
        strCurSetFilename = strCurFolderName + '.pickle'
        #add the name of the current pickled subfolder to the list
        dataset_all_genres.append(strCurSetFilename)
        #if the pickled folder already exists, skip
        if os.path.exists(strCurSetFilename) and not p_bForce:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % strCurSetFilename)
        else:
            #call the load_letter function def above 
            print('Pickling %s.' % strCurSetFilename)
            dataset_cur_genre = load_genre(strCurFolderName)
            try:
                #and try to pickle it
                with open(strCurSetFilename, 'wb') as f:
                    # TODO: WHEN IS THIS UNPICKLED^???
                    pickle.dump(dataset_cur_genre, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', strCurSetFilename, ':', e)
    return dataset_all_genres


# load data for each genre
def load_genre(genre_folder):
    """Load all song data for a single genre"""
    
    global overall_song_id

    #figure out the path to all the genre's song files, and how many songs we have
    all_song_paths = []
    for path, dirs, files in os.walk(genre_folder):
        #insert file in correct label id
        for file in files:
            if not file.startswith('.') and (file.endswith('.wav') or file.endswith('.mp3')):
                all_song_paths.append(path+"/"+file)

    #our dataset 2d ndarray will be len(all_song_paths) x sample_count

    dataset_cur_genre = np.ndarray(shape=(len(all_song_paths), TOTAL_INPUTS), dtype=np.float32)
    
    songId = 0
    #for each song in the current genre
    for cur_song_file in all_song_paths:
        try:
            # convert current song to numpy array. when using images we were normalizing using pixel depth... should we do something different? Or pcmm is already [0,1]?
            # genre_path = '/media/kxstudio/LUSSIER/music/audiobooks/Alice_In_Wonderland_complete/'
            # song_path = genre_path +'Alice_In_Wonderland_ch_01.mp3'
            cur_song_pcm = songFile2pcm(cur_song_file)

            # only keep the first sample_count samples
            cur_song_pcm = cur_song_pcm[0:SAMPLE_COUNT]

            # test whether song is correctly extracted
            # write_test_wav(cur_song_pcm, str(overall_song_id))

            print ("song", overall_song_id, np.mean(cur_song_pcm))
            overall_song_id += 1


            #and put it in the dataset_cur_genre
            dataset_cur_genre[songId, :] = cur_song_pcm
            songId += 1
        except IOError as e:
            print('skipping ', cur_song_file, ':', e)
    #in case we skipped some songs, only keep the first songId songs in dataset_cur_genre
    dataset_cur_genre = dataset_cur_genre[0:songId, :]
    
    # print('Full dataset_cur_genre tensor:', dataset_cur_genre.shape)
    # print('Mean:', np.mean(dataset_cur_genre))
    # print('Standard deviation:', np.std(dataset_cur_genre))
    return dataset_cur_genre
    #END LOAD GENRE

def songFile2pcm(song_path):
    command = [ 'ffmpeg',
            '-i', song_path,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '44100', # sms tools wavread can only read 44100 Hz
            '-ac', '1', # mono file
            '-loglevel', 'quiet',
            '-']    #instead of having an output file, using '-' sends it in the pipe. not actually sure how this works.
    #run the command
    pipe = sp.Popen(command, stdout=sp.PIPE)
    #read the output into a numpy array
    stdoutdata = pipe.stdout.read()
    audio_array = np.fromstring(stdoutdata, dtype="int16")

    # size = len(audio_array)
    # print ("size: ", size)

    #export this to a wav file, to test it
    # write_test_wav(audio_array)
    return audio_array
    #END SONGFILE2PCM

# Merge individual genre datasets. Tune s_iTrainSize as needed to be able to fit all data in memory.
# Also create a validation dataset_cur_genre for hyperparameter tuning.
def merge_dataset(p_allPickledFilenames, p_iTrainSize, p_iValidSize=0):
    iNum_classes = len(p_allPickledFilenames)
    #make empty arrays for validation and training sets and labels
    whole_valid_dataset, valid_labels = make_arrays(p_iValidSize, TOTAL_INPUTS)
    whole_train_dataset, train_labels = make_arrays(p_iTrainSize, TOTAL_INPUTS)
    
    #number of items per class. // is an int division in python3, not sure in python2
    iNbrOfValidItemsPerClass = p_iValidSize // iNum_classes
    iNbrOfTrainItemPerClass  = p_iTrainSize // iNum_classes
  
    #figure out useful indexes for the loop
    iStartValidId, iStartTrainId = 0, 0
    iEndValidId, iEndTrainId = iNbrOfValidItemsPerClass, iNbrOfTrainItemPerClass
    iEndListId = iNbrOfValidItemsPerClass+iNbrOfTrainItemPerClass
  
    #for each file in p_allPickledFilenames
    for iPickleFileId, strPickleFilename in enumerate(p_allPickledFilenames):    
        try:
            #open the file
            with open(strPickleFilename, 'rb') as f:
                #unpicke 3d array for current file
                cur_genre_dataset = pickle.load(f)
                # let's shuffle the items to have random validation and training set. np.random.shuffle suffles only first dimension
                np.random.shuffle(cur_genre_dataset)
        
                #if we asked for a validation set, use the first items for it
                if whole_valid_dataset is not None:
                    #the first iNbrOfValidItemsPerClass items in letter_set are used for the validation set
                    whole_valid_dataset[iStartValidId:iEndValidId, :] = cur_genre_dataset[:iNbrOfValidItemsPerClass, :]
                    #label all images with the current file id 
                    valid_labels[iStartValidId:iEndValidId] = iPickleFileId
                    #update ids for the train set
                    iStartValidId += iNbrOfValidItemsPerClass
                    iEndValidId   += iNbrOfValidItemsPerClass
                    
                #the rest of the items are used for the training set
                whole_train_dataset[iStartTrainId:iEndTrainId, :] = cur_genre_dataset[iNbrOfValidItemsPerClass:iEndListId, :]
                train_labels[iStartTrainId:iEndTrainId] = iPickleFileId
                iStartTrainId += iNbrOfTrainItemPerClass
                iEndTrainId   += iNbrOfTrainItemPerClass
        except Exception as e:
            print('Unable to process data from', strPickleFilename, ':', e)
            raise 
    return whole_valid_dataset, valid_labels, whole_train_dataset, train_labels
    #END OF merge_dataset

def make_arrays(p_iNb_rows, p_iNb_cols):
    if p_iNb_rows:
        dataset_cur_genre = np.ndarray((p_iNb_rows, p_iNb_cols), dtype=np.float32)
        labels = np.ndarray(p_iNb_rows, dtype=np.int32)
    else:
        dataset_cur_genre, labels = None, None
    return dataset_cur_genre, labels

# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.
def randomize(p_3ddataset_cur_genre, p_vLabels):
    #with int x as parameter, np.random.permutation returns a random permutation of np.arange(x)
    vPermutation = np.random.permutation(p_vLabels.shape[0])
    threeDShuffleddataset_cur_genre = p_3ddataset_cur_genre[vPermutation,:]
    threeDShuffledLabels  = p_vLabels  [vPermutation]
    return threeDShuffleddataset_cur_genre, threeDShuffledLabels

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        songs_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    songs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, TOTAL_INPUTS))
    labels_placeholder = tf.placeholder(tf.int32,   shape=(batch_size))
    return songs_placeholder, labels_placeholder

def inference(images, hidden1_units, hidden2_units):
    #Build the MNIST model up to where it may be used for inference.

    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([TOTAL_INPUTS, hidden1_units], stddev=1.0 / math.sqrt(float(TOTAL_INPUTS))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss_funct(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
    }

    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next `batch size ` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = { images_pl: images_feed, labels_pl: labels_feed}
    return feed_dict

def do_eval(sess, eval_correct, songs_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        songs_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
            input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, songs_placeholder, labels_placeholder)
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
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
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
