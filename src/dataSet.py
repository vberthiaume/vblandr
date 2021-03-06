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


import tensorflow as tf
from tensorflow.python.framework import dtypes
import collections

#sms-tools stuff
import sys, os, os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sms-tools/software/models/'))
import utilFunctions as UF
import stft as STFT
from scipy.signal import get_window
from scipy.fftpack import fft, ifft

#ffmpeg + audio stuff 
import subprocess as sp
import scikits.audiolab
import bisect
#general stuff?
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import math

# we have 7 music genres
NUM_CLASSES     = 7
s_iTrainSize    = 8 * NUM_CLASSES  # 200000
s_iValid_size   = 6 * NUM_CLASSES  # 10000
s_iTestSize     = 6 * NUM_CLASSES  # 10000


SAMPLE_COUNT = 1 * 44100   # first 10 secs of audio
exponent = math.log(SAMPLE_COUNT, 2)+1
TOTAL_INPUTS = 2 ** int(exponent)

FORCE_PICKLING = False
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
overall_song_id = 0
ONE_HOT = False


# LIBRARY_PATH = '/media/kxstudio/LUSSIER/music/'
# LIBRARY_PATH = '/media/sf_stuff_for_virtual_machines/music/'
# LIBRARY_PATH = '/Volumes/Untitled/music/'

#nicolai -- not installed at this point
#LIBRARY_PATH = '/Users/nicolai/Music/vblandr'

#hesse
LIBRARY_PATH = '/mnt/c/Users/barth/Documents/vblandr/'

#gris
#LIBRARY_PATH = '/home/gris/Music/vblandr/'


def write_test_wav(cur_song_samples, str_id = ""):
    filename = LIBRARY_PATH +'test'+ str_id +'.wav'
    print ("writing", filename)
    scikits.audiolab.wavwrite(cur_song_samples, filename, fs=44100, enc='pcm16')

def getAllDataSets(train_dir, dtype=np.float32):

    pickle_file = getAllDataPickle(FORCE_PICKLING)

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset   = save['wholeTrainDataset']
        train_labels    = save['wholeTrainLabels']
        valid_dataset   = save['wholeValidDataset']
        valid_labels    = save['wholeValidLabels']
        test_dataset    = save['wholeTestDataset']
        test_labels     = save['wholeTestLabels']
        del save  # hint to help gc free up memory
        #print('after pickling, Training set',   train_dataset.shape, train_labels.shape)
        #print('after pickling, Validation set', valid_dataset.shape, valid_labels.shape)
        #print('after pickling, Test set',       test_dataset.shape,  test_labels.shape)

    train       = DataSet(train_dataset, train_labels,  dtype=dtype)
    validation  = DataSet(valid_dataset, valid_labels,  dtype=dtype)
    test        = DataSet(test_dataset,  test_labels,   dtype=dtype)

    return Datasets(train=train, validation=validation, test=test)

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels      = labels_dense.shape[0]
    index_offset    = np.arange(num_labels) * num_classes
    labels_one_hot  = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, songs, labels, dtype=np.float32):
        #global overall_song_id
       
        """Construct a DataSet. `dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`."""
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        
        #check that we have the same number of songs and labels
        assert songs.shape[0] == labels.shape[0], ('songs.shape: %s labels.shape: %s' % (songs.shape, labels.shape))
        self._num_examples = songs.shape[0]

#======================= DATA CONVERSION AND SHIT ===============================
        #the original range for int16 is [-32768, 32767]
        #if dtype == dtypes.float32:  
        #    songs = songs.astype(np.float32)            #cast the array into float32
        #    songs = np.multiply(songs, 1.0 / 65536)     #convert int16 range into [-.5, .5]
        #    songs = np.add(songs, .5)                   #convert int16 [-.5, .5] range into [0,1.0]
            
            # original code for pixels; #Convert from [0, 255] -> [0.0, 1.0].
            #songs = np.multiply(songs, 1.0 / 255.0) 

        #check that song files are valid 
        for cur_song, cur_song_dft in enumerate(songs):
            if cur_song == 0:
                print ("-----DATASET CONSTRUCTOR--------")
                print ("max: ",  np.amax(cur_song_dft))
                print ("min: ",  np.amin(cur_song_dft))
                print ("mean: ", np.mean(cur_song_dft))
                #overall_song_id += 1

        #check labels
        #use this for issue #3
        #labels = dense_to_one_hot(labels, NUM_CLASSES)

#================================================================================

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

def getAllDataPickle(p_bForce=False):
    #get relevant paths
    trainGenreNames, trainGenrePaths = getAllGenrePaths(LIBRARY_PATH + 'train_small/')
    testGenreNames, testGenrePaths   = getAllGenrePaths(LIBRARY_PATH + 'test_small/')
    pickle_file =                                       LIBRARY_PATH + 'allData.pickle'
    
    #obtain data for each genre in their individual pickle file
    allPickledTrainFilenames = getIndividualGenrePickles(trainGenrePaths, p_bForce)
    allPickledTestFilenames  = getIndividualGenrePickles(testGenrePaths,  p_bForce)

    #merge and randomize data from all genres into wholedatasets for training, validation, and test
    wholeValidDataset, wholeValidLabels, wholeTrainDataset, wholeTrainLabels = getWholeDataFromIndividualGenrePickles(allPickledTrainFilenames, s_iTrainSize, s_iValid_size)
    _,                                _, wholeTestDataset,  wholeTestLabels  = getWholeDataFromIndividualGenrePickles(allPickledTestFilenames,  s_iTestSize)
    wholeTrainDataset, wholeTrainLabels = randomize(wholeTrainDataset, wholeTrainLabels)
    wholeTestDataset,  wholeTestLabels  = randomize(wholeTestDataset,  wholeTestLabels)
    wholeValidDataset, wholeValidLabels = randomize(wholeValidDataset, wholeValidLabels)

    #save the data for later reuse: 
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

    print ('\n================== DATASETS BUILT ================')
    return pickle_file
    # ENDOF BUILDDATASETS

def getAllGenrePaths(music_dir):
    """return a list of all music genres, e.g., 'audiobook',  and their complete path"""
    dirs = os.listdir(music_dir)
    allAudioGenrePaths = []
    allAudioGenres = []
    for cur_dir in dirs:
        if not cur_dir.startswith('.') and not cur_dir.endswith('pickle') :
            allAudioGenrePaths.append(music_dir+cur_dir)
            allAudioGenres.append(cur_dir)
    return allAudioGenres, allAudioGenrePaths

def getIndividualGenrePickles(p_strDataFolderNames, p_bForce=False):
    """serialize list of data folders in their own pickle files, and return list of pickle filenames"""
    all_pickle_filenames = []
    for strCurFolderName in p_strDataFolderNames:
        cur_pickle_filename = strCurFolderName + '.pickle'
        all_pickle_filenames.append(cur_pickle_filename)
        if os.path.exists(cur_pickle_filename) and not p_bForce:
            print('%s already present - Skipping pickling.' % cur_pickle_filename)
        else:
            print '\nPickling',
            print cur_pickle_filename,
            dataset_cur_genre = getDataForGenre(strCurFolderName)
            try:
                #and try to pickle it
                with open(cur_pickle_filename, 'wb') as f:
                    pickle.dump(dataset_cur_genre, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', cur_pickle_filename, ':', e)
    return all_pickle_filenames


def removeInitialSilence(cur_song_pcm):
    #using absolute value
    env = abs(cur_song_pcm)
    env = env.astype(np.float32)            #cast the array into float32
    max = np.max(env)
    env = np.multiply(env, 1.0 / max)       #normalize so that max value is == 1.0
    
    #convolving as a way to do a fast moving average
    N = 100
    env = np.convolve(env, np.ones((N,))/N)[(N-1):]
    
    #detect first non-silent sample
    threshold = .01
    
    endOfSilence = bisect.bisect(env,threshold)
    
    print "\nend of silence: ", endOfSilence
    return cur_song_pcm[endOfSilence:]

# load data for each genre
def getDataForGenre(genre_folder):   
    """figure out the path to all the genre's song files, and how many songs we have"""
    
    
    global overall_song_id
    
    all_song_paths = []
    for path, dirs, files in os.walk(genre_folder):
        #insert file in correct label id
        for file in files:
            if not file.startswith('.') and (file.endswith('.wav') or file.endswith('.mp3')):
                all_song_paths.append(path+"/"+file)

    #data for cur genre will have shape all_song_paths x TOTAL_INPUTS
    #dataset_cur_genre = np.ndarray(shape=(len(all_song_paths), TOTAL_INPUTS), dtype=np.int16)
    dataset_cur_genre = np.ndarray(shape=(len(all_song_paths), TOTAL_INPUTS), dtype=np.float32)
    
    songId = 0
    #for each song in the current genre
    for cur_song_file in all_song_paths:
        try:
            # convert current song to np.int16 array. 
            print cur_song_file
            cur_song_pcm = songFile2pcm(cur_song_file)

            cleaned_cur_song_pcm = removeInitialSilence(cur_song_pcm)
            write_test_wav(cur_song_pcm, str(overall_song_id))
            
            overall_song_id = overall_song_id +1
            
            # only keep the first 2x TOTAL_INPUTS samples. since the fft is symetrical, we can use that to store more stuff
            short_cur_song_pcm = cleaned_cur_song_pcm[:2*TOTAL_INPUTS]
            #do the fft, keeping only the real numbers, ie the magnitude. mX has same len as cur_song_pcm, but is np.float64
            mX = fft(short_cur_song_pcm).real
            #only keep the first half since symmetrical, and we know len(mX) is multiple of 2
            mX = mX[:len(mX)/2]
            
            #PLOT THE THING
            #if songId == 0:
            #    fft_buffer = np.insert(mX, np.zeros(len(mX)), 0)
            #    for i in np.arange (len(fft_buffer)/2):
            #        fft_buffer[i] = fft_buffer[len(fft_buffer)-i-1]
            #    plt.plot(fft_buffer)
            #    plt.show()
           
            #need to convert to range 0,1 for tensorflow learning. 
            max = np.amax(mX)
            min = np.amin(mX)
            range = max - min
            mX = mX - min
            mX = mX/range

            #and put it in the dataset_cur_genre
            dataset_cur_genre[songId, :] = mX
            
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
    song_path2 = song_path + '.wav'
    command = [ 'ffmpeg',
            '-i', song_path,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '44100', # sms tools wavread can only read 44100 Hz
            '-ac', '1',     # mono file
            '-loglevel', 'quiet',
            '-']            #instead of having an output file, using '-' sends it in the pipe. not actually sure how this works.
    #run the command
    #print(song_path)
    print ".", 
    sys.stdout.flush()
    pipe = sp.Popen(command, stdout=sp.PIPE)
    #read the output into a numpy array
    stdoutdata = pipe.stdout.read()
    audio_array = np.fromstring(stdoutdata, dtype=np.int16)

    # size = len(audio_array)
    # print ("size: ", size)

    #export this to a wav file, to test it
    # write_test_wav(audio_array)
    return audio_array
    #END SONGFILE2PCM

# Merge individual genre datasets. Tune s_iTrainSize as needed to be able to fit all data in memory.
# Also create a validation dataset_cur_genre for hyperparameter tuning.
def getWholeDataFromIndividualGenrePickles(p_allPickledFilenames, p_iTrainSize, p_iValidSize=0):
    iNum_classes = len(p_allPickledFilenames)
    #make empty arrays for validation and training sets and labels
    whole_valid_dataset, valid_labels = make_arrays(p_iValidSize, TOTAL_INPUTS, ONE_HOT)
    whole_train_dataset, train_labels = make_arrays(p_iTrainSize, TOTAL_INPUTS, ONE_HOT)
    
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
            with open(strPickleFilename, 'rb') as f:
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
#END OF getWholeDataFromIndividualGenrePickles


def make_arrays(p_iNb_rows, p_iNb_cols, one_hot):
    if p_iNb_rows:
        #dataset_cur_genre = np.ndarray((p_iNb_rows, p_iNb_cols), dtype=np.int16)
        dataset_cur_genre = np.ndarray((p_iNb_rows, p_iNb_cols), dtype=np.float32)
        if one_hot:
            labels = np.ndarray((p_iNb_rows, NUM_CLASSES), dtype=np.int32)
        else:
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
