from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import os, os.path
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

#sms-tools stuff
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sms-tools/software/models/'))
import utilFunctions as UF
#ffmpeg stuff
import subprocess as sp


#random init stuff
np.random.seed(133)
s_sample_count = 10 * 44100   # first 10 secs of audio

def main():
        
    #build train, valid and test datasets
    pickle_file = buildDataSets()

    # ---------------------------------- MAKE A GODDAMN NEURAL NETWORK ----------------------------------

    # END MAIN
      
def buildDataSets():
    # this algorithm will use the same number of train, valid, and test patterns for each genre/class.
    s_iTrainSize  = 2*7 # 200000
    s_iValid_size = 7  # 10000
    s_iTestSize   = 7  # 10000

    # get a list of genres for training and testing
    # using test for now to test training
    trainGenreNames, trainGenrePaths = listGenres('/media/kxstudio/LUSSIER/music/train_small/') #listGenres('/media/kxstudio/LUSSIER/music/train/')
    testGenreNames, testGenrePaths  = listGenres('/media/kxstudio/LUSSIER/music/test/')

    allPickledTrainFilenames = maybe_pickle(trainGenrePaths)
    allPickledTestFilenames  = maybe_pickle(testGenrePaths)

    #call merge_dataset on data_sets and labels
    wholeValidDataset, wholeValidLabels, wholeTrainDataset, wholeTrainLabels = merge_dataset(allPickledTrainFilenames, s_iTrainSize, s_iValid_size)
    _,                                _, wholeTestDataset,  wholeTestLabels  = merge_dataset(allPickledTestFilenames,  s_iTestSize)

    print('Training:',   wholeTrainDataset.shape, wholeTrainLabels.shape)
    print('Validation:', wholeValidDataset.shape, wholeValidLabels.shape)
    print('Testing:',    wholeTestDataset.shape,  wholeTestLabels.shape)

    wholeTrainDataset, wholeTrainLabels = randomize(wholeTrainDataset, wholeTrainLabels)
    wholeTestDataset,  wholeTestLabels  = randomize(wholeTestDataset,  wholeTestLabels)
    wholeValidDataset, wholeValidLabels = randomize(wholeValidDataset, wholeValidLabels)

    print(wholeTrainDataset.shape)
    print(wholeTestDataset.shape)
    print(wholeValidDataset.shape)


        # Finally, let's save the data for later reuse:
    pickle_file = '/media/kxstudio/LUSSIER/music/allData.pickle'

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
    print('Compressed pickle size:', statinfo.st_size/1000000, "Mb")

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
                    pickle.dump(dataset_cur_genre, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', strCurSetFilename, ':', e)
    return dataset_all_genres


# load data for each genre
def load_genre(genre_folder):
    """Load all song data for a single genre"""
    
    #figure out the path to all the genre's song files, and how many songs we have
    all_song_paths = []
    for path, dirs, files in os.walk(genre_folder):
        #insert file in correct label id
        for file in files:
            if not file.startswith('.') and (file.endswith('.wav') or file.endswith('.mp3')):
                all_song_paths.append(path+"/"+file)

    #our dataset 2d ndarray will be len(all_song_paths) x sample_count

    dataset_cur_genre = np.ndarray(shape=(len(all_song_paths), s_sample_count), dtype=np.float32)
    
    songId = 0
    #for each song in the current genre
    for cur_song_file in all_song_paths:
        try:
            # convert current song to numpy array. when using images we were normalizing using pixel depth... should we do something different? Or pcmm is already [0,1]?
            # genre_path = '/media/kxstudio/LUSSIER/music/audiobooks/Alice_In_Wonderland_complete/'
            # song_path = genre_path +'Alice_In_Wonderland_ch_01.mp3'
            cur_song_pcm = songFile2pcm(cur_song_file)

            # only keep the first sample_count samples
            cur_song_pcm = cur_song_pcm[0:s_sample_count]

            #and put it in the dataset_cur_genre
            dataset_cur_genre[songId, :] = cur_song_pcm
            songId += 1
        except IOError as e:
            print('skipping ', cur_song_file, ':', e)
    #in case we skipped some songs, only keep the first songId songs in dataset_cur_genre
    dataset_cur_genre = dataset_cur_genre[0:songId, :]
    
    print('Full dataset_cur_genre tensor:', dataset_cur_genre.shape)
    print('Mean:', np.mean(dataset_cur_genre))
    print('Standard deviation:', np.std(dataset_cur_genre))
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
    #export this to a wav file, to test it
    # import scikits.audiolab
    # scikits.audiolab.wavwrite(audio_array, path+'test.wav', fs=44100, enc='pcm16')
    return audio_array
    #END SONGFILE2PCM

# Merge individual genre datasets. Tune s_iTrainSize as needed to be able to fit all data in memory.
# Also create a validation dataset_cur_genre for hyperparameter tuning.
def merge_dataset(p_allPickledFilenames, p_iTrainSize, p_iValidSize=0):
    iNum_classes = len(p_allPickledFilenames)
    #make empty arrays for validation and training sets and labels
    whole_valid_dataset, valid_labels = make_arrays(p_iValidSize, s_sample_count)
    whole_train_dataset, train_labels = make_arrays(p_iTrainSize, s_sample_count)
    
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


if __name__=="__main__":
   main()