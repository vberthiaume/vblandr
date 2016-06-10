from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
#sms-tools stuff
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sms-tools/software/models/'))
import utilFunctions as UF
#ffmpeg stuff
import subprocess as sp

s_sample_count = 10 * 44100   # first 10 secs of audio

def main():
    np.random.seed(133)
    
    #---------------------------------- BUILD DATA SETS ----------------------------------
    #this algorithm will use the same number of train, valid, and test patterns for each genre/class.
    s_iTrainSize  = 2*7 # 200000
    s_iValid_size = 7  # 10000
    s_iTestSize   = 0  # 10000

    #get a list of genres for training and testing
    #using test for now to test training
    trainGenreNames, trainGenrePaths = listGenres('/media/kxstudio/LUSSIER/music/test/') #listGenres('/media/kxstudio/LUSSIER/music/train/')
    testGenreNames  = listGenres('/media/kxstudio/LUSSIER/music/test/')

    allPickledTrainFilenames = maybe_pickle(trainGenrePaths)
    # allPickledTestFilenames  = maybe_pickle(testGenreNames)

    #call merge_dataset on data_sets and labels
    wholeValidDataset, wholeValidLabels, wholeTrainDataset, wholeTrainLabels = merge_dataset(allPickledTrainFilenames, s_iTrainSize, s_iValid_size)
    _,                                _, wholeTestDataset,  wholeTestLabels  = merge_dataset(allPickledTestFilenames,  s_iTestSize)

    if False:
        #print shapes for data sets and their respective labels. data sets are 3d arrays with [image_id,x,y] and labels
        #are [image_ids]
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
        pickle_file = 'notMNIST.pickle'

        try:
          f = open(pickle_file, 'wb')
          save = {
            'whole_train_dataset': wholeTrainDataset,
            'train_labels': wholeTrainLabels,
            'whole_valid_dataset': wholeValidDataset,
            'valid_labels': wholeValidLabels,
            'test_dataset_cur_genre': wholeTestDataset,
            'test_labels': wholeTestLabels,
            }
          pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
          f.close()
        except Exception as e:
          print('Unable to save data to', pickle_file, ':', e)
          raise


        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)

        # Problem 6
        # Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
        # Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
        # Optional question: train an off-the-shelf model on all the data!
        ### taking inspiration from http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#example-calibration-plot-compare-calibration-py
        from sklearn import dataset_cur_genres
        from sklearn.calibration import calibration_curve

        train_samples = 100  # number of samples used for training
        test_samples = 50   #number of samples for test

        #training patterns. x is input pattern, y is target pattern or label
        X_train = wholeTrainDataset[:train_samples]
        #fit function below expects to have a vector as the second dimension, not an array
        X_train = X_train.reshape([X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
        y_train = wholeTrainLabels[:train_samples]

        #test patterns
        X_test = wholeTestDataset[:test_samples]
        X_test = X_test.reshape([X_test.shape[0],X_test.shape[1]*X_test.shape[2]])
        y_test = wholeTestLabels[:test_samples]

        # Create classifier
        lr = LogisticRegression()

        #create plots
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")   

        #try to fit the training data
        lr.fit(X_train, y_train)

        #assess how confident (how probable it is correct) the model is at predicting test classifications
        prob_pos = lr.predict_proba(X_test)[:, 1]
            
        #fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        #ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name, ))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label='Logistic', histtype="step", lw=2)

        # ax1.set_ylabel("Fraction of positives")
        # ax1.set_ylim([-0.05, 1.05])
        # ax1.legend(loc="lower right")
        # ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

    # END MAIN
        
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
                print('Unable to save data to', set_filename, ':', e)
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
                print (strPickleFilename)
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
    threeDShuffleddataset_cur_genre = p_3ddataset_cur_genre[vPermutation,:,:]
    threeDShuffledLabels  = p_vLabels  [vPermutation]
    return threeDShuffleddataset_cur_genre, threeDShuffledLabels


if __name__=="__main__":
   main()