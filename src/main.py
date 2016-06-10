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

def main():
    np.random.seed(133)
    
    #---------------------------------- BUILD DATA SETS ----------------------------------
    s_iTrainSize  = 200000
    s_iValid_size = 10000
    s_iTestSize   = 10000

    #get a list of genres for training and testing
    #using test for now to test training
    trainGenreNames, trainGenrePaths = listGenres('/media/kxstudio/LUSSIER/music/test/') #listGenres('/media/kxstudio/LUSSIER/music/train/')
    testGenreNames  = listGenres('/media/kxstudio/LUSSIER/music/test/')

    s_strListPickledTrainFilenames = maybe_pickle(trainGenrePaths, True)
    # s_strListPickledTestFilenames  = maybe_pickle(testGenreNames)

    if False:

        #call merge_dataset_cur_genres on data_sets and labels
        s_threeDValiddataset_cur_genre, s_vValidLabels, s_threeDTraindataset_cur_genre, s_vTrainLabels = merge_dataset_cur_genres(s_strListPickledTrainFilenames, s_iTrainSize, s_iValid_size)
        _,                  _,            s_threeDTestdataset_cur_genre,  s_vTestLabels  = merge_dataset_cur_genres(s_strListPickledTestFilenames, s_iTestSize)

        #print shapes for data sets and their respective labels. data sets are 3d arrays with [image_id,x,y] and labels
        #are [image_ids]
        print('Training:',   s_threeDTraindataset_cur_genre.shape, s_vTrainLabels.shape)
        print('Validation:', s_threeDValiddataset_cur_genre.shape, s_vValidLabels.shape)
        print('Testing:',    s_threeDTestdataset_cur_genre.shape,  s_vTestLabels.shape)


        s_threeDTraindataset_cur_genre, s_vTrainLabels = randomize(s_threeDTraindataset_cur_genre, s_vTrainLabels)
        s_threeDTestdataset_cur_genre,  s_vTestLabels  = randomize(s_threeDTestdataset_cur_genre,  s_vTestLabels)
        s_threeDValiddataset_cur_genre, s_vValidLabels = randomize(s_threeDValiddataset_cur_genre, s_vValidLabels)

        print(s_threeDTraindataset_cur_genre.shape)
        print(s_threeDTestdataset_cur_genre.shape)
        print(s_threeDValiddataset_cur_genre.shape)

        # Finally, let's save the data for later reuse:
        pickle_file = 'notMNIST.pickle'

        try:
          f = open(pickle_file, 'wb')
          save = {
            'train_dataset_cur_genre': s_threeDTraindataset_cur_genre,
            'train_labels': s_vTrainLabels,
            'valid_dataset_cur_genre': s_threeDValiddataset_cur_genre,
            'valid_labels': s_vValidLabels,
            'test_dataset_cur_genre': s_threeDTestdataset_cur_genre,
            'test_labels': s_vTestLabels,
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
        X_train = s_threeDTraindataset_cur_genre[:train_samples]
        #fit function below expects to have a vector as the second dimension, not an array
        X_train = X_train.reshape([X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
        y_train = s_vTrainLabels[:train_samples]

        #test patterns
        X_test = s_threeDTestdataset_cur_genre[:test_samples]
        X_test = X_test.reshape([X_test.shape[0],X_test.shape[1]*X_test.shape[2]])
        y_test = s_vTestLabels[:test_samples]

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

    #END MAIN
        
def listGenres(music_dir):
    dirs = os.listdir(music_dir)
    allAudioGenrePaths = []
    allAudioGenres = []
    for cur_dir in dirs:
        if not cur_dir.startswith('.') :
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
    sample_count = 1000
    dataset_cur_genre = np.ndarray(shape=(len(all_song_paths), sample_count), dtype=np.float32)
    
    songId = 0
    #for each song in the current genre
    for cur_song_file in all_song_paths:
        try:
            # convert current song to numpy array. when using images we were normalizing using pixel depth... should we do something different? Or pcmm is already [0,1]?
            # genre_path = '/media/kxstudio/LUSSIER/music/audiobooks/Alice_In_Wonderland_complete/'
            # song_path = genre_path +'Alice_In_Wonderland_ch_01.mp3'
            cur_song_pcm = songFile2pcm(cur_song_file)

            # only keep the first sample_count samples
            cur_song_pcm = cur_song_pcm[0:sample_count]

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


# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune s_iTrainSize as needed. The labels will be stored into a separate array of integers 0 through 9.

# Also create a validation dataset_cur_genre for hyperparameter tuning.
#from p_iNb_rows and p_iImg_size: 
#  return dataset_cur_genre:  an empty 3d array that is [p_iNb_rows, p_iImg_size, p_iImg_size]
#  return labels: an empty vector that is [p_iNb_rows]
def make_arrays(p_iNb_rows, p_iImg_size):
    if p_iNb_rows:
        dataset_cur_genre = np.ndarray((p_iNb_rows, p_iImg_size, p_iImg_size), dtype=np.float32)
        labels = np.ndarray(p_iNb_rows, dtype=np.int32)
    else:
        dataset_cur_genre, labels = None, None
    return dataset_cur_genre, labels

#p_strListPickle_files is an array containing the filenames of the pickled data
def merge_dataset_cur_genres(p_strListPickledFilenames, p_iTrainSize, p_iValidSize=0):
    iNum_classes = len(p_strListPickledFilenames)
    #make empty arrays for validation and training sets and labels
    valid_dataset_cur_genre, valid_labels = make_arrays(p_iValidSize, s_iImage_size)
    train_dataset_cur_genre, train_labels = make_arrays(p_iTrainSize, s_iImage_size)
    
    #number of items per class. // is an int division in python3, not sure in python2
    iNbrOfValidItemsPerClass = p_iValidSize // iNum_classes
    iNbrOfTrainItemPerClass = p_iTrainSize // iNum_classes
  
    #figure out useful indexes for the loop
    iStartValidId, iStartTrainId = 0, 0
    iEndValidId, iEndTrainId = iNbrOfValidItemsPerClass, iNbrOfTrainItemPerClass
    iEndListId = iNbrOfValidItemsPerClass+iNbrOfTrainItemPerClass
  
    #for each file in p_strListPickledFilenames
    for iPickleFileId, strPickleFilename in enumerate(p_strListPickledFilenames):    
        try:
            #open the file
            with open(strPickleFilename, 'rb') as f:
                print (strPickleFilename)
            #unpicke 3d array for current file
            threeDCurLetterSet = pickle.load(f)
            # let's shuffle the items to have random validation and training set. 
            # np.random.shuffle suffles only first dimension
            np.random.shuffle(threeDCurLetterSet)
        
            #if we asked for a validation set
            if valid_dataset_cur_genre is not None:
                #the first iNbrOfValidItemsPerClass items in letter_set are used for the validation set
                threeDValidItems = threeDCurLetterSet[:iNbrOfValidItemsPerClass, :, :]
                valid_dataset_cur_genre[iStartValidId:iEndValidId, :, :] = threeDValidItems
                #label all images with the current file id 
                valid_labels[iStartValidId:iEndValidId] = iPickleFileId
                #update ids for the train set
                iStartValidId += iNbrOfValidItemsPerClass
                iEndValidId   += iNbrOfValidItemsPerClass
                    
            #the rest of the items are used for the training set
            threeDTrainItems = threeDCurLetterSet[iNbrOfValidItemsPerClass:iEndListId, :, :]
            train_dataset_cur_genre[iStartTrainId:iEndTrainId, :, :] = threeDTrainItems
            train_labels[iStartTrainId:iEndTrainId] = iPickleFileId
            iStartTrainId += iNbrOfTrainItemPerClass
            iEndTrainId += iNbrOfTrainItemPerClass
        except Exception as e:
            print('Unable to process data from', strPickleFilename, ':', e)
            raise 
    return valid_dataset_cur_genre, valid_labels, train_dataset_cur_genre, train_labels
    #END OF merge_dataset_cur_genres



# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.
def randomize(p_3ddataset_cur_genre, p_vLabels):
    #with int x as parameter, np.random.permutation returns a random permutation of np.arange(x)
    vPermutation = np.random.permutation(p_vLabels.shape[0])
    threeDShuffleddataset_cur_genre = p_3ddataset_cur_genre[vPermutation,:,:]
    threeDShuffledLabels  = p_vLabels  [vPermutation]
    return threeDShuffleddataset_cur_genre, threeDShuffledLabels


if __name__=="__main__":
   main()