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


#TODO : figure out a test set


music_dir = '/media/kxstudio/LUSSIER/music/'
# path, dirs, files = os.walk(music_dir)
dirs = os.listdir(music_dir)
allAudioGenres = []
for cur_dir in dirs:
            if not cur_dir.startswith('.') :
                allAudioGenres.append(music_dir+cur_dir)

s_iNum_classes = len(allAudioGenres)

print (s_iNum_classes, allAudioGenres)

asdfa

np.random.seed(133)

def maybe_extract(filename, force=False):
  	root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  	if os.path.isdir(root) and not force:
		print('%s already present - Skipping extraction of %s.' % (root, filename))
  	else:
		print('Extracting data for %s. This may take a while. Please wait.' % root)
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
  	data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
  	if len(data_folders) != s_iNum_classes:
		raise Exception('Expected %d folders, one per class. Found %d instead.' % (s_iNum_classes, len(data_folders)))
  	print(data_folders)
  	return data_folders

print("s_strListExtractedTrainFolderNames: ")
s_strListExtractedTrainFolderNames = maybe_extract(strRawCompressedTrainSetFilename)
print("\ns_strListExtractedTestFolderNames: ")
s_strListExtractedTestFolderNames = maybe_extract(strRawCompressedTestSetFilename)

# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.

# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.

# A few images might not be readable, we'll just skip them.
s_iImage_size = 28  # Pixel width and height.
s_fPixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label, insuring you have at least min_num_images."""
  image_files = os.listdir(folder)
  #An ndarray is a (often fixed) multidimensional container of items of the same type and size
  #so here, we're building a 3d array with indexes (image index, x,y), and type float32
  dataset = np.ndarray(shape=(len(image_files), s_iImage_size, s_iImage_size), dtype=np.float32)
  image_index = 0
  #for each image in the current folder (A, B, etc)
  print(folder)
  for image in os.listdir(folder):
	#get the full image path
	image_file = os.path.join(folder, image)
	try:
	  #read image as a bunch of floats, and normalize those floats by using pixel_depth 
	  image_data = (ndimage.imread(image_file).astype(float) - s_fPixel_depth / 2) / s_fPixel_depth
	  #ensure image shape is standard
	  if image_data.shape != (s_iImage_size, s_iImage_size):
		raise Exception('Unexpected image shape: %s' % str(image_data.shape))
	  #and put it in the dataset
	  dataset[image_index, :, :] = image_data
	  image_index += 1
	except IOError as e:
	  print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
	
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
	raise Exception('Many fewer images than expected: %d < %d' %
					(num_images, min_num_images))
	
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
		
def maybe_pickle(p_strDataFolderNames, p_iMin_num_images_per_class, p_bForce=False):
  dataset_names = []
  #data_folders are either the train or test set. folders within those are A, B, etc
  for strCurFolderName in p_strDataFolderNames:
	#we will serialize those subfolders (A, B, etc), that's what pickling is
	strCurSetFilename = strCurFolderName + '.pickle'
	#add the name of the current pickled subfolder to the list
	dataset_names.append(strCurSetFilename)
	#if the pickled folder already exists, skip
	if os.path.exists(strCurSetFilename) and not p_bForce:
	  # You may override by setting force=True.
	  print('%s already present - Skipping pickling.' % strCurSetFilename)
	else:
	  #call the load_letter function def above 
	  print('Pickling %s.' % strCurSetFilename)
	  dataset = load_letter(strCurFolderName, p_iMin_num_images_per_class)
	  try:
		#and try to pickle it
		with open(strCurSetFilename, 'wb') as f:
		  pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
	  except Exception as e:
		print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

s_strListPickledTrainFilenames = maybe_pickle(s_strListExtractedTrainFolderNames, 45000)
s_strListPickledTestFilenames = maybe_pickle(s_strListExtractedTestFolderNames, 1800)

print("\ns_strListPickledTrainFilenames: ", s_strListPickledTrainFilenames)
print("\ns_strListPickledTestFilenames: ", s_strListPickledTestFilenames)

# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune s_iTrainSize as needed. The labels will be stored into a separate array of integers 0 through 9.

# Also create a validation dataset for hyperparameter tuning.
#from p_iNb_rows and p_iImg_size: 
#  return dataset:  an empty 3d array that is [p_iNb_rows, p_iImg_size, p_iImg_size]
#  return labels: an empty vector that is [p_iNb_rows]
def make_arrays(p_iNb_rows, p_iImg_size):
  if p_iNb_rows:
	dataset = np.ndarray((p_iNb_rows, p_iImg_size, p_iImg_size), dtype=np.float32)
	labels = np.ndarray(p_iNb_rows, dtype=np.int32)
  else:
	dataset, labels = None, None
  return dataset, labels

#p_strListPickle_files is an array containing the filenames of the pickled data
def merge_datasets(p_strListPickledFilenames, p_iTrainSize, p_iValidSize=0):
  iNum_classes = len(p_strListPickledFilenames)
  #make empty arrays for validation and training sets and labels
  valid_dataset, valid_labels = make_arrays(p_iValidSize, s_iImage_size)
  train_dataset, train_labels = make_arrays(p_iTrainSize, s_iImage_size)
	
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
		if valid_dataset is not None:
		  #the first iNbrOfValidItemsPerClass items in letter_set are used for the validation set
		  threeDValidItems = threeDCurLetterSet[:iNbrOfValidItemsPerClass, :, :]
		  valid_dataset[iStartValidId:iEndValidId, :, :] = threeDValidItems
		  #label all images with the current file id 
		  valid_labels[iStartValidId:iEndValidId] = iPickleFileId
		  #update ids for the train set
		  iStartValidId += iNbrOfValidItemsPerClass
		  iEndValidId   += iNbrOfValidItemsPerClass
					
		#the rest of the items are used for the training set
		threeDTrainItems = threeDCurLetterSet[iNbrOfValidItemsPerClass:iEndListId, :, :]
		train_dataset[iStartTrainId:iEndTrainId, :, :] = threeDTrainItems
		train_labels[iStartTrainId:iEndTrainId] = iPickleFileId
		iStartTrainId += iNbrOfTrainItemPerClass
		iEndTrainId += iNbrOfTrainItemPerClass
	except Exception as e:
	  print('Unable to process data from', strPickleFilename, ':', e)
	  raise	
  return valid_dataset, valid_labels, train_dataset, train_labels

#original values			
# s_iTrainSize = 200000
# s_iValid_size = 10000
# s_iTestSize = 10000
s_iTrainSize = 200000
s_iValid_size = 10000
s_iTestSize = 10000

#call merge_datasets on data_sets and labels
s_threeDValidDataset, s_vValidLabels, s_threeDTrainDataset, s_vTrainLabels = merge_datasets(s_strListPickledTrainFilenames, s_iTrainSize, s_iValid_size)
_,					_,			  s_threeDTestDataset,  s_vTestLabels  = merge_datasets(s_strListPickledTestFilenames, s_iTestSize)

#print shapes for data sets and their respective labels. data sets are 3d arrays with [image_id,x,y] and labels
#are [image_ids]
print('Training:',   s_threeDTrainDataset.shape, s_vTrainLabels.shape)
print('Validation:', s_threeDValidDataset.shape, s_vValidLabels.shape)
print('Testing:',	s_threeDTestDataset.shape,  s_vTestLabels.shape)

# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.
def randomize(p_3dDataset, p_vLabels):
	#with int x as parameter, np.random.permutation returns a random permutation of np.arange(x)
	vPermutation = np.random.permutation(p_vLabels.shape[0])
	threeDShuffledDataset = p_3dDataset[vPermutation,:,:]
	threeDShuffledLabels  = p_vLabels  [vPermutation]
	return threeDShuffledDataset, threeDShuffledLabels

s_threeDTrainDataset, s_vTrainLabels = randomize(s_threeDTrainDataset, s_vTrainLabels)
s_threeDTestDataset,  s_vTestLabels  = randomize(s_threeDTestDataset,  s_vTestLabels)
s_threeDValidDataset, s_vValidLabels = randomize(s_threeDValidDataset, s_vValidLabels)

print(s_threeDTrainDataset.shape)
print(s_threeDTestDataset.shape)
print(s_threeDValidDataset.shape)

# Finally, let's save the data for later reuse:
pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
	'train_dataset': s_threeDTrainDataset,
	'train_labels': s_vTrainLabels,
	'valid_dataset': s_threeDValidDataset,
	'valid_labels': s_vValidLabels,
	'test_dataset': s_threeDTestDataset,
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
from sklearn import datasets
from sklearn.calibration import calibration_curve

train_samples = 100  # number of samples used for training
test_samples = 50	#number of samples for test

#training patterns. x is input pattern, y is target pattern or label
X_train = s_threeDTrainDataset[:train_samples]
#fit function below expects to have a vector as the second dimension, not an array
X_train = X_train.reshape([X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
y_train = s_vTrainLabels[:train_samples]

#test patterns
X_test = s_threeDTestDataset[:test_samples]
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