import os, os.path


# ==========================================================================================
# # this is taken from ass 1. it counts folders, but I don't want to count folders. 
# def printAndCountFolders(filename):
#   	data_folders = [os.path.join(filename, d) for d in sorted(os.listdir(filename)) if os.path.isdir(os.path.join(filename, d))]
#   	print data_folders
#   	return len(data_folders)

# file_cnt = printAndCountFolders(music_dir)

# print "folder", music_dir, "contains", file_cnt, "folders."
# ==========================================================================================


# I want to access the filenames of all audio files within sub folders
# i need to ignore
# 	-files that start with .
# 	-files that end in something else than .mp3, wav, flac, wmv ???


def count_files_rec(dir):
	#lstNumberOfFilesContainedInEachFolder = [len(files) for path, dirs, files in os.walk(dir)]
	#return sum(lstNumberOfFilesContainedInEachFolder)

	#os.walk will go through each folder in dir (including dir), and for each folder will
	#	* set path to current path
	#	* set dirs to a list of directories that the current path contains
	#	* set files to a list of files that the current path contains

	allAudioFiles = []
	for path, dirs, files in os.walk(dir):
		# print path
		# print dirs
		# print files
		for file in files:
			if not file.startswith('.') and (file.endswith('.wav') or file.endswith('.mp3')):
				allAudioFiles.append(path+file)
	return allAudioFiles

	

music_dir = '/media/kxstudio/Wisdom/music/Audiobooks'
print "total audio files = ", len(count_files_rec(music_dir))
#print "folder", music_dir, "contains", count_files_rec(music_dir), "files."




