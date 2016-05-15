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
	# return sum([len(files) for path, dirs, files in os.walk(dir)])
	lstNumberOfFilesContainedInEachFolder = [len(files) for path, dirs, files in os.walk(dir)]
	# oddly, this doesn't work. why? Isn't path, dirs and files returned from os.walk()?
	# print dirs
	return sum(lstNumberOfFilesContainedInEachFolder)

music_dir = '/media/kxstudio/Wisdom/music/Audiobooks'

print "folder", music_dir, "contains", count_files_rec(music_dir), "files."




