import os, os.path

# def count_files( dir ):
# 	return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

# def count_files_rec(dir):
# 	"this will count recursively in sub folders"
# 	path, dirs, files = os.walk(dir).next()
# 	print files
# 	return len(files)

# # music_dir = '/media/kxstudio/Wisdom/music/From Parts Unknown'
music_dir = '/media/kxstudio/Wisdom/music/Audiobooks'

# print "folder", music_dir, "contains", count_files_rec(music_dir), "files."



def printAndCountFolders(filename):

  	data_folders = [os.path.join(filename, d) for d in sorted(os.listdir(filename)) if os.path.isdir(os.path.join(filename, d))]

  	print data_folders
  	return len(data_folders)

file_cnt = printAndCountFolders(music_dir)

print "folder", music_dir, "contains", file_cnt, "folders."