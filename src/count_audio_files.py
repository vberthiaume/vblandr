import os, os.path


def count_files( dir ):
	return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

music_dir = '/media/kxstudio/Wisdom/music/From Parts Unknown'

print "folder", music_dir, "contains", count_files(music_dir), "files."
