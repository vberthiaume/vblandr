import os, os.path

# path joining version for other paths
music_dir = '/media/kxstudio/Wisdom/music/From Parts Unknown'
file_cnt = len([name for name in os.listdir(music_dir) if os.path.isfile(os.path.join(music_dir, name))])
print "folder", music_dir, "contains", file_cnt, "files."
