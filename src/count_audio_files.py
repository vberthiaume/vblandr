import os, os.path

#import stuff from sms-tools
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sms-tools/software/models/'))
import utilFunctions as UF



def find_audio_files(dir):
	"return the path of all audio files (wav and mp3) in dir and its subfolders"
	#os.walk will go through each folder in dir (including dir), and for each folder will
	#	* set path to current path
	#	* set dirs to a list of directories that the current path contains
	#	* set files to a list of files that the current path contains
	allAudioFiles = []
	for path, dirs, files in os.walk(dir):
		for file in files:
			if not file.startswith('.') and (file.endswith('.wav') or file.endswith('.mp3')):
				allAudioFiles.append(path+file)
	return allAudioFiles

	

music_dir = '/media/kxstudio/Wisdom/music/Audiobooks'
allAudioFiles = find_audio_files(music_dir)
print "total audio files = ", len(allAudioFiles)

(fs, x) = UF.wavread(allAudioFiles[1])
print fs
#print "folder", music_dir, "contains", count_files_rec(music_dir), "files."




