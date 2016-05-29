import os, os.path

#import stuff from sms-tools
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sms-tools/software/models/'))
import utilFunctions as UF

def find_audio_genres_and_files(dir):
    "return the path of all audio files (wav and mp3) in dir and its subfolders"
    #os.walk will go through each folder in dir (including dir), and for each folder will
    #   * set path to current path
    #   * set dirs to a list of directories that the current path contains
    #   * set files to a list of files that the current path contains

    #could try to use a big array and keep count of indexes in each sub array
    allAudioFiles = [6][30000]
    in_top_dir = True
    for path, dirs, files in os.walk(dir):
        #top sub folders will be used as labels
        if in_top_dir:
            #this will need to be a numpy array or whatever tensorflow uses
            allAudioGenres = list(dirs)
            # allAudioFiles = [6][100]
            in_top_dir = False

           

        #find label id

        #insert file in correct label id
        for file in files:
            if not file.startswith('.') and (file.endswith('.wav') or file.endswith('.mp3')):
                allAudioFiles.append(path+"/"+file)

    return (allAudioGenres, allAudioFiles)

# for now using .wavs, but most of my collection is in mp3, so need to figure how to use mp3s.

# I need to build a training set. First thing is to match input patterns with target patterns (labels).
# let's say the first folder is the label, and all audio files in sub dirs are
    
music_dir = '/media/kxstudio/LUSSIER/music/audiobooks'
allAudioGenres, allAudioFiles = find_audio_genres_and_files(music_dir)

# print "total audio files = ", len(allAudioFiles)
# print allAudioFiles[1]


# (fs, x) = UF.wavread(allAudioFiles[1])
# print fs

#print "folder", music_dir, "contains", count_files_rec(music_dir), "files."




