
#TESTING PRING AND MAIN

# def main():
#     allAudioGenres = ["punk", "poil", "classic"]
#     printStuff(allAudioGenres)
    
# def printStuff(stuff):
#     print stuff

# if __name__=="__main__":
#    main()


#PICKLING
from six.moves import cPickle as pickle
import subprocess as sp
import scikits.audiolab
import numpy as np

SAMPLE_COUNT = 10 * 44100   # first 10 secs of audio


def main():

    pickle1_filename = pickle_song('/media/kxstudio/LUSSIER/music/train_small/metal/01-architects-early_grave.mp3')
    pickle2_filename = pickle_song('/media/kxstudio/LUSSIER/music/train_small/metal/01. The Bitter End.mp3')

    all_pickle_files = [pickle1_filename, pickle2_filename]   
    all_pickle_filename = '/media/kxstudio/LUSSIER/music/train_small/TEST.pickle'
    
    #builddatasets
    f = open(all_pickle_filename, 'wb')
    save = {'all_pickle_files':    all_pickle_files}
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    with open(all_pickle_filename, 'rb') as f:
        save = pickle.load(f)
        unpickled_all_pickle_files   = save['all_pickle_files']

        for id, cur_pickled_filename in enumerate (unpickled_all_pickle_files):
            cur_unpickled_song_pcm = pickle.load( open( cur_pickled_filename, "rb" ) )
            write_test_wav(cur_unpickled_song_pcm, '/media/kxstudio/LUSSIER/music/train_small/TEST' + str(id) + '.wav')


def pickle_song (song_filename):
    pickle_filename = song_filename + '.pickle'
    song_pcm = songFile2pcm(song_filename)
    song_pcm = song_pcm[0:SAMPLE_COUNT] 
    #maybe_pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(song_pcm, f, pickle.HIGHEST_PROTOCOL)
    return pickle_filename



def songFile2pcm(song_path):
    command = [ 'ffmpeg',
            '-i', song_path,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '44100', # sms tools wavread can only read 44100 Hz
            '-ac', '1', # mono file
            '-loglevel', 'quiet',
            '-']    #instead of having an output file, using '-' sends it in the pipe. not actually sure how this works.
    pipe = sp.Popen(command, stdout=sp.PIPE)
    stdoutdata = pipe.stdout.read()
    audio_array = np.fromstring(stdoutdata, dtype="int16")
    return audio_array

def write_test_wav(cur_song_samples, filename):
    scikits.audiolab.wavwrite(cur_song_samples, filename, fs=44100, enc='pcm16')


if __name__=="__main__":
    main()
