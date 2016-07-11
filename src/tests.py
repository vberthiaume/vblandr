
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

def main():
    
    song_filename = '/media/kxstudio/LUSSIER/music/train_small/metal/01-architects-early_grave.mp3'

    pickle_filename = song_filename + '.pickle'

    song_pcm = songFile2pcm(song_filename)
    SAMPLE_COUNT = 10 * 44100   # first 10 secs of audio
    song_pcm = song_pcm[0:SAMPLE_COUNT]     

    #maybe_pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(song_pcm, f, pickle.HIGHEST_PROTOCOL)

    
    with open(pickle_filename, 'rb') as f:
        unpickled_song_pcm = pickle.load(f)
        
    write_test_wav(unpickled_song_pcm, song_filename + '.wav')




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
