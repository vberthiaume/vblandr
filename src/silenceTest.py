import subprocess as sp
import scikits.audiolab
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import bisect
import matplotlib.pyplot as plt
import time

plt.rcParams['agg.path.chunksize'] = 10000

#--CONVERT MP3 TO WAV------------------------------------------
#song_path = '/home/gris/Music/vblandr/test_small/punk/07 Alkaline Trio - Only Love.mp3'
#song_path = '/mnt/c/Users/barth/Documents/vblandr/train_small/punk/01 - True North.mp3'
#song_path = '/mnt/c/Users/barth/Documents/vblandr/train_small/audiobook/Blaise_Pascal_-_Discours_sur_les_passions_de_l_amour.mp3'
song_path = '/home/gris/Music/vblandr/train_small/audiobook/Blaise_Pascal_-_Discours_sur_les_passions_de_l_amour.mp3'
command = [ 'ffmpeg',
        '-i', song_path,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',         
        '-ar', '44100', # sms tools wavread can only read 44100 Hz
        '-ac', '1',     # mono file     
        '-loglevel', 'quiet',           
        '-']            #instead of having an output file, using '-' sends it in the pipe. not actually sure how this works.
#run the command
pipe = sp.Popen(command, stdout=sp.PIPE)
#read the output into a numpy array
stdoutdata = pipe.stdout.read() 
audio_array = np.fromstring(stdoutdata, dtype=np.int16)
#--------------------------------------------------------------

def removeInitialSilence(cur_song_pcm):
    
    #start_time = time.clock()

    #raw
    #min = np.min(cur_song_pcm)
    #max = np.max(cur_song_pcm)
    #print ("raw min:", min, "max:", max)
    
    #using absolute value
    env = abs(cur_song_pcm)
    #min = np.min(env)
    #max = np.max(env)
    #print ("abs min:", min, "max:", max)
 
    #float
    env = env.astype(np.float32)            #cast the array into float32
    #min = np.min(env)
    #max = np.max(env)
    #print ("float min:", min, "max:", max)
 
    #norm1
    max = np.max(env)
    env = np.multiply(env, 1.0 / max)     #convert int16 range into [-.5, .5], but really because of the abs we're already between [0,.5]
    min = np.min(env)
    max = np.max(env)
    print ("norm1 min:", min, "max:", max)
 
    #end_time = time.clock()
    #print ("time:", end_time - start_time)
    
    plt.plot(env)
    plt.show()
 
    #convolving as a way to do a fast moving average
    N = 100
    env = np.convolve(env, np.ones((N,))/N)[(N-1):]
    
    #first 44100 samples are silent. what is their max amplitude?
    #print np.max(env[:44100])
    
    #at 1.5s, we're clearly into audio, what is the max amplitude?
    #print "before .5s, max: ", np.max(env[:.5*44100])
    
    #here we're still in noise part
    #print "in vocal part, max: ", np.max(env[.625*44100])
    
    #detect first non-silent sample
    threshold = .00004
    
    endOfSilence = bisect.bisect(env,threshold)
    
    print "end of silence: ", endOfSilence

    #these don't work on hesse
    plt.plot(env)
    plt.show()

    return cur_song_pcm[endOfSilence:]

#---- REMOVE SILENCE --------------------
ifft_output = removeInitialSilence(audio_array)
#truncate to 1 sec
ifft_output = ifft_output[:1*44100]

#--SAVE WAVE AS NEW FILE ----------------
ifft_output = np.round(ifft_output).astype('int16')
wavfile.write('/home/gris/Music/vblandr/silenceTest.wav', 44100, ifft_output)
#wavfile.write('/mnt/c/Users/barth/Documents/vblandr/silenceTest.wav', 44100, ifft_output)






