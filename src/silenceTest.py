import subprocess as sp
import scikits.audiolab
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import bisect
import matplotlib.pyplot as plt

#--CONVERT MP3 TO WAV------------------------------------------

#song_path = '/home/gris/Music/vblandr/test_small/punk/07 Alkaline Trio - Only Love.mp3'
song_path = '/mnt/c/Users/barth/Documents/vblandr/train_small/punk/01 - True North.mp3'
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
#audio_array = audio_array[:2**16]
#--------------------------------------------------------------

def removeInitialSilence(cur_song_pcm):
    #using absolute value
    env = abs(cur_song_pcm)
    #convolving as a way to do a fast moving average
    N = 100
    env = np.convolve(env, np.ones((N,))/N)[(N-1):]
    
    #first 44100 samples are silent. what is their max amplitude?
    #print np.max(env[:44100])
    
    #at 1.5s, we're clearly into audio, what is the max amplitude?
    print "before 1.5s, max: ", np.max(env[:66150])
    
    #here we're still in noise part
    print "in noise part, max: ", np.max(env[:45467])
    
    #these don't work on hesse
    #plt.plot(env)
    #plt.show()

    #detect first non-silent sample
    threshold = 1500#(2**16)/10
    
    endOfSilence = bisect.bisect(env,threshold)
    
    print "end of silence: ", endOfSilence
    return cur_song_pcm[endOfSilence:]

#---- REMOVE SILENCE --------------------
ifft_output = removeInitialSilence(audio_array)



#---- FFT THEN IFFT ------------------------
#fft_output  = fft (audio_array)
#ifft_output = ifft(fft_output).real

#this stuff is equivalent
#fft_output  = np.fft.rfft (audio_array, axis=0)
#print "fft_output is type", type(fft_output[0])
#ifft_output = np.fft.irfft(fft_output,  axis=0)
#print "ifft_output is type", type(ifft_output[0])



#--SAVE WAVE AS NEW FILE ----------------
ifft_output = np.round(ifft_output).astype('int16')
#wavfile.write('/home/gris/Music/vblandr/testIfft.wav', 44100, ifft_output)
wavfile.write('/mnt/c/Users/barth/Documents/vblandr/silenceTest.wav', 44100, ifft_output)


#scikits.audiolab.wavwrite(ifft_output, '/home/gris/Music/vblandr/testIfft.wav', fs=44100, enc='pcm16')




