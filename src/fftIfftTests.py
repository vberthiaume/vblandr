import subprocess as sp
import scikits.audiolab
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile


#--CONVERT MP3 TO WAV------------------------------------------
song_path = '/home/gris/Music/vblandr/test_small/punk/07 Alkaline Trio - Only Love.mp3'
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
audio_array = audio_array[:2**16]
#--------------------------------------------------------------



#---- FFT THEN IFFT ------------------------
fft_output  = fft (audio_array)
ifft_output = ifft(fft_output).real

#this stuff is equivalent
#fft_output  = np.fft.rfft (audio_array, axis=0)
#print "fft_output is type", type(fft_output[0])
#ifft_output = np.fft.irfft(fft_output,  axis=0)
#print "ifft_output is type", type(ifft_output[0])



#--SAVE WAVE AS NEW FILE ----------------
ifft_output = np.round(ifft_output).astype('int16')
wavfile.write('/home/gris/Music/vblandr/testIfft.wav', 44100, ifft_output)
#scikits.audiolab.wavwrite(ifft_output, '/home/gris/Music/vblandr/testIfft.wav', fs=44100, enc='pcm16')
