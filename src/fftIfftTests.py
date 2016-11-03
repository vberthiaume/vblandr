import subprocess as sp
import scikits.audiolab
import numpy as np
from scipy.fftpack import fft, ifft



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
#--------------------------------------------------------------




audio_array = audio_array[:2**16]

fft_output  = fft (audio_array).real
ifft_output = ifft(fft_output).real

print (type(fft_output[0]))

#SAVE WAVE AS NEW FILE
filename = song_path = '/home/gris/Music/vblandr/test_small/punk/07 Alkaline Trio - Only Love.wav'
scikits.audiolab.wavwrite(audio_array, filename, fs=44100, enc='pcm16')