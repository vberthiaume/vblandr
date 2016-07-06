#ffmpeg stuff
import subprocess as sp
import numpy as np

song_path = "/Volumes/Untitled/music/train_small/audiobook/Blaise_Pascal_-_Discours_sur_les_passions_de_l_amour.mp3"

command = [ 'ffmpeg',
        '-i', song_path,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '44100', # sms tools wavread can only read 44100 Hz
        '-ac', '1', # mono file
        '-loglevel', 'quiet',
        '-']    #instead of having an output file, using '-' sends it in the pipe. not actually sure how this works.

#run the command
pipe = sp.Popen(command, stdout=sp.PIPE)
#read the output into a numpy array
stdoutdata = pipe.stdout.read()
audio_array = np.fromstring(stdoutdata, dtype="int16")

length = len(audio_array)
for i in range (lenght, length+100):
    print audio_array[i]