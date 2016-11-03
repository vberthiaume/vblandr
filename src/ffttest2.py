import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

def fourier_series(x, y, wn, n=None):
    # get FFT
    fft_output = fft(y, n)
    # make new series
    y2 = ifft(fft_output).real
    # find constant y offset
    myfft[1:]=0
    c = ifft(myfft)[0]
    # remove c, apply factor of 2 and re apply c
    y2 = (y2-c)*2 + c

    plt.figure(num=None)
    plt.plot(x, y, x, y2)
    plt.show()



x = np.array([float(i) for i in range(0,360)])
print (x)
y = np.sin(2*np.pi/360*x) + np.sin(2*2*np.pi/360*x) + 5

fourier_series(x, y, 3, 360)