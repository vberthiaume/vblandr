"""Softmax."""
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #sum e^x for all x, accross rows (dimention 0). this gives a vector with as many elements as x has columns
    sum_rows = np.sum(np.exp(x),0) 
    #return the normalization of e^x 
    return np.exp(x) / sum_rows

#x is a vector that ranges from -2 to 6 with increments of .1
x = np.arange(-2.0, 6.0, 0.1)

#vstack stacks its arguments as rows, so here scores will have 3 rows. 
#And ones_like produces an array of the same form as its argument, but filled with ones.
#so first row is x, second is a row of 1s, third is a row of .2s
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# Plot softmax curves. X axis is x, the arange. Y axis is the softmax values for the scores
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
