import numpy as np

#SPLIT AND RECONSTRUCT FFT OUTPUT
x = np.array([5., 4., 3., 2., 1., 1., 2., 3., 4., 5.])
half = len(x)/2
x = x[:half]
x = np.append(x, np.zeros(len(x)))
for i in np.arange(half):
    # x[half+i] = x[half-i-1]
    x[i] = x[len(x)-i-1]

print x


# # SPLIT AND RECONSTRUCT FFT OUTPUT, REVERSING THE DROPPED HALF TO MAKE IT LOOK SIMMETRICAL
# x = np.array([5., 4., 3., 2., 1., 1., 2., 3., 4., 5.])
# half = len(x)/2
# x = x[:half]
# x = np.insert(x, np.zeros(len(x)), 0)
# for i in np.arange(half):
#     x[i] = x[len(x)-i-1]

# print x