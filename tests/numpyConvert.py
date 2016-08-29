import numpy as np

#x = np.int16(32766)
x = np.int16(-32766)
x_float = x.astype(np.float32)

#convert int16 range into [-1.0,1.0]
#x_float = x_float * 1.0/32767


#convert int16 range into [-.5, .5], then [-.5, .5] range into [0,1.0]
x_float = x_float * 1.0/65536
x_float = x_float +.5

print(x)
print(x_float)

