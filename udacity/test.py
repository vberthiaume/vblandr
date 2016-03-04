import numpy as np

#x = np.array([[1.,0.,0.],[1.,1.,1.],[1.,2.,2.],[1.,3.,3.],[1.,4.,4.]])	#array with 4 rows, 3 colums
#x = x.T						#transpose array, so it becomes 3 rows, 4 cols

x = np.arange(-2.0, 6.0, 0.1)
x = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

sum_rows = np.sum(x,0)				#vector containing the sum of rows

#print (np.divide(x,sum_rows[np.newaxis,:]))

print (x /sum_rows)




