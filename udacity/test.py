import numpy as np

A = np.array([[1.,0.,0.],[1.,1.,1.],[1.,2.,2.],[1.,2.,3.]])	#array with 4 rows, 3 colums
A = A.T						#transpose array, so it becomes 3 rows, 4 cols
sum_rows = np.sum(A,0)				#vector containing the sum of rows

#print (np.divide(A,sum_rows[np.newaxis,:]))

print (A /sum_rows)




