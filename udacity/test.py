import numpy as np

x = np.array([[1.,0.,0.],[1.,1.,1.],[1.,2.,2.],[1.,3.,3.],[1.,4.,4.]])	#array with 4 rows, 3 colums
x = x.T									#transpose array, so it becomes 3 rows, 4 cols
sum_rows = np.sum(x,0)							#vector containing the sum of rows
print (x /sum_rows)							#print original array with items divided by the sum of rows




