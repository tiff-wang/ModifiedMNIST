import numpy as np 

files = ['dataset/test_x.csv']

for file in files: 
	x = np.loadtxt(file, delimiter = ',')
	x[x < 235] = 0
	x = x / 255
	np.savetxt(file.split('.')[0] + '_proc.csv', x, delimiter = ',')

