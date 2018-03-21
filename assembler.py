import os 
import numpy as np
import pandas as pd

files = [file for file in os.listdir('Ensemble') if ".csv" in file]

res = pd.DataFrame()
count = 0

for file in files:
	op = pd.read_csv('Ensemble/' + file, header=None)
	op = op.iloc[1:]
	op = op.drop([0], axis=1)
	res[count] = op[1]
	count+=1

res = np.asarray(res.as_matrix()).astype(int)

n = 0
final = []
for row in res:
	count = np.bincount(row)
	final.append(np.argmax(count))

final = np.array(final)

arr = np.arange(len(final))

np.savetxt('Ensemble/ensembler_output.csv', np.dstack((arr, final))[0], "%d,%d", header = "Id,Label", comments='')
