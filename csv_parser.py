import argparse
import numpy as np

parser = argparse.ArgumentParser(description='output csv parser')
parser.add_argument('-f',
                    help='file name')

args = parser.parse_args().f
output = np.loadtxt(args).astype(np.float32)
output = np.argmax(output, axis=1)


arr = np.arange(len(output))
filename = args.split('.')[0] + 'sub.csv'
print(filename)
np.savetxt(filename, np.dstack((arr, output))[0], "%d,%d", header = "Id,Label", comments='')

