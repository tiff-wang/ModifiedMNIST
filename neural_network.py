import math
import random
import numpy as np 

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

# Make a matrix 
def matrix(m, n, fill=0.0):
    return np.zeros(shape=(m,n)) + fill

# Make a random matrix
def rand_matrix(m, n, a=0, b=1):
	return np.random.rand(m, n) * (b - a) + a

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        # self.ni = ni + 1 # +1 for bias node
        self.ni = ni
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        # default to range (-0.2, 0.2)
        self.wi = rand_matrix(self.ni, self.nh, -0.2, 0.2)
        self.wo = rand_matrix(self.nh, self.no, -0.2, 0.2)

        # last change in weights for momentum
        self.ci = matrix(self.ni, self.nh)
        self.co = matrix(self.nh, self.no)

    def propagate(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai = inputs

        # hidden activations
        for j in range(self.nh):
            inner_prod = np.dot(self.ai, self.wi.T[j])
            self.ah[j] = sigmoid(inner_prod)

        # output activations
        for k in range(self.no):
           	inner_prod = np.dot(self.ah, self.wo.T[j])
            self.ao[k] = sigmoid(inner_prod)

        return self.ao

if __name__: 
	pass