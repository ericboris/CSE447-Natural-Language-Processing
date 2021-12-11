import numpy as np

class XOR:
	def __init__(self, train_set, learning_rate=0.1, num_iters=10000):
		self.lr = learning_rate
		self.num_iters = num_iters
		self.m, self.n = train_set.shape

	def relu(self, x):
		''' Return the max of 0 and x. '''
		return np.maximum(0, x)

	def sign(self, x):
		''' Return 1 if x > 0 and -1 otherwise. '''
		return self.sigmoid(x) - 0.5

	def sigmoid(self, x):
		''' Return the sigmoid of x. '''
		return 1 / (1 + np.exp(-x))

	def feedforward(self, x):
		''' Return the prediction of a forward pass through the network. '''
		return self.sign(np.dot(self.relu(np.dot(x, self.w1) + self.b1), self.w2) + self.b2)

	def predict(self, x):
		''' Return a boolean vector of predictions on the network. '''
		return self.feedforward(x) > 0

	def train(self, x, y):	
		''' Train the model and return the weights and biases. '''
		# Let w1 and b1 be the hidden layer weights and biases, respectively.
		self.w1 = np.array([[-1.68, 0.76], [1,68, -0.76]])
		self.b1 = np.array([[-0.000047, -0.000047]])
		
		# Let w2 and b2 be the output layer weights and biases, respectively.
		self.w2 = np.array([[1.10], [1.97]])
		self.b2 = np.array([[-0.48]])

		# Train the model.
		yhat = self.feedforward(x)
		cost = self.binary_cross_entropy(y, yhat)

		# TODO Resume here
		print(f'YHAT: {yhat}')
		print(f'COST: {cost}')
	
	def binary_cross_entropy(self, y, yhat):
		''' Return the model's cost. '''
		return (-1 / self.m) * np.sum(y * np.log(yhat)) + ((1 - y) * np.log(1 - yhat))

	def backpropagation(self, x, y, yhat):
		''' Return the gradients for training. '''

if __name__ == '__main__':
	train_set = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
	y = np.array([[False], [True], [True], [False]])
	# Make y a column vector.
	y = (np.array(y).T)[:, np.newaxis]

	xor = XOR(train_set)	
	xor.train(train_set, y)
