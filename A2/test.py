import numpy as np

class FFN:
	def __init__(self, w1, w2, b1, b2):
		self.w1 = w1
		self.w2 = w2
		self.b1 = b1
		self.b2 = b2

	def pred(self, x):
		z = self.relu(np.dot(self.w1, x) + self.b1)
		yhat = self.sign(np.dot(self.w2, z) + self.b2)
		return yhat

	def relu(self, x):
		return np.maximum(0, x)

	def sign(self, x):
		return 1 if x > 0 else -1

if __name__ == '__main__':
	w1 = np.array([[1, -1], [-1, 1]])
	w2 = np.array([[1, 1]])
	b1 = np.array([[0], [0]])
	b2 = np.array([[0]])

	X = [np.array([[1], [1]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[0], [0]])]

	ffn = FFN(w1, w2, b1, b2)
		
	for x in X:
		print(f'x=\n{x}\nres=\n{ffn.pred(x)}')	
