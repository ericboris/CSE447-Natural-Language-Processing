import numpy as np

def relu(x):
	return np.maximum(0, x)

def score(x, w):
	''' Compute the score as given in q2. '''
	return w.dot(relu(x))

if __name__ == '__main__':
	# 2
	# Output all possible combinations of values
	# s.t. inequalities (1) and (2) are both true.
	vals = [-1, 1]
	i = 0
	for g1 in vals:
		for g2 in vals:
			g = np.array([[g1], [g2]])
			for n1 in vals:
				for n2 in vals:
					n = np.array([[n1], [n2]])
					for b1 in vals:
						for b2 in vals:
							b = np.array([[b1], [b2]])
							for w1 in vals:
								for w2 in vals:
									w = np.array([[w1, w2]])
									if score(g, w) > score(g+n, w) and score(b, w) < score(b+n, w):
										print(f'g={g}\nn={n}\nb={b}\nw={w}')
										print(f'{score(g, w)} > {score(g+n, w)} and {score(b, w)} < {score(b+n, w)}\n')
