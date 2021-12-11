import numpy as np
from itertools import combinations_with_replacement as cwr
from itertools import combinations as com
import sys

def score(x, w):
	''' Return the score given list of word embedding vectors x and weight vector w. '''
	m = len(x)
	return (1 / m) * w.dot(np.sum(x, axis=0))

def scores(a, b, c, d, w):
	return score(a, w), score(b, w), score(c, w), score(d, w)

def ineq(a, b, c, d):
	p = a[0] > b[0]
	q = c[0] < d[0]
	r = p and q
	return p==True and q==True and r==False

def arrs():
	''' Yield a tuple of np arrays of the form (g, n, b, w)
	where g, n, b represent the words good, not, and bad, respectively and are (2, 1) vectors
	and where w represents the weights vector theta and is (1, 2) '''
	vals = [-1, 0, 1]
	i = 0
	for w1 in vals:
		for w2 in vals:
			w = np.array([[w1, w2]])	
			for g1 in vals:
				for g2 in vals:
					g = np.array([[g1], [g2]])
					for n1 in vals:
						for n2 in vals:
							n = np.array([[n1], [n2]])
							for b1 in vals:
								for b2 in vals:
									b = np.array([[b1], [b2]])
									yield g, n, b, w, i

def lists(g, n, b, w):
	''' Yield lists a, b, c, and d which are combinations of g, n, and b arrays. '''
	x = (g, n, b)
	j = 0
	for a_len in range(1, 2):
		for a in com(x, a_len):
			for b_len in range(1, 3):
				for b in com(x, b_len):
					for c_len in range(1, 2):
						for c in com(x, c_len):
							for d_len in range(1, 3):
								for d in com(x, d_len):
									sa, sb, sc, sd = scores(a, b, c, d, w)
									if ineq(sa, sb, sc, sd):
										print(f'w={w}\na={a}\nb={b}\nc={c}\nd={d}\nsa={sa}\nsb={sb}\nsc={sc}\nsd={sd}\ninq={ineq(sa, sb, sc, sd)}\n\n')
										sys.exit(0)

if __name__ == '__main__':
	good = np.array([[-1], [-1]])
	not_ = np.array([[-1], [1]])
	bad = np.array([[-1], [-1]])
	w = np.array([[1, 1]])

	a = (good, good)
	b = (not_, good)
	c = (bad, bad)
	d = (not_, bad)

	print(f'{scores(a, b, c, d, w)}')	


