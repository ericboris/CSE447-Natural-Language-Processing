import numpy as np
import math
import random
from functools import partial
from collections import defaultdict
from copy import deepcopy
from os import listdir
from os.path import join

def load_word_vectors(file_path, verbose=False):
	''' Return the lines from the given file as a word->vector mapping. '''
	if verbose:
		print('Loading data')
	data = {}
	with open(file_path) as f:
		for line in f:
			line = line.split()
			key = line[0]
			# Convert the list of strings to a numpy array of floats.
			val = np.array(list(map(float, line[1:])))
			data[key] = val
	return data

def mag(v):
	''' Return the magnitude of vector v. '''
	return np.sqrt(v.dot(v))

def cos(v1, v2):
	''' Compute cosine between the angle formed by vectors v1 and v2. 
		Return 0 if the two vectors are identical. '''
	return (v1.dot(v2)) / (mag(v1) * mag(v2)) if (v1 != v2).any() else 0

def similarity(v, data, verbose=False):
	''' Given a word vector v, return a cosine similarity mapping, w2 -> cos(v, v2)
		where w2 are the words in data and v2 is the word embedding for w2. '''
	if verbose:
		print(f'{w} similarity')

	d = defaultdict(float)
	for w2 in data:
		d[w2] = cos(v, data[w2]) 
	return d

def argmax(d, n=1, a=()):
	''' Return the top n keys in d with the maximum value. '''
	# Prevent side-effects.
	d = deepcopy(d)
	# Let the following be the list of keys to return.
	keys = []
	while len(keys) < n:
		k = max(d, key=d.get)
		d.pop(k)
		if k not in a:
			keys.append(k)
	return keys

def analogue(u1, u2, v1):
	''' Return a vector v2 = -u1 + u2 + v1. '''
	return -u1 + u2 + v1


if __name__ == '__main__':
	file_path = 'data/glove.6B/glove.6B.50d.txt'
	data = load_word_vectors(file_path, verbose=True)

	# 3.1
	words = ['dog', 'whale', 'before', 'however', 'fabricate']
	for w in words:
		v = data[w]
		sim = similarity(v, data)
		w2 = argmax(sim)[0]
		print(f'{w}\t{w2}\t{sim[w2]}')

	# 3.2
	analogies = (
		('dog', 'puppy', 'cat'),
		('speak', 'speaker', 'sing'),
		('france', 'french', 'england'),
		('france', 'wine', 'england'))
	for a1, a2, b1 in analogies:
		u1, u2, v1 = data[a1], data[a2], data[b1]
		v2 = analogue(u1, u2, v1)
		sim = similarity(v2, data)
		b2 = argmax(sim, 3, (a1, a2, b1))
		print(f'{a1}:{a2}::{b1}')
		for b in b2:
			print(f'\t{b}\t{sim[b]}')
