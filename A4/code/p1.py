# Implementation of A4 Problem 1 - Sentence decoding using Viterbi algorithm.

from collections import defaultdict
from functools import partial
from copy import deepcopy
from math import log

MASK = '<mask>'

def main():
	print('Loading transition model')
	transition_model = load_transition_model('../data/p1/lm.txt')

	print('Loading masked lines')
	masked_lines = load_masked_lines('../data/p1/15pctmasked.txt')

	print('Decoding lines')
	decoded_lines = [decode(line, transition_model) for line in masked_lines]

	print('Saving decoded lines')
	save(decoded_lines, '../data/p1/decoded_lines.txt')	

	print('Done')

def load_transition_model(file_dir):
	''' Return a dictionary of transition probabilities that maps
		previous_state -> current_state -> probability, that is
		the probability that current_state follows previous state. '''
	transitions = defaultdict(partial(defaultdict, float))
	with open(file_dir) as f:
		for line in f:
			# This assumes a line consists of 3 space separated items.
			prev, curr, prob = line.split()

			transitions[prev][curr] = float(prob)
	return transitions

def load_masked_lines(file_dir):
	''' Return the masked lines as a list of lists of strings where each 
		interior list represents a line in the file and each string represents 
		a letter or special character. '''
	masked_lines = []
	with open(file_dir) as f:
		for line in f:
			masked_lines.append(line.split())
	return masked_lines

def save(data, file_dir):
	''' Save save the data as a space separated lines. '''
	with open(file_dir, 'wt') as f:
		for line in data:
			formatted_line = ' '.join(line) 
			formatted_line += '\n'
			f.write(formatted_line)

def score_and_backpointer(char, prev_col, transition_model):
	''' Return the score and backpointer to the previous char that maximizes 
		score(prev_char, char) + prev_score. ''' 
	score = 0
	backpointer = None
	for i, entry in enumerate(prev_col):
		prev_char, prev_score, _ = entry
		s = transition_model[prev_char][char] * prev_score
		if s >= score:
			score = s
			backpointer = i

	return score, backpointer

def decode(line, transition_model):
	''' Decode the given line, replacing MASK characters with the predicted
		replacement characters under the transition model using Viterbi algorithm. '''
	# Let dp be the dynamic programming table that is a list of lists of entries 
	# such that each entry is of the form (char, score, backpointer).
	dp = []

	# Start dp with the start char to make iteration straightforward.
	start_char = line[0]
	dp.append([(start_char, 1, None)])
		
	# Forward. Compute the scores and backpointers of each entry.
	for i in range(1, len(line)):
		col = []
		prev_col = dp[-1]
		if line[i] == MASK:	
			# Compute the transition scores and backpointers for each char
			# because the chars in this column are uncertain.
			for char in transition_model:
				score, backpointer = score_and_backpointer(char, prev_col, transition_model)
				col.append((char, score, backpointer))
		else:
		# Only compute one transition score and backpointer because the
			# char in this column is certain.
			char = line[i]
			score, backpointer = score_and_backpointer(char, prev_col, transition_model)
			col.append((char, score, backpointer))
		dp.append(col)

	# Backward. Follow the backpointers to construct the decoded line.
	decoded_line = []
	char, _, backpointer = dp[-1][0]
	decoded_line.append(char)
	for i in reversed(range(len(dp)-1)):
		char, _, backpointer = dp[i][backpointer]
		decoded_line.append(char)
	decoded_line.reverse()

	return decoded_line

if __name__ == '__main__':
	main()
