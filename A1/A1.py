import random
import math
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import Counter

class DataHandler():
	def tokens_list(file_path, tokens_list=[]):
		''' Return a list of tokenized files in the given file path. '''
		for file_name in listdir(file_path):
			file_name = join(file_path, file_name)
			if isfile(file_name):
				tokens = DataHandler.tokenize(file_name)
				tokens_list.append(tokens)
		return tokens_list 

	def tokenize(file_name):
		''' Tokenize the file's contents. '''
		tokens = Counter() 
		with open(file_name) as o:
			for line in o:
				for word in line.split():
					tokens[word] += 1
		return tokens

	def actual_sentiments(file_path, sentiment_value, actual_sentiment_list=[]):
		''' Return a list of sentiment values of length equivalent 
		to the number of files in file_path. '''
		count_observations = len(listdir(file_path))
		tmp_sentiment_list = [sentiment_value] * count_observations
		actual_sentiment_list.extend(tmp_sentiment_list)
		return actual_sentiment_list

	def sentiment_lexicon(path, sentiment, sentiment_lexicon={}):
		''' Return a mapping of word -> sentiment value. '''
		e = 'ISO-8859-1'
		with open(path, encoding=e) as file_name:
			for line in file_name:
				# Ignore the file header.
				words = line.split()
				if len(words) == 1 and not words[0].startswith(';'):
					word = words[0]
					sentiment_lexicon[word] = sentiment
		return sentiment_lexicon

	def shuffle(list_a, list_b):
		''' Shuffle two lists such that both are shuffled into the same permutation. '''
		new_list_a = []
		new_list_b = []	
		indices = [i for i in range(len(list_a))]	
		random.shuffle(indices)
		for i in indices:
			new_list_a.append(list_a[i])
			new_list_b.append(list_b[i])
		return new_list_a, new_list_b	

	def split(data, test_size):
		''' Split and return test and training sets such that
		the test set is of size test_size and train set = data size - test_size. '''
		train = data[test_size:]
		test = data[:test_size]
		return (train, test)
	
	def column_vector(to_vector):
		''' Return the given list as an (n x 1) column vector. ''' 
		return (np.array(to_vector).T)[:,np.newaxis]

	def count_type_feature_matrix(tokens_list, sentiment_lexicon):
		''' Return a list of features = len(sentiments) s.t. for an
		arbitrary feature f_i for 0 <= i < len(sentiments)
		abs(f_i) = count(sentiment word appears in the observation)
		sign(f_i) = + is sentiment value is positive and - otherwise
		and f_i = 0 if sentiment word is not in the observation. '''
		feature_matrix = []
		for tokens in tokens_list:
			features = [0] * len(sentiment_lexicon)
			for i, word in enumerate(sentiment_lexicon.keys()):
				if word in tokens:
					sentiment_value = 1 if sentiment_lexicon[word] else -1
					features[i] += tokens[word] * sentiment_value
			feature_matrix.append(features)
		feature_matrix = np.array(feature_matrix)
		return feature_matrix

	def token_sum(tokens_list, verbose=True):
		''' Let the following be a count of each token appearing in all documents.
		token -> sum of times token appears across all documents '''
		if verbose:
			print('Getting Token Sum')
		token_sum = Counter()
		for tokens in tokens_list:
			for t in tokens:
				token_sum[t] += tokens[t]
		return token_sum

	def document_sum(tokens_list, token_sum, verbose=True):
		''' Let the following be a count of the number of documents in which each token appears.
		token -> sum of documents in which token appears '''
		if verbose:
			print('Getting Document Sum')
		document_sum = Counter()	
		for token in token_sum:
			for document in tokens_list:
				if token in document:
					document_sum[token] += 1
		return document_sum

	def tfidf(tokens_list, token_sum, document_sum, verbose=True):
		''' Return a tf idf features matrix of the observations in the tokens list. '''
		# Let the following be the total number of documents.
		num_documents = len(tokens_list)

		if verbose:
			print('Getting tfidf Features Matrix')	
		# Build the features matrix.
		features_matrix = []
		for observation in tokens_list:
			features = []
			for t in token_sum:
				tfidf = observation[t] * math.log(num_documents / (1 + document_sum[t]))
				features.append(tfidf)
			features_matrix.append(features)
		
		features_matrix = np.array(features_matrix)	
		return features_matrix

	def scale(x, min_val, max_val):
		''' Return a given value scaled between min and max vals. '''
		#return (abs(x) - min_val) / (max_val - min_val)
		numerator = (x - x.min(axis=0)) * (max_val - min_val)
		denominator = x.max(axis=0) - x.min(axis=0)
		denominator[denominator == 0] = 1
		return min_val + (numerator / denominator)

	def true_positive(actual_sentiment, predicted_sentiment):
		''' Return the count of true positives. '''
		return np.sum(np.logical_and(actual_sentiment==1, predicted_sentiment==1))

	def true_negative(actual_sentiment, predicted_sentiment):
		''' Return the count of true negative. '''
		return np.sum(np.logical_and(actual_sentiment==0, predicted_sentiment==0))

	def false_positive(actual_sentiment, predicted_sentiment):
		''' Return the count of false positives. '''
		return np.sum(np.logical_and(actual_sentiment==0, predicted_sentiment==1))

	def false_negative(actual_sentiment, predicted_sentiment):
		''' Return the count of false negatives. '''
		return np.sum(np.logical_and(actual_sentiment==1, predicted_sentiment==0))

	def accuracy(actual_sentiment, predicted_sentiment):
		''' Return the accuracy of the model. '''
		tp = DataHandler.true_positive(actual_sentiment, predicted_sentiment)
		tn = DataHandler.true_negative(actual_sentiment, predicted_sentiment)
		fp = DataHandler.false_positive(actual_sentiment, predicted_sentiment)
		fn = DataHandler.false_negative(actual_sentiment, predicted_sentiment)
		return (tp + tn) / (tp + tn + fp + fn)
	
	def precision(actual_sentiment, predicted_sentiment):
		''' Return the precision of the model. '''
		tp = DataHandler.true_positive(actual_sentiment, predicted_sentiment)
		fp = DataHandler.false_positive(actual_sentiment, predicted_sentiment)
		return tp / (fp + tp)

	def recall(actual_sentiment, predicted_sentiment):
		''' Return the recall of the model. '''
		tp = DataHandler.true_positive(actual_sentiment, predicted_sentiment)
		fn = DataHandler.false_negative(actual_sentiment, predicted_sentiment)
		return tp / (fn + tp)

	def f1(actual_sentiment, predicted_sentiment):
		''' Return the f1 score of the model. '''
		precision = DataHandler.precision(actual_sentiment, predicted_sentiment)
		recall = DataHandler.recall(actual_sentiment, predicted_sentiment)
		return 2 * precision * recall / (precision + recall)		
	
class SentimentLexicon():
	def predict(self, test_set):
		''' Predict the sentiments. '''
		weights = np.ones((test_set.shape[1], 1))
		predicted_output = np.dot(test_set, weights)
		predicted_labels = predicted_output > 0
		return predicted_labels

class LogisticRegression():
	def __init__(self, train_set, learning_rate=0.1, num_iterations=10000):
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations

		# Let the training set be a m x n matrix, s.t. m=observations and n=features.
		self.observations, self.features = train_set.shape

	def train(self, train_set, actual_output, verbose=False):
		''' Train the model. '''
		self.weights = np.zeros((self.features, 1))
		self.bias =	0 

		if verbose:
			print('Training Iteration: Cost')

		for i in range(self.num_iterations+1):
			predicted_output = self.sigmoid(np.dot(train_set, self.weights) + self.bias)
			cost = abs(self.cost(self.weights, self.bias, actual_output, predicted_output))
			dw, db = self.backpropagation(train_set, actual_output, predicted_output)
			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db	

			if verbose and i % 1000 == 0:
				print(f'\t{i}:\t{cost[0]}')

		return self.weights, self.bias
				
	def predict(self, test_set):
		''' Predict the sentiments. '''
		predicted_output = self.sigmoid(np.dot(test_set, self.weights) + self.bias)
		predicted_labels = predicted_output > 0.5
		return predicted_labels

	def sigmoid(self, z):
		''' Define a sigmoid function. '''
		# Where z = W^T * x_i + b.
		return 1 / (1 + np.exp(-z))

	def cost(self, weights, bias, actual_output, predicted_output):
		''' Return the cost of the model. '''
		return (-1 / self.observations) * np.sum(actual_output * np.log(predicted_output)) + ((1 - actual_output) * np.log(1 - predicted_output))

	def backpropagation(self, train_set, actual_output, predicted_output):
		''' Return the gradients for training. '''
		dw = 1 / self.observations * np.dot(train_set.T, (predicted_output - actual_output))
		db = 1 / self.observations * np.sum(predicted_output - actual_output)
		return dw, db
		
if __name__ == '__main__':
	# Define the sentiment values.
	pos_sentiment = True
	neg_sentiment = False

	# Define the paths to get the reviews.
	pos_review_path = 'data/txt_sentoken/pos/'
	neg_review_path = 'data/txt_sentoken/neg/'
	
	# Get the tokens_list.
	tokens_list = DataHandler.tokens_list(pos_review_path)
	tokens_list = DataHandler.tokens_list(neg_review_path, tokens_list)

	# Get the actual review sentiments.
	actual_sentiments = DataHandler.actual_sentiments(pos_review_path, pos_sentiment)
	actual_sentiments = DataHandler.actual_sentiments(neg_review_path, neg_sentiment, actual_sentiments)

	# Jointly shuffle the tokens and sentiments.	
	tokens_list, actual_sentiments = DataHandler.shuffle(tokens_list, actual_sentiments)

	# Split into training and test data.
	test_size = 400
	train_tokens_list, test_tokens_list = DataHandler.split(tokens_list, test_size)
	train_actual_sentiments, test_actual_sentiments = DataHandler.split(actual_sentiments, test_size)

	# Define the paths to get the sentiment lexicons.
	pos_sentiment_path = 'sentiment lexicon/positive-words.txt'
	neg_sentiment_path = 'sentiment lexicon/negative-words.txt'

	# Get the sentiment lexicon.
	sentiment_lexicon = DataHandler.sentiment_lexicon(pos_sentiment_path, pos_sentiment)
	sentiment_lexicon = DataHandler.sentiment_lexicon(neg_sentiment_path, neg_sentiment, sentiment_lexicon)

	# Prepare the data for the sentiment lexicon classifier.
	test_set = DataHandler.count_type_feature_matrix(test_tokens_list, sentiment_lexicon)	
	test_actual_sentiments = DataHandler.column_vector(test_actual_sentiments)

	# Run the sentiment lexicon classifier.
	print('Sentiment Lexicon Classifier')
	sentiment_lexicon_classifier = SentimentLexicon()
	predicted_sentiment = sentiment_lexicon_classifier.predict(test_set)
	sentiment_lexicon_accuracy = DataHandler.accuracy(test_actual_sentiments, predicted_sentiment)
	sentiment_lexicon_f1 = DataHandler.f1(test_actual_sentiments, predicted_sentiment)
	print(f'Accuracy: {sentiment_lexicon_accuracy}')
	print(f'F1: {sentiment_lexicon_f1}')

	# Run the logistic regression classifier.
	print('Logistic Regression Classifier')
	train_actual_sentiments = DataHandler.column_vector(train_actual_sentiments)

	# Build the necessary dictionaries for tfidf feature set.
	token_sum = DataHandler.token_sum(train_tokens_list)
	document_sum = DataHandler.document_sum(train_tokens_list, token_sum)
		
	# Create the training and test sets.
	train_set = DataHandler.tfidf(train_tokens_list, token_sum, document_sum)
	test_set = DataHandler.tfidf(test_tokens_list, token_sum, document_sum)
	train_set = DataHandler.scale(train_set, 0.01, 1)
	test_set = DataHandler.scale(test_set, 0.01, 1)
		
	# Train the classifier.
	logistic_regression_classifier = LogisticRegression(train_set)
	w, b = logistic_regression_classifier.train(train_set, train_actual_sentiments, verbose=True)	

	# Output the results.
	predicted_sentiment = logistic_regression_classifier.predict(test_set)
	logistic_regression_accuracy = DataHandler.accuracy(test_actual_sentiments, predicted_sentiment)
	logistic_regression_f1 = DataHandler.f1(test_actual_sentiments, predicted_sentiment)
	print(f'Accuracy: {logistic_regression_accuracy}')
	print(f'F1: {logistic_regression_f1}')
