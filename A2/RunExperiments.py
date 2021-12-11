from MLE import get_data, MLE 
from constants import UNK

if __name__ == '__main__':
	# Load the training, validation, and test data sets.
	train_data = get_data('data/1b_benchmark.train.tokens')
	valid_data = get_data('data/1b_benchmark.dev.tokens')
	test_data = get_data('data/1b_benchmark.test.tokens')

	# Take half the training data.
	half_train_data = train_data[:int(len(train_data) / 2)]

	# Set the list of data to build and test models on.
	data_sets = [('TRAIN', train_data), ('VALID', valid_data), ('TEST', test_data)]

	# Show or don't show progress of the experiments. 
	verbose = False

	# Set the list of unk thresholds to the models to use.
	unk_thresholds = [3, 5]

	# Set the unsmoothed models to build and relevant parameters.
	# unsmoothed_models = [('UNI', MLE.unigram, 1), ('BI', MLE.bigram, 2), ('TRI', MLE.trigram, 3)]

	# Set the smoothed models to build and relevant parameters.
	# smoothed_models = [(MLE.li_ngram, 3)]
	
	# Set the lambda weights for the Linear Interpolation trigram models.
	lambda_weights = [
		[0.0, 0.1, 0.9], 	# Heavily weighted towards trigram.
		[0.1, 0.3, 0.6],
		[0.3, 0.4, 0.3],	# Balanced weights between trigram and unigram.
		[0.6, 0.3, 0.1],
		[0.9, 0.1, 0.0]]	# Heavily weighted towards unigram.

	# Use the following over the validation data set to find the best
	# Lambda values.
	# NOTE: This function not called below.
	# Results: Lambdas that are more heavily weighted towards trigram
	# 	with the least weighting towards unigram perform best.
	def get_lambdas():
		''' Explore the total range of len=3 lambdas to 1 decimal place. '''
		for l1 in range(11):
			for l2 in range(11 - l1): 
				l3 = 10 - (l1 + l2)
				l1, l2, l3 = l1 * 0.1, l2 * 0.1, l3 * 0.1
				yield [l1, l2, l3]

	def run(train_data):
		''' Perform the experiments after training the model on the given data. '''
		# Build, train, and score the models.
		for unk in unk_thresholds:
			# Build the model with the current unk threshold.
			mle = MLE(train_data, unk)
		
			# Train and score the unsmoothed models.	
			models = [
				('UNI', mle.unigram(verbose), 1), 
				('BI', mle.bigram(verbose), 2), 
				('TRI', mle.trigram(verbose), 3)]
			for model_name, model, n in models:
				for data_name, data in data_sets:
					# NOTE: The following is a hack to handle whole training vs half training
					if data_name == 'TRAIN':
						data = train_data
					print(f'UNK={unk}\t{model_name}\t{data_name}\t{mle.perplexity(data, model, n, verbose)}')
		
			# Train and score the smoothed models.
			n = 3
			for w in lambda_weights:
				model = mle.li_ngram(n, w, verbose)
				for data_name, data in data_sets:
					# NOTE: The following is a hack to handle whole training vs half training
					if data_name == 'TRAIN':
						data = train_data
					print(f'UNK={unk}\tLI_TRI\t{w}\t{data_name}\t{mle.perplexity(data, model, n, verbose)}')
	
	run(train_data)
	print('\n')
	run(half_train_data)
