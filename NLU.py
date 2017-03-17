# Assignment 4
# Robin Mehta
# robimeht

from __future__ import division, print_function
import numpy as np 
import sys
import string
import operator
from collections import defaultdict
from string import punctuation
import math
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

#method found online (https://www.quora.com/How-do-I-remove-punctuation-from-a-Python-string)
def strip_punctuation(s):
	return ''.join(c for c in s if c not in punctuation)

#regex in this method found online (http://stackoverflow.com/questions/3368969/find-string-between-two-substrings)
def findMiddleText(start, end, line):
	foundWord = ""
	if line.find(start):
		startAndword = line[line.find(start):line.rfind(end)]
		foundWord = startAndword[len(start):]
		return foundWord

def findID(line):
	foundStr = ""
	if line.find('id=') != -1:
		foundStr = line[3:]
	return foundStr

def findName(line):
	foundStr = ""
	if line.find('name=') != -1:
		foundStr = line[5:]
	return foundStr

def parse_data_set(path):
	paragraphs = []
	sentences = [] #contains list of sentence, ID, and name
	with open(path,'r') as data:
		for par in data.read().split('\n\n'):
			if len(par) > 2:
				paragraphs.append(par)
				chunk = par.split('\n')
				sentence = ""
				ID = ""
				name = ""
				if len(chunk) > 2:
					if chunk[1] == "<class":
						sentence = chunk[0]
				#find Name and ID
				for line in chunk:
					if ID == "":
						ID = findID(line)
					if name == "":
						name = findName(line)
				if sentence != "":
					sentences.append(list([chunk[0], ID, name]))
	return sentences

def assignLabel(word, ID, name):
	label = ""
	if word == ID:
		label = 'B'
	if word in name:
		if word == name.split()[0]:
			label = 'B'
		else:
			label = 'I'
	if label == "":
		label = "O"
	return label

def make_new_features(data_set):
	feature_vectors = []
	print('Generating feature vectors...')

	token_value = [] #strings
	is_token_upper = [] #bools
	is_first_letter_upper = [] #bools
	token_length = [] #ints
	is_token_num = [] #bools
	targets = [] 

	# 3 additional features:
	is_token_in_paren = [] #bools
	is_token_last_word = []
	is_token_EECS = []

	test_sentences = parse_data_set(sys.argv[2])
	test_words = []
	for index, point in enumerate(test_sentences):
		sentence = point[0]
		for word in sentence.split():
			test_words.append(word)

	train_words = []
	train_only_words = []
	for index, point in enumerate(data_set):
		sentence = point[0]
		ID = point[1]
		name = point[2]
		for word in sentence.split():
			# take the ID and name and put it in a dict for train_words
			train_only_words.append(word)
			train_words.append(list([word, ID, name]))

	all_tokens = train_only_words + test_words
	le = preprocessing.LabelEncoder()
	le.fit(all_tokens)
	encoded_tokens = le.transform(all_tokens)

	for index, point in enumerate(train_words):
		word = point[0]
		ID = point[1]
		name = point[2]

		token_value.append(word)
		label = assignLabel(word, ID, name)
		targets.append(label)
		is_first_letter_upper.append(1 if word[0].isupper() else 0)
		is_token_upper.append(1 if word.isupper() else 0)
		token_length.append(len(word))
		is_token_num.append(1 if word.isdigit() else 0)

		# 3 additional features:
		is_token_in_paren.append(1 if word[0] == '(' and word[len(word)-1] == ')' else 0)
		is_token_EECS.append(1 if word == "EECS" else 0)
		is_token_last_word.append(1 if word[len(word)-1]=='.' or word[len(word)-1]=='!' else 0)

		feature_vector = []
		feature_vector.append(encoded_tokens[index])
		feature_vector.append(is_token_upper[index])
		feature_vector.append(is_first_letter_upper[index])
		feature_vector.append(token_length[index])
		feature_vector.append(is_token_num[index])
		feature_vector.append(is_token_in_paren[index])
		feature_vector.append(is_token_EECS[index])
		feature_vector.append(is_token_last_word[index])

		feature_vectors.append(feature_vector)
	return feature_vectors, targets

def main():
	train_set = parse_data_set(sys.argv[1])
	test_set = parse_data_set(sys.argv[2])
	training_vectors, targets = make_new_features(train_set)
	test_vectors, test_targets = make_new_features(test_set)
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(training_vectors, targets)
	
	total_seen = 0
	total_correct = 0

	for i, test_example in enumerate(test_vectors):
		correct = test_targets[i]
		pred = classifier.predict(np.array(test_example).reshape(1,-1))
		if str(pred[0]) == str(correct):
			total_correct += 1
			total_seen += 1
		else:
			total_seen += 1

		accuracy = (total_correct/total_seen)*100
	print("System Accuracy:", accuracy, "%")

if __name__ == '__main__':
	main()