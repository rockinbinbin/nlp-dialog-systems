# Assignment 4.3
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

def parse_file(path):
	dialogDict = dict() # make dict where key=dialogAct and val=list of all words associated
	numDialogDict = dict() # make dict where key=dialogAct and val=#of times it appears in train data
	uniqueDialogDict = dict() # make a dict where key=dialogAct and val=list of unique words associated
	countUniqueDialogDict = dict() # make a dict where key=dialogAct and val=# of unique words associated

	with open(path,'r') as data:
		for line in data.read().split("\n"):
			if len(line) > 2:
				prevStudentArray = []
				sentenceArray = []
				# get prev student words + next advisor dialog label
				if line.startswith("Student:"):
					for word in line.split(" "):
						if not word == "Student:" and not word == "":
							sentenceArray.append(word)
				prevStudentArray = sentenceArray
				if line.startswith("Advisor:"):
					dialogAct = findMiddleText("[", "]", line)
					if dialogAct != "social" and dialogAct != "push" and dialogAct != "pull":
						if not dialogAct in numDialogDict:
							numDialogDict[dialogAct] = 1
						else:
							numDialogDict[dialogAct] += 1
					else:
						dialogAct = ""
				# print(dialogAct)
				# print(prevStudentArray)

				if len(prevStudentArray) > 0 and len(dialogAct) > 0: #if both are written to
					if not dialogAct in dialogDict:
						dialogDict[dialogAct] = prevStudentArray
					else:
						for word in prevStudentArray:
							dialogDict[dialogAct].append(word)
					prevStudentArray = []
					dialogAct = ""

	for dialogAct in dialogDict:
		countUniqueDialogDict[dialogAct] = 0
		uniqueDialogDict[dialogAct] = list()
		for word in dialogDict[dialogAct]:
			if not word in uniqueDialogDict[dialogAct]:
				uniqueDialogDict[dialogAct].append(word)
		countUniqueDialogDict[dialogAct] = len(uniqueDialogDict[dialogAct])

	return dialogDict, numDialogDict, uniqueDialogDict, countUniqueDialogDict

def extractTestData(path):
	wordsAndTestIDsDict = dict()
	lineNum = 0
	with open(path, 'r') as data:
		for line in data.read().split("\n"):
			if line.startswith("Student:"):
				for word in line.split(" "):
					if word != "Student:":
						if not lineNum in wordsAndTestIDsDict:
							wordsAndTestIDsDict[lineNum] = [word]
						else:
							wordsAndTestIDsDict[lineNum].append(word)
			lineNum += 1
	return wordsAndTestIDsDict

def probabilitiesOfSenses(numDialogDict):
	probSensesDict = dict()
	totalNumSenses = 0
	for sense in numDialogDict:
		totalNumSenses += numDialogDict[sense]
	for sense in numDialogDict:
		probSensesDict[sense] = numDialogDict[sense] / totalNumSenses
	#print(probSensesDict)
	return probSensesDict

#iterates a dict of scores, IDs, and senses, finds argmax and best sense for ID
def keyOfMaxValue(scoreDict):
	solvedDict = dict()
	argMax = -999999999999
	labelSense = ""
	for ID in scoreDict:
		for scoreList in scoreDict[ID]:
			for sense in scoreList:
				score = scoreList[sense]
				if score > argMax:
					argMax = score
					labelSense = sense
		solvedDict[ID] = labelSense
	return solvedDict

def naiveBayesAddOneSmoothing(wordsAndTestIDsDict, sensesDict, numSensesDict, uniqueSensesDict, probSensesDict, outFile):
	scoreDict = dict()
	solvedDict = dict()
	outFile.write("Naive Bayes Add-One Smoothing")
	outFile.write("\n\n")

	totalSenseNum = 0
	for sense in uniqueSensesDict:
		totalSenseNum += len(uniqueSensesDict[sense])

	count = 0
	for ID in wordsAndTestIDsDict:
		for sense in sensesDict:
			# print(sense)
			total = 0
			for word in wordsAndTestIDsDict[ID]:
				# print(word)
				numerator = sensesDict[sense].count(word) + 1
				# print("num of times word appears in sense: ", numerator)
				denominator = numSensesDict[sense] + totalSenseNum
				# print("num times sense appears: ", numSensesDict[sense])
				total *= math.log((numerator / denominator), 2)
			score = total + math.log(probSensesDict[sense], 2)
			#print(score)
			if ID not in scoreDict:
				scoreDict[ID] = list()
			scoreDict[ID].append({sense:score})

		# count += 1
		# if count > 1:
		# 	break


	solvedDict = keyOfMaxValue(scoreDict)
	for ID in solvedDict:
		#outFile.write(str(ID))
		#outFile.write(" ")
		for word in wordsAndTestIDsDict[ID]:
			outFile.write(word)
			outFile.write(" ")
		outFile.write("\n")
		outFile.write("Label: ")
		outFile.write(str(solvedDict[ID]))
		outFile.write("\n\n")
	return solvedDict

def calculateAccuracies(solvedDict, testOutFileName, outFile):
	solutionsDict = dict() #dict where key=instanceID, value=list of context words (TEST DATA)
	with open(testOutFileName,'r') as data:
		lineCount = 0
		for line in data.read().split("\n"):
			if len(line) > 2:
				prevStudentArray = []
				sentenceArray = []
				# get prev student words + next advisor dialog label
				if line.startswith("Student:"):
					if len(dialogAct) > 0:
						solutionsDict[lineCount] = dialogAct
				if line.startswith("Advisor:"):
					dialogAct = findMiddleText("[", "]", line)
			lineCount += 1

	numCorrect = 0
	numTotal = 0
	for key in solvedDict:
		if key in solvedDict and key in solutionsDict:
			if solvedDict[key] == solutionsDict[key]:
				numCorrect += 1
			numTotal += 1
	outFile.write("\n")
	outFile.write("System Accuracy: ")
	accuracy = str((numCorrect/numTotal)*100)
	outFile.write(accuracy)
	outFile.write("%")
	outFile.write("\n\n")
	print("System Accuracy:", accuracy, "%")
	return accuracy

def main():
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	dialogDict, numDialogDict, uniqueDialogDict, countUniqueDialogDict = parse_file(train_path)
	probSensesDict = probabilitiesOfSenses(numDialogDict)
	wordsAndTestIDsDict = extractTestData(test_path)
	outFile = open('DialogAct.test.out', 'w')
	solvedDict = naiveBayesAddOneSmoothing(wordsAndTestIDsDict, dialogDict, numDialogDict, uniqueDialogDict, probSensesDict, outFile)
	calculateAccuracies(solvedDict, test_path, outFile)

if __name__ == '__main__':
	main()