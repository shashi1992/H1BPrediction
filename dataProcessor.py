from six.moves import cPickle as pickle
import numpy as np
import os
import csv

class DataMapping:
	def __init__(self):
		self.employerDict = {}
		self.employerCount = 0
		self.occupationDict = {}
		self.occupationCount = 0
		self.cityDict = {}
		self.cityCount = 0

	def Status(self, status):
		statusMappingDict = {'CERTIFIED-WITHDRAWN': 1, 'CERTIFIED': 1, 'DENIED': 0, 'REJECTED': 0}
		return statusMappingDict.get(status, 'NA')
		
	def Employer(self, employer):
		if not employer in self.employerDict:
			self.employerDict[employer] = self.employerCount
			self.employerCount += 1 
		return self.employerDict[employer]

	def Occupation(self, occupation):
		if not occupation in self.occupationDict:
			self.occupationDict[occupation] = self.occupationCount
			self.occupationCount += 1 
		return self.occupationDict[occupation]
	
	def City(self, city):
		if not city in self.cityDict:
			self.cityDict[city] = self.cityCount
			self.cityCount += 1 
		return self.cityDict[city]
	
	def Year(self, year):
		try:
			return int(year) - 2011
		except:
			return 'NA'

	def Fulltime(self, fulltime):
		return 1 if fulltime == 'Y' else 0

dm = DataMapping()

def loadData():
	if os.path.isfile('dataset.pickle'):
		with open('dataset.pickle', 'rb') as file:
			return pickle.load(file)
	global dm
	mapTable = [None, dm.Status, dm.Employer, dm.Occupation, None, dm.Fulltime, 0, dm.Year, dm.City, None, None]
	fullyLoaded = list()
	with open('h1b_kaggle.csv', 'r') as file:
		csvFile = csv.reader(file, delimiter = ',')
		rowCounter = 0
		for eachRow in csvFile:
			if rowCounter == 0: 
				rowCounter += 1
				continue
			if rowCounter % 100000 == 0: print('Loaded', rowCounter, 'rows')
			rowInfo = list()
			for index, element in enumerate(eachRow):
				mappingFunction = mapTable[index]
				if mappingFunction is None: continue
				elif mappingFunction == 0: rowInfo.append(element)
				else: rowInfo.append(mappingFunction(element))
			if len(rowInfo) != 0:
				if 'NA' in rowInfo:
					pass
					#print(rowCounter, rowInfo)
				else:
					fullyLoaded.append(rowInfo)
					rowCounter += 1
	dataset = np.array(fullyLoaded, dtype=np.float32)
	del fullyLoaded
	f = open('dataset.pickle', 'wb')
	pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
	f.close()
	return dataset

dataset = loadData()
print('Data set: ', dataset.shape)
print()
def categorize(data):
	length = data.shape[0]
	training = int(length * 0.8)
	valid = int(length * 0.1)
	trainingSet = data[:training]
	validSet = data[training:training+valid+1]
	testSet = data[training+valid+1:]
	return trainingSet, validSet, testSet

trainingSet, validSet, testSet = categorize(dataset)
print('Training set', trainingSet.shape)
print('Validation set', validSet.shape)
print('Test set', testSet.shape)
print()
del dataset

def splitDataLabel(data):
	data = data[:, 1:]
	label = data[:, 0]
	label = (np.arange(2) == label[:,None]).astype(np.float32)
	return data, label

trainingData, trainingLabel = splitDataLabel(trainingSet)
validData, validLabel = splitDataLabel(validSet)
testData, testLabel = splitDataLabel(testSet)
del trainingSet
del validSet
del testSet

print('Training set', trainingData.shape, trainingLabel.shape)
print('Validation set', validData.shape, validLabel.shape)
print('Test set', testData.shape, testLabel.shape)

np.random.seed(79)
def randomize(data, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = data[permutation, :]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

trainingData, trainingLabel = randomize(trainingData, trainingLabel)
validData, validLabel = randomize(validData, validLabel)
testData, testLabel = randomize(testData, testLabel)

f = open('seperatedData.pickle', 'wb')
pickle.dump({'trainingData': trainingData,
						'trainingLabel': trainingLabel,
						'validData': validData,
						'validLabel': validLabel,
						'testData': testData,
						'testLabel': testLabel}, f, pickle.HIGHEST_PROTOCOL)
f.close()

