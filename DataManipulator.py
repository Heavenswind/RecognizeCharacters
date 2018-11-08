from sklearn import tree
from sklearn import naive_bayes

#Read the the Information Label DS1
with open('ds1/ds1Info.csv', 'r') as file:
    infoData = [line.split(',') for line in file.read().split('\n')][1:]
infoData = [[element for element in row] for row in infoData]
infoFeatures = [d[:-1] for d in infoData]
infoLabels = [d[-1] for d in infoData]

#Read the the training set DS1
with open('ds1/ds1Train.csv', 'r') as file:
    trainData = [line.split(',') for line in file.read().split('\n')]
trainData = [[element for element in row] for row in trainData]
trainFeatures = [d[:-1] for d in trainData]
trainLabels = [d[-1] for d in trainData]

#Read the the validation set DS1
with open('ds1/ds1Val.csv', 'r') as file:
    validationData = [line.split(',') for line in file.read().split('\n')]
validationFeatures = [[element for element in row] for row in validationData]
validationFeatures = [d[:-1] for d in validationData]
validationLabels = [d[-1] for d in validationData]

'''
#Prints all the labels in a row.
print("All labels")
for row in infoLabels:
    print(row)
    
#Prints all the labels in InfoData.
print("All Training data")
for row in trainData:
    print(row)
    
#Prints all the labels in InfoData.
print("All validation data")
for row in validationData:
    print(row)
'''

#Prints all the labels (index/character) in InfoData.
print("All Labels")
for row in infoData:
    print(row)
print('\n')

#Row 0 of each array of each file
print("Info data row 0:", (infoData[0]))
print("Info feature 0: ", infoFeatures[0])
print("Info label  0: ", infoLabels[0])

print('\n')

print("Training feature row 1: ", trainFeatures[0])
print("Training label  0: ", trainLabels[0])
print("Training data row 1960: ", trainFeatures[1959])
print("Training label  1960: ", trainLabels[1959])
print('\n')

print("Validation feature row 1: ", validationFeatures[0])
print("Validation label row 1: ", validationLabels[0])
print("Validation feature row 514: ", validationFeatures[513])
print("Validation label row 514: ", validationLabels[513])
print('\n')


#Decision Tree Implementation
#classifier = tree.DecisionTreeClassifier()
#classifier.fit(trainFeatures, trainLabels)
#validationPredicted = classifier.predict(validationFeatures)


#Naive Bayes
classifier = naive_bayes.MultinomialNB()
#classifier.fit(trainFeatures, trainLabels)
#validation_predicted = classifier.predict(validationFeatures)

