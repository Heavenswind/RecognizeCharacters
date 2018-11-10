import numpy as np
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#Read the the Information Label DS1
with open('ds1/ds1Info.csv', 'r') as file:
    infoData = [line.split(',') for line in file.read().split('\n')][1:]
infoData = [[element for element in row] for row in infoData]
infoFeatures = [d[:-1] for d in infoData]
infoLabels = [d[-1] for d in infoData]

#Read the the training set DS1
with open('ds1/ds1Train.csv', 'r') as file:
    trainData = [line.split(',') for line in file.read().split('\n')]
trainData = [[int(element) for element in row] for row in trainData]
trainFeatures = [d[:-1] for d in trainData]
trainLabels = [d[-1] for d in trainData]

#Read the the validation set DS1
with open('ds1/ds1Val.csv', 'r') as file:
    validationData = [line.split(',') for line in file.read().split('\n')]
validationData = [[int(element) for element in row] for row in validationData]
validationFeatures = [d[:-1] for d in validationData]
validationLabels = [d[-1] for d in validationData]


#Parsing the data to arrays
trainFeaturesArray = np.asarray(trainFeatures)
trainLabelsArray = np.asarray(trainLabels)
validationFeaturesArray = np.asarray(validationFeatures)



#Decision Tree Implementation
classifier = tree.DecisionTreeClassifier()
classifier.fit(trainFeaturesArray, trainLabelsArray)
validationPredicted = classifier.predict(validationFeaturesArray)
validationPredictedArray = np.asarray(validationPredicted)
#Decision Tree Predicted Labels
print('Predicted Labels for decision Tree for DS1')
#print(validationPredictedArray)
accuracy = accuracy_score(validationLabels, validationPredictedArray)
print('Accuracy of Decision Tree Classifier for DS1: ', "{0:.3%}".format(accuracy))
print('\n')
with open('ds1Val-dt.csv', 'w') as file:
 for i in range(len(validationPredictedArray)):
    file.write('%d,%d\n' % (i + 1, validationPredictedArray[i]))


#Naive Bayes Implementation
classifier = naive_bayes.BernoulliNB()
classifier.fit(trainFeaturesArray, trainLabelsArray)
validationPredicted = classifier.predict(validationFeaturesArray)
validationPredictedArray = np.asarray(validationPredicted)
#Predicted Naive Bayes Labels
print('Predicted Labels for Naive Bayes for DS1')
#print(validationPredictedArray)
accuracy = accuracy_score(validationLabels, validationPredictedArray)
print('Accuracy of Naive Bayes Classifier for DS1: ', "{0:.3%}".format(accuracy))
print('\n')
with open('ds1Val-nb.csv', 'w') as file:
 for i in range(len(validationPredictedArray)):
    file.write('%d,%d\n' % (i + 1, validationPredictedArray[i]))



#SVM Implementation
classifier = svm.SVC(gamma="scale", kernel="poly",  degree=8)
classifier.fit(trainFeaturesArray, trainLabelsArray)
validationPredicted = classifier.predict(validationFeaturesArray)
validationPredictedArray = np.asarray(validationPredicted)
#Predicted SVM Model Labels
print('Predicted Labels for SVM for DS1')
accuracy = accuracy_score(validationLabels, validationPredictedArray)
#print(validationPredictedArray)
print('Accuracy of SVM for DS1: ', "{0:.3%}".format(accuracy))
print("Classification report for classifier %s:\n%s\n"
      % (classifier, classification_report(validationLabels, validationPredictedArray)))
print("Confusion matrix:\n%s" % confusion_matrix(validationLabels, validationPredictedArray))
print('\n')
with open('ds1Val-3.csv', 'w') as file:
    for i in range(len(validationPredictedArray)):
        file.write('%d,%d\n' % (i + 1, validationPredictedArray[i]))


#################################################################################
##DS2
#Read the the Information Label DS2
with open('ds2/ds2Info.csv', 'r') as file:
    infoData = [line.split(',') for line in file.read().split('\n')][1:]
infoData = [[element for element in row] for row in infoData]
infoFeatures = [d[:-1] for d in infoData]
infoLabels = [d[-1] for d in infoData]

#Read the the training set DS2
with open('ds2/ds2Train.csv', 'r') as file:
    trainData = [line.split(',') for line in file.read().split('\n')]
trainData = [[int(element) for element in row] for row in trainData]
trainFeatures = [d[:-1] for d in trainData]
trainLabels = [d[-1] for d in trainData]

#Read the the validation set DS2
with open('ds2/ds2Val.csv', 'r') as file:
    validationData = [line.split(',') for line in file.read().split('\n')]
validationData = [[int(element) for element in row] for row in validationData]
validationFeatures = [d[:-1] for d in validationData]
validationLabels = [d[-1] for d in validationData]

# Parsing the data to arrays
trainFeaturesArray = np.asarray(trainFeatures)
trainLabelsArray = np.asarray(trainLabels)
validationFeaturesArray = np.asarray(validationFeatures)

# Decision Tree Implementation
classifier = tree.DecisionTreeClassifier()
classifier.fit(trainFeaturesArray, trainLabelsArray)
validationPredicted = classifier.predict(validationFeaturesArray)
validationPredictedArray = np.asarray(validationPredicted)
# Decision Tree Predicted Labels
print('Predicted Labels for decision Tree for DS2')
#print(validationPredictedArray)
accuracy = accuracy_score(validationLabels, validationPredictedArray)
print('Accuracy of Decision Tree Classifier for DS2: ', "{0:.3%}".format(accuracy))
print('\n')
with open('ds2Val-dt.csv', 'w') as file:
    for i in range(len(validationPredictedArray)):
        file.write('%d,%d\n' % (i + 1, validationPredictedArray[i]))


# Naive Bayes Implementation
classifier = naive_bayes.BernoulliNB()
classifier.fit(trainFeaturesArray, trainLabelsArray)
validationPredicted = classifier.predict(validationFeaturesArray)
validationPredictedArray = np.asarray(validationPredicted)
# Naive Bayes Predicted Labels
print('Predicted Labels for Naive Bayes for DS2')
#print(validationPredictedArray)
accuracy = accuracy_score(validationLabels, validationPredictedArray)
print('Accuracy of Naive Bayes Classifier for DS2: ', "{0:.3%}".format(accuracy))
print('\n')
with open('ds2Val-nb.csv', 'w') as file:
    for i in range(len(validationPredictedArray)):
        file.write('%d,%d\n' % (i + 1, validationPredictedArray[i]))



#SVM Implementation
classifier = svm.SVC(gamma="scale", kernel="poly", degree=8)
classifier.fit(trainFeaturesArray, trainLabelsArray)
validationPredicted = classifier.predict(validationFeaturesArray)
validationPredictedArray = np.asarray(validationPredicted)
#Predicted SVM Model Labels
print('Predicted Labels for SVM for DS2')
accuracy = accuracy_score(validationLabels, validationPredictedArray)
#print(validationPredictedArray)
print('Accuracy of SVM for DS2: ', "{0:.3%}".format(accuracy))

print("Classification report for classifier %s:\n%s\n"
      % (classifier, classification_report(validationLabels, validationPredictedArray)))
print("Confusion matrix:\n%s" % confusion_matrix(validationLabels, validationPredictedArray))

print('\n')
with open('ds2Val-3.csv', 'w') as file:
    for i in range(len(validationPredictedArray)):
        file.write('%d,%d\n' % (i + 1, validationPredictedArray[i]))


#https://pdfs.semanticscholar.org/1b1d/097926a976114afbe2026cc13f1bcbec72e1.pdf