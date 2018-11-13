import numpy as np
import pickle
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# Load a trained model
def load_trained_model(name):
    with open('model/' + name + '.pkl', 'rb') as fn:
        new_classifier = pickle.load(fn)
    return new_classifier


# Save a trained model
def save_trained_model(name, classifier):
    with open('model/' + name + '.pkl', 'wb') as fn:
        pickle.dump(classifier, fn)


# Save Output Labels
def save_output_labels(name, algorithm, prediction):
    with open(name + algorithm + '.csv', 'w') as fn:
        for i in range(len(prediction)):
            fn.write('%d,%d\n' % (i + 1, prediction[i]))


# Train a model with test data
def train(name):
    with open('' + name + '/' + name + 'Train.csv', 'r') as file:
        trainData = [line.split(',') for line in file.read().split('\n')]
    trainData = [[int(element) for element in row] for row in trainData]
    trainFeatures = [d[:-1] for d in trainData]
    trainLabels = [d[-1] for d in trainData]

    # Parsing the data to arrays
    trainFeaturesArray = np.asarray(trainFeatures)
    trainLabelsArray = np.asarray(trainLabels)

    # Decision Tree Implementation
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_features=0.80, max_depth=26)
    classifier.fit(trainFeaturesArray, trainLabelsArray)
    save_trained_model(name+'-dt', classifier)

    # Naive Bayes Implementation
    classifier = naive_bayes.BernoulliNB()
    classifier.fit(trainFeaturesArray, trainLabelsArray)

    save_trained_model(name+'-nb', classifier)

    # SVM Implementation
    classifier = svm.SVC(gamma="scale", kernel="poly", degree=8)
    classifier.fit(trainFeaturesArray, trainLabelsArray)
    save_trained_model(name+'-3', classifier)


def test(name):
    with open('' + name + '/' + name + 'Test.csv', 'r') as file:
        testData = [line.split(',') for line in file.read().split('\n')]
    testData = [[int(element) for element in row] for row in testData]
    testFeatures = [d for d in testData]

    # Parsing the data to arrays
    test_features_array = np.asarray(testFeatures)

    # Decision tree prediction
    dt_classifier = load_trained_model(name + '-dt')
    test_predicted = dt_classifier.predict(test_features_array)
    save_output_labels(name + 'Test-', 'dt', test_predicted)

    # Naive Bayes prediction
    nb_classifier = load_trained_model(name + '-nb')
    test_predicted = nb_classifier.predict(test_features_array)
    save_output_labels(name + 'Test-', 'nb', test_predicted)

    # SVM  prediction
    svm_classifier = load_trained_model(name + '-3')
    test_predicted = svm_classifier.predict(test_features_array)
    save_output_labels(name + 'Test-', '3', test_predicted)


# Validate our models with the given data-set
def validate(name):
    with open(name + '/' + name + 'Val.csv', 'r') as file:
        validationData = [line.split(',') for line in file.read().split('\n')]
    validationData = [[int(element) for element in row] for row in validationData]
    validationFeatures = [d[:-1] for d in validationData]
    validationLabels = [d[-1] for d in validationData]

    # Parse the array for prediction
    validation_array = np.asarray(validationFeatures)

    # Decision tree prediction
    dt_classifier = load_trained_model(name + '-dt')
    validation_predicted = dt_classifier.predict(validation_array)
    accuracy = accuracy_score(validationLabels, validation_predicted)
    print('Accuracy of Decision Tree Classifier for ' + name + ': ', "{0:.3%}".format(accuracy))
    print('\n')
    save_output_labels(name + 'Val-', 'dt', validation_predicted)

    # Naive Bayes prediction
    nb_classifier = load_trained_model(name + '-nb')
    validation_predicted = nb_classifier.predict(validation_array)
    accuracy = accuracy_score(validationLabels, validation_predicted)
    print('Accuracy of Naive Bayes Classifier for ' + name + ': ', "{0:.3%}".format(accuracy))
    print('\n')
    save_output_labels(name + 'Val-', 'nb', validation_predicted)

    # SVM  prediction
    svm_classifier = load_trained_model(name + '-3')
    validation_predicted = svm_classifier.predict(validation_array)
    accuracy = accuracy_score(validationLabels, validation_predicted)
    print('Accuracy of SVM Classifier for ' + name + ': ', "{0:.3%}".format(accuracy))
    save_output_labels(name + 'Val-', '3', validation_predicted)
    print('\n')


train('ds1')
validate('ds1')
test('ds1')
train('ds2')
validate('ds2')
test('ds2')
