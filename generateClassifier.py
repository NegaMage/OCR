# import the necessary packages
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import tensorflow as tf

# load the MNIST digits dataset
mnist = datasets.load_digits()

# take the MNIST data and construct the training and testing split
(trainData, trainLabels), (testData, testLabels) = tf.keras.datasets.mnist.load_data()
# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

#reshaping (xxx, 28, 28) into (xxx, 784)
trainData = trainData.reshape(54000, 784)
valData = valData.reshape(6000, 784)
testData = testData.reshape(10000, 784)

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = 3
# loop over various values of `k` for the k-Nearest Neighbor classifier
k = 3
# train the k-Nearest Neighbor classifier with the current value of `k`
model = KNeighborsClassifier(n_neighbors=k)
model.fit(trainData, trainLabels)
# evaluate the model and update the accuracies list
score = model.score(valData, valLabels)
print("k=%d, accuracy=%.2f%%" % (k, score * 100))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (3,score * 100))
# re-train our classifier using the best k value and predict the labels of the
# test data
model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainData, trainLabels)
predictions = model.predict(testData)
# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

#saving the trained classifier
joblib.dump(model, "digits_cls.pkl", compress=3)