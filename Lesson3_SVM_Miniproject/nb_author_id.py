#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

# ## features_train and features_test are the features for the training
# ## and testing datasets, respectively
# ## labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# ########################################################
# ## your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

counter = 0
t0 = time()

# Linear Kernel Variant
# clf_svm = svm.SVC(kernel='linear')

# Updated Variant with better C and Kernel
clf_svm = svm.SVC(C=10000.0, kernel='rbf')

# Part dataset
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
clf_svm.fit(features_train, labels_train)
print("training time:", round(time() - t0, 3), "s")
t1 = time()
prediction = clf_svm.predict(features_test)
print(prediction[10], prediction[26], prediction[50])
print("Prediction time:", round(time() - t1, 3), "s")
# Chris email Counters
for i in range(len(prediction)):
    if prediction[i] > 0:
        counter += 1
print("Chris emails:", counter)
print(accuracy_score(clf_svm.predict(features_test), labels_test))
