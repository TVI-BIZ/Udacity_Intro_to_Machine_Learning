#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]
from sklearn.model_selection import train_test_split

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
# First variant. Train and predict on the same dataset
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)




# ## it's all yours from here forward!
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
# First variant. Train and predict on the same dataset
#clf_tree = clf_tree.fit(features,labels)
clf_tree = clf_tree.fit(features_train,labels_train)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#print(accuracy_score(clf_tree.predict(features_test,labels_test),labels_test))
predictions = clf_tree.predict(features_test,labels_test)
print("Predictions",predictions)
print("True Labels",labels_test)
print("Precision Score",precision_score(labels_test,predictions))
print("Recall Score",recall_score(labels_test,predictions))
print(clf_tree.score(features_test, labels_test))
print("We have",labels_test.count(1.0),"POI in training set")
print("How many members we have ",len(labels_test))

