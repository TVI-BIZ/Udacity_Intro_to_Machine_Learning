#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

# ## Task 1: Select what features you'll use.
# ## features_list is a list of strings, each of which is a feature name.
# ## The first feature must be "poi".
features_list = ['poi','salary','total_assets_value'] # You will need to use more features
# interested feature
# 'long_term_incentive','bonus','total_stock_value','director_fee'
# 'from_poi_to_this_person','shared_receipt_with_poi'
# also features
# 'email_address',

# ## Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# ## Task 2: Remove outliers
data_dict.pop('TOTAL')
# ####### Illustration We Find Max Salary, Average Salary, Del the TOTAL
dataItems = data_dict.items()
notInterestedPersons = []
for i in dataItems:
    if i[1]['email_address'] == 'NaN' and i[1]['director_fees'] == 'NaN' and i[1]['total_stock_value'] == 'NaN' and i[1]['salary'] == 'NaN':
        print("Candidate to move OUT from list",i[0])
        notInterestedPersons.append(i[0])
# Delete outliers from our dataset
for i in range(len(notInterestedPersons)):
    data_dict.pop(notInterestedPersons[i])
feature_matrix = []
valMessage = 0
for i in dataItems:
    # if i[1]['poi'] == 1 and i[1]['email_address'] == 'NaN':
    # if i[1]['total_payments'] == 'NaN' and i[1]['total_stock_value']== 'NaN' and i[1]['director_fees']== 'NaN':
    # if i[1]['total_stock_value'] == 'NaN' and i[1]['director_fees']== 'NaN':
    #     print(i[0],i[1])
    # if i[1]['total_stock_value'] != 'NaN' and i[1]['exercised_stock_options'] == 'NaN' and i[1]['restricted_stock'] == 'NaN':
     if i[1]['total_payments'] == 'NaN' and i[1]['total_stock_value'] == 'NaN' and i[1]['director_fees'] == 'NaN':
        print(i[0],i[1])


def sortedFeatureLists(dictData,feature):
    sortedList = []
    for i in dictData:
        if i[1][feature] == 'NaN':
            sortedList.append(0)
        else:
            sortedList.append(i[1][feature])
    sortedList = sorted(sortedList)
    nonZeroElems = np.count_nonzero(sortedList)
    print("Average ",feature,float(sum(sortedList))/nonZeroElems)
    print("Max ",feature,sorted(sortedList)[-1])
    print("Min ",feature,sorted(sortedList)[0])
    return sortedList

salaryList = sortedFeatureLists(dataItems,'salary')
#print(salaryList)
print(salaryList[-1])
print(min(x for x in salaryList if x > 0))
total_stock_valueList = sortedFeatureLists(dataItems,'total_stock_value')
total_stock_valueMin = min(x for x in total_stock_valueList if x > 0)
total_stock_valueMax = total_stock_valueList[-1]

director_feesList = sortedFeatureLists(dataItems,'director_fees')
director_feesMin = min(x for x in director_feesList if x > 0)
director_feesMax = director_feesList[-1]

total_paymentsList = sortedFeatureLists(dataItems,'total_payments')
total_paymentsMin = min(x for x in salaryList if x > 0)
total_paymentsMax = total_paymentsList[-1]





# ## Task 3: Create new feature(s)
# ## Store to my_dataset for easy export below.
my_dataset = data_dict
for i in my_dataset.items():
    if i[1]['total_payments'] == 'NaN':
        i[1]['total_payments'] = 0
    if i[1]['director_fees'] == 'NaN':
        i[1]['director_fees'] = 0
    if i[1]['total_stock_value'] == 'NaN':
        i[1]['total_stock_value'] = 0

for i in my_dataset.items():
    #print(i[1])
    total_Value = i[1]['total_payments'] + i[1]['director_fees'] + i[1]['total_stock_value']
    i[1]['total_assets_value'] = total_Value
    # if i[0] == 'GLISAN JR BEN F':
    #     print(i[1])

# ## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)


# ## Task 4: Try a varity of classifiers
# ## Please name your classifier clf for easy export below.
# ## Note that if you want to do PCA or other multi-stage operations,
# ## you'll need to use Pipelines. For more info:
# ## http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#Test Different Algorithms
clf_GaNB = GaussianNB()
clf_GaNB = clf_GaNB.fit(features_train,labels_train)
predict_GaNB = clf_GaNB.predict(features_test)
print("Gaussian Naive Bayes Accuracy",accuracy_score(predict_GaNB,labels_test))
clf_SVM = svm.SVC()
clf_SVM = clf_SVM.fit(features_train,labels_train)
predict_SVM = clf_SVM.predict(features_test)
print("Support Vector Machine Accuracy",accuracy_score(labels_test,predict_SVM))
clf_Tree = DecisionTreeClassifier()
clf_Tree = clf_Tree.fit(features_train, labels_train)
predict_Tree = clf_Tree.predict(features_train)
predict_Tree_val_set = clf_Tree.predict(features_test)
print("Decigion Tree Accuracy",accuracy_score(predict_Tree_val_set,labels_test))
# In this case for the my choosen parameters Gaussian the Best.

# ## Task 5: Tune your classifier to achieve better than .3 precision and recall
# ## using our testing script. Check the tester.py script in the final project
# ## folder for details on the evaluation method, especially the test_classifier
# ## function. Because of the small size of the dataset, the script uses
# ## stratified shuffle split cross validation. For more info:
# ## http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print("Precision Score Gaussian",precision_score(labels_test,predict_GaNB, average='macro'))
print("Recall Score Gaussian",recall_score(labels_test,predict_GaNB, average='weighted'))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_GaNB, my_dataset, features_list)