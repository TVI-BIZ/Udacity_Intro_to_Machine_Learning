#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

counter = 0
for x,y in enron_data.items():
    data = [x,y]
    #print(data[1])
    # if data[0] == 'PRENTICE JAMES':
    #     print("Total Stock Value by",data[0],":",data[1]['total_stock_value'])
    # if data[0] == 'COLWELL WESLEY':
    #     print("Email Messages From",data[0],"to POI:", data[1]['from_this_person_to_poi'])
    # if data[0] == 'SKILLING JEFFREY K':
    #     print("Stock Options Exercised by", data[0], ":", data[1]['exercised_stock_options'])

    # Count a POI
    # for d in data:
        # if len(d) == 21:
        #     if data[1]['poi'] == 1:
        #         counter += 1

    # Find a SEO and salary volume
    # for d in data:
    #     if len(d) == 21:
    #         if data[1]['poi'] == 1:
    #             print("Director Fee:",data[1]['director_fees'])
    #             print("Salary:", data[1]['salary'],data[0],data[1]['total_payments'])
    # Total salary quantifed
    # for d in data:
    #     if len(d) == 21:
    #         if data[1]['salary'] != 'NaN':
    #             counter += 1
    # Total email Quantified
    # for d in data:
    #     if len(d) == 21:
    #         if data[1]['email_address'] != 'NaN':
    #             counter += 1

    for d in data:
        if len(d) == 21:
            if data[1]['total_payments'] == 'NaN' and data[1]['poi'] == 1 :
                counter += 1

print("Have no Total Payment",counter)
#print("Total email adress",counter)
#print("Total salary quantified ", counter)
# print("Toat POI in dataset: ", counter)






