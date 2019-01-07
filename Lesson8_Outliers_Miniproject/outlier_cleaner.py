#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    just_data = []

    # ## your code goes here
    for i in range(90):
        error = predictions[i][0]-net_worths[i][0]
        data_tuple = (ages[i][0],net_worths[i][0],error)
        just_data.append(data_tuple)
    # sorting
    sorted_by_error = sorted(just_data, key=lambda tup: tup[2])
    cleaned_data = sorted_by_error[:81]


    return cleaned_data

