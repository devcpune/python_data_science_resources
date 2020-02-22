#!/usr/bin/python3
import numpy as np
import pandas as pd
import time
from apyori import apriori

"""
A demo Python3 code demonstrating the Apriori algorithm being used to gain insights from a movie dataset. Tested on Python 3.6.7 and 3.7.6, should work on 3.x 

This example is picked from: https://medium.com/@fabio.italiano/the-apriori-algorithm-in-python-expanding-thors-fan-base-501950d55be9

"""

# Step 1: Read the dataset
print("-"*40)
print("Reading dataset....")
print("-"*40)
dataset_movies = pd.read_csv("./datasets/movies/movie_dataset.csv")
print("Completed. Number of records found: ", len(dataset_movies))

# Step 2: Convert data frame into list of lists
print("-"*40)
print("Transforming data frame into list of lists....")
print("-"*40)
dataset_as_list = []
for each_row in range(0, 200):
    row_as_list_of_strings = []
    for each_column in range(0, 20):
        single_entry_as_string = str(dataset_movies.values[each_row, each_column])
        row_as_list_of_strings.append(single_entry_as_string)
    dataset_as_list.append(row_as_list_of_strings)
print("Completed. First element from the dataset when converted into a list of lists is: ", dataset_as_list[0])

# Step 3: Generate association rules
print("-"*40)
print("Generating association rules....")
print("-"*40)
start = time.time()
association_rules = apriori(dataset_as_list, min_support = 0.0053, min_confidence = 0.20, min_lift = 3, min_length = 2)
end = time.time()
print("Time: ", end - start)
association_results = list(association_rules)

print("Completed. Number of association results generated: ", len(association_results))

print("-X-"*40)
print("The first association rule is: ", association_results[0])

