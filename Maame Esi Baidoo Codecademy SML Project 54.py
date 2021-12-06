def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Predicting Income with Random Forests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Loading data into a pandas dataframe
#Setting  delimiter to remove the spaces in front of the strings
income_data = pd.read_csv("income.csv", header = 0, delimiter = ", ")

#Inspecting the data
print(income_data.iloc[0])

#Adding new columns with re-encoded values from the sex and native-country columns
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

#Getting the labels for the model
labels = income_data[["income"]]
#Getting the columns to be used in the prediction
#data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex"]]
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

#Splitting the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

#Creating the random forest classifier
forest = RandomForestClassifier(random_state = 1)
forest.fit(train_data, train_labels)
#Removing the sex column from data as it is made of strings
#Finding the accuracy of the classifier
print("RF accuracy: " + str(forest.score(test_data, test_labels)))
#Adding the sex column to data by changing how the data is encoded in a new column above

#Transforming the values in the native-country column to int so it can be used in the classification
#Checking the different values in the column to know sure how to encode them
print(income_data["native-country"].value_counts())
#Since the majority of the data comes from "United-States", making a column where every row that contains "United-States" become 0 and any other country become 1 above

#Finding which features tend to be more relevant
print(forest.feature_importances_)

#Creating a decision tree classifier using same data to find out which classifier has a higher accuracy (changing the features in data to see how their accuracies also change along with different features)
tree = tree.DecisionTreeClassifier()
tree.fit(train_data, train_labels)
print("DT accuracy: " + str(tree.score(test_data, test_labels)))