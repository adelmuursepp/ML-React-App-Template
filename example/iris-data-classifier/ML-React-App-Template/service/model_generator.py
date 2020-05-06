# Import libraries
import numpy as np
print('imported numpy')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import pandas as pd





#Otsustuspuud
from sklearn.tree import DecisionTreeClassifier

print('imported all')

data_table = pd.read_csv('postags_lemmas_levels_data.csv')
data_table = data_table.drop(['Unnamed: 0','tekstikood', 'filename'], 1)

print('read data')

# data_table.groupby("keeletase").A.plot(kind='kde')
#data_table.groupby("keeletase").A.hist(alpha=0.4)|

from sklearn.preprocessing import LabelEncoder
labelencoder_0 = LabelEncoder() #independent variable encoder
data_table.iloc[:,17] = labelencoder_0.fit_transform(data_table.iloc[:,17])

#Transforming values into percentages of total and splitting into target and features
features = data_table.loc[:, "A":"Z"]
target_var = data_table.loc[:, "keeletase"]

print('split to test and train')
# X_train, X_test, y_train, y_test =\
#     train_test_split(features.loc[:,'A':"Z"], target_var, test_size = 0.5, random_state=1111)



# Get the dataset
# dataset = datasets.load_iris()

# Split the dataset into features and labels
X = features
y = target_var

# Split the dataset into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)

# Build the classifier and make prediction
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
print('fit trainging data')
prediction = classifier.predict(X_test)

# Print the confusion matrix


# Save the model to disk
joblib.dump(classifier, 'classifier.joblib')





