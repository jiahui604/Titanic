import pandas
import sklearn

titanic = pandas.read_csv("train.csv")
print titanic.head()
print titanic.describe()

#fill the Age "N/A" values with median
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print titanic.describe()

print titanic["Sex"].unique()
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
print titanic["Sex"].unique()

print titanic["Embarked"].unique()
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
print titanic["Embarked"].unique()

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
model = LinearRegression()

#created n folds, titanic.shape[0] is the first value (or number of records) from (m,n) format
kf = KFold(titanic.shape[0], n_folds = 4, random_state = 1)

predictions = []
for train, test in kf:
    #only take the rows from train data. Data.iloc purely integer-location based indexing for selection by position
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train]
    model.fit(train_predictors, train_target)
    test_predictions = model.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

import numpy as np
#predictions are in three separate arrays, concatenate them into one
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print accuracy
