import pandas

titanic = pandas.read_csv("train.csv")
print titanic.head()
print titanic.describe()

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print titanic.describe()

print titanic["Sex"].unique()
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "famele", "Sex"] = 1
print titanic["Sex"].unique()

print titanic["Embarked"].unique()
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
print titanic["Embarked"].unique()