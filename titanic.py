import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

titanic_data = pd.read_csv('train.csv')
# no need for name or ticket number
# let's recode gender

# visualize correlations b/w variables; survival and fare; survival and pclass watch out for
sns.heatmap(titanic_data.corr(numeric_only=True), cmap="YlGnBu")
plt.show() 
# lower pclass (higher class) means higher survival 
# higher fare also means higher survival 

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2) # 20% of data in test set
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]): # uses these columns for stratification
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]
# distribution of classes remains consistent across both sets
# let's see if distribution makes sense
    
plt.subplot(1, 2, 1) # grid with 1 row, 2 columns, sets subplot to first position 
strat_train_set['Survived'].hist() # index 1 
strat_train_set['Pclass'].hist() # index 2
plt.subplot(1, 2, 2) 
strat_test_set['Survived'].hist() # index 2 
strat_test_set['Pclass'].hist() # index 1

plt.show() 
# we can see distribution is similar from train to test

# check missing values
strat_train_set.info()
# age and cabin have missing values

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer # fill missing values using mean

class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean") # imputing missing values using mean
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X
    
from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()

        column_names = ["C", "S", "Q", "N"]

        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        column_names = ["female", "male"]

        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        
        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"], axis=1, errors="ignore")
    
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("ageimputer", AgeImputer()),
                     ("featureencoder", FeatureEncoder()),
                     ("featuredropper", FeatureDropper())])

strat_train_set = pipeline.fit_transform(strat_train_set)
print(strat_train_set)
strat_train_set.info() # non-null values now

from sklearn.preprocessing import StandardScaler

X = strat_train_set.drop(['Survived'], axis=1)
Y = strat_train_set['Survived']

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
Y_data = Y.to_numpy() 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # split your data into 10 folds - test 9/10 folds against 1/10 

clf = RandomForestClassifier()
param_grid = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]} 
]
# does all combinations and gives us best performance

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data, Y_data)

final_clf = grid_search.best_estimator_
print(final_clf)

# do it for test set now
strat_test_set = pipeline.fit_transform(strat_test_set)

X_test = strat_test_set.drop(['Survived'], axis=1)
Y_test = strat_test_set['Survived']

scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
Y_data_test = Y_test.to_numpy()

final_clf.score(X_data_test, Y_data_test)

# titanic data
final_data = pipeline.fit_transform(titanic_data)

X_final = final_data.drop(['Survived'], axis=1)
Y_final = final_data['Survived']

scaler = StandardScaler()
X_final_data = scaler.fit_transform(X_final)
Y_final_data = Y_final.to_numpy()

prod_clf = RandomForestClassifier()
param_grid = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]} 
]
# does all combinations and gives us best performance

grid_search = GridSearchCV(prod_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_final_data, Y_final_data)

prod_final_clf = grid_search.best_estimator_

titanic_test_data = pd.read_csv("test.csv")
final_test_data = pipeline.fit_transform(titanic_test_data)

x_final_test = final_test_data
x_final_test = x_final_test.ffill()

scaler=StandardScaler()
X_data_final_test = scaler.fit_transform(x_final_test)
predictions = prod_final_clf.predict(X_data_final_test)

final_df = pd.DataFrame(titanic_test_data['PassengerId'])
final_df['Survived'] = predictions
final_df.to_csv("predictions.csv", index=False)