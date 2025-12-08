# import packegs 

import numpy as np
import pandas as pd

from seaborn import load_dataset

# --------------------------

# ----1-----> load dataset
df = load_dataset('titanic')

# ----2----> EDA

# print(df.info())

# print(df.head())

# ----3----> preprocessing

# ----3.1----> fillna the null values
# print(df.isnull().sum())  # -> 177 + 688

df['age'] = df['age'].fillna(df['age'].mean())

# print(df.isnull().sum()) # -> 'age' 0 null

# -----3.2-----> drop the useless columns

df = df.drop(['alive', 'deck', 'adult_male', 'who', 'class', 'embarked'], axis=1)

# -------3.3-------> object -> numerical

# print(df['embark_town'].unique()) # -> 1.Southampton 2.Cherbourg 3.Queenstown

df['sex'].map({'male':0, 'female':1})
df['embark_town'].map({'Southampton':0, 'Cherbourg':1, 'Queenstown':2})

df = pd.get_dummies(df)

# drop the redundant columns after get dummies
df = df.drop(['sex_female', 'embark_town_Cherbourg'], axis=1)

# Data and label separation
label = df.survived
data = df.drop(['survived'], axis=1)

# train test splition

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=.3, random_state=42)

# train phase
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# hyperTiuning
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}

gnb = GaussianNB()
new1_gnb = GridSearchCV(gnb, param_grid, cv=5)

gnb.fit(X_train, y_train)
new1_gnb.fit(X_train, y_train)

# print(f"the Best parameters : {new1_gnb.best_params_}")

# print(gnb.score(X_test, y_test))
# print(new1_gnb.score(X_test,y_test))

















