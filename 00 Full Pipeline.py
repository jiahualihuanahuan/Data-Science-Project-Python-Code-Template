# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:21:13 2020

data science project full pipeline template

@author: james li
"""



# import data
import pandas as pd
df = pd.read_csv('xxx.csv',index_col=0)


# separate X and y
X = df.drop('y', axis=1)
y = df['y']

# train and test data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# the pipeline: impute (median, mean, mode), transform (standarization, normalization) and encode (one-hot-encoding)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# for numeric columns, use scaler to transform
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
# for categorical columns, use one-hot-encoding to transform
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# compose transformer to create preprocessor
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# create a pipeline to combine preprocessor and model (any model)
from sklearn.ensemble import RandomForestClassifier
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', RandomForestClassifier())])

# fit the mode
model.fit(X_train, y_train)

# make prediction on test dataset
y_pred = model.predict(X_test)


# model selection
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
models = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]
for model in models:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])
    pipe.fit(X_train, y_train)   
    print(model)
    print("model score: %.3f" % pipe.score(X_test, y_test))


# hyper-parameter tunning
param_grid = { 
    'model__n_estimators': [200, 500],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth' : [4,5,6,7,8],
    'model__criterion' :['gini', 'entropy']}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(model, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)



# save the model to disk
from joblib import dump, load
dump(model, 'saved_model.joblib') 

saved_model = load('saved_model.joblib') 