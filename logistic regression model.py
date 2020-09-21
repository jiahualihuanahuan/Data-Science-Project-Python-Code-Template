import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
param_dist={
    'solver':['lbfgs', 'liblinear', 'sag', 'saga'],
    'C':np.logspace(-9, 9, num=25, base=10),
    'tol':np.logspace(-9, 9, num=25, base=10)
}

lr_cv = RandomizedSearchCV(LogisticRegression(),param_dist,cv=5,scoring='accuracy',verbose=1, n_jobs=-1, n_iter=100)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('random_search', lr_cv)
                             ])

my_pipeline.fit(X_train, y_train)


# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(lr_cv.best_params_))
print("Best score is {}".format(lr_cv.best_score_))

# save grid search cross validation results to file
results = pd.DataFrame(lr_cv.cv_results_)
results.to_csv('credit default logistic regression random search result.csv',index=False)
print('random search cross validation results saved')


# use the best hyper parameters to build the model
lr_clf = LogisticRegression(lr_cv.best_estimator_)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('clf', lr_clf)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)
y_train_pred = my_pipeline.predict(X_train)
y_pred = my_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score
print('training set model accuracy = ', accuracy_score(y_train, y_train_pred))

from sklearn.metrics import f1_score
print('training set model f1 score = ', f1_score(y_train, y_train_pred))

from sklearn.metrics import accuracy_score
print('testing set model accuracy = ', accuracy_score(y_test, y_pred))

from sklearn.metrics import f1_score
print('testing set model f1 score = ', f1_score(y_test, y_pred))


from sklearn.metrics import classification_report
print('testing set model classification report : \n ', classification_report(y_test, y_pred))