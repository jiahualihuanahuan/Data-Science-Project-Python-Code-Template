import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from scipy.stats import randint

param_dist = {"max_depth": randint(1, 20),
              "max_features": randint(1, 22),
              "min_samples_leaf": randint(1, 20),
              "criterion": ["gini", "entropy"]}

tree_cv = RandomizedSearchCV(DecisionTreeClassifier(),param_dist,cv=5,scoring='f1',n_jobs=-1, verbose=5,n_iter=1000)



# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('grid_search', tree_cv)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# save grid search cross validation results to file
results = pd.DataFrame(tree_cv.cv_results_)
results.to_csv('credit default decision tree random search result.csv',index=False)
print('random search cross validation results saved')


# use the best hyper parameters to build the model
tree_clf = DecisionTreeClassifier(tree_cv.best_estimator_)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('clf', tree_clf)
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