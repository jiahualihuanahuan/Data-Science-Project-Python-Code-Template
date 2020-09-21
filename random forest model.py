from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist={
    'criterion':['gini', 'entropy'],
    'max_depth':randint(1, 30),
    'n_estimators': randint(1, 200),
    'min_samples_split':randint(1, 20),
    'min_samples_leaf':randint(1, 20),
    'max_features':randint(1, 20),
    'max_leaf_nodes':randint(1, 20)
    
}

rf_cv = RandomizedSearchCV(RandomForestClassifier(),param_dist,cv=5,scoring='f1',verbose=1, n_jobs=-1, n_iter=1000)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('random_search', rf_cv)
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(rf_cv.best_params_))
print("Best score is {}".format(rf_cv.best_score_))

# save grid search cross validation results to file
results = pd.DataFrame(rf_cv.cv_results_)
results.to_csv('credit default random forest random search result.csv',index=False)
print('random search cross validation results saved')

# use the best hyper parameters to build the model
rf_clf = rf_cv.best_estimator_

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('clf', rf_clf)
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