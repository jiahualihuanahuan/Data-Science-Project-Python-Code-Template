import numpy as np
from sklearn.model_selection import RandomizedSearchCV

param_dist={
    'n_neighbors':randint(1, 30),
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    'weights':['uniform', 'distance'],
    'leaf_size':randint(1, 60)
}

from sklearn.neighbors import KNeighborsClassifier
knn_cv = RandomizedSearchCV(KNeighborsClassifier(),param_dist,cv=5,scoring='accuracy',verbose=5,n_jobs=-1, n_iter=100)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('random_search', knn_cv)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))

# save grid search cross validation results to file
results = pd.DataFrame(knn_cv.cv_results_)
results.to_csv('credit default knn random search result.csv',index=False)
print('random search cross validation results saved')