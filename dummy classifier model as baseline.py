from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy="most_frequent")
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of test data, get predictions
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