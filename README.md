# Data Science Project Python Code Template

# Goal
The goal in this respository is to create a python code template for any data science project using [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) and [sklearn](https://scikit-learn.org/stable/index.html).

## Read Data
Explore various ways of importing data from various formats into pandas DataFrame or numpy n dimensional array

- .csv
- .txt
- .excel
- .json
- .tar.gz

## Exploratory Data Analysis (EDA)

### Why EDA?
#### Reasons for the analyst
- Identify patterns and develop hypotheses.
- Test technical assumptions.
- Inform model selection and feature engineering.
- Build an intuition for the data.

#### Reasons for consumer of analysis
- Ensures delivery of technically-sound results.
- Ensures right question is being asked.
- Tests business assumptions.
- Provides context necessary for maximum applicability and value of results.
- Leads to insights that would otherwise not be found.

Use [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) to perform data wrangling

Use [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) to visualize the data

- histogram
- distribution plot
- line plot
- bar plot
- boxplot
- pie chart
- pairwise plot

Use pandas to check basic data statistics

## Separate X and y if needed

## Split data into train and test set

# Build transformation and machine learning pipeline 

## Numerical transformation (standardization, normalization etc.)
use StandardScaler() and MinMaxScaler() in Sci-kit learn to standardize the data
please check the [link](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) to explore more 
## Categorical transformation (one-hot-encoding)

## Build the machine learning model: classification models, regression models

## Model evaluation
Introduce various evaluation metrics/socres for different models

## Select the best model


## Perform hyper-parameter tunning
Introduce various hyper-parameter tunning techniques for all kinds of models

## Save the model to local disk
Use pickle and joblib library to save model onto local disk and reload the model later from the saved file

## Reload the model in a latter date
