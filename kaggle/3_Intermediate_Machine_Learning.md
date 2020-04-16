# Intermediate Machine Learning

# 1. Introduction

In this micro-course, you will accelerate your machine learning expertise by learning how to:

- tackle data types often found in real-world datasets (**missing values**, **categorical variables**),
- design **pipelines** to improve the quality of your machine learning code,
- use advanced techniques for model validation (**cross-validation**),
- build state-of-the-art models that are widely used to win Kaggle competitions (**XGBoost**), and
- avoid common and important data science mistakes (**leakage**).

## 1.1 Prerequisites

You're ready for this micro-course if you've built a machine learning model before, and you're familiar with topics such as [model validation](https://www.kaggle.com/dansbecker/model-validation), [underfitting and overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting), and [random forests](https://www.kaggle.com/dansbecker/random-forests).

If you're completely new to machine learning, please check out our [introductory micro-course](https://www.kaggle.com/learn/intro-to-machine-learning), which covers everything you need to prepare for this intermediate micro-course.



## 1.2 Exercise: Introduction

## 1.2.1 Setup

```python
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")  
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")
```

You will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course) to predict home prices in Iowa using 79 explanatory variables describing (almost) every aspect of the homes.

Run the next code cell without changes to load the training and validation features in `X_train` and `X_valid`, along with the prediction targets in `y_train` and `y_valid`. The test features are loaded in `X_test`. (*If you need to review **features** and **prediction targets**, please check out [this short tutorial](https://www.kaggle.com/dansbecker/your-first-machine-learning-model). To read about model **validation**, look [here](https://www.kaggle.com/dansbecker/model-validation). Alternatively, if you'd prefer to look through a full course to review all of these topics, start [here](https://www.kaggle.com/learn/machine-learning).)*



```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
```



### 1.2.2 Step 1: Evaluate several models

> The next code cell defines five different random forest models. Run this code cell without changes. (*To review **random forests**, look [here](https://www.kaggle.com/dansbecker/random-forests).*)

```python
from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
```

```python
from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```

```
Model 1 MAE: 24015
Model 2 MAE: 23740
Model 3 MAE: 23528
Model 4 MAE: 23996
Model 5 MAE: 23706
```

```python
# Fill in the best model
best_model = models[2]
```



### 1.2.3 Step 2: Generate test predictions

> Great. You know how to evaluate what makes an accurate model. Now it's time to go through the modeling process and make predictions. In the line below, create a Random Forest model with the variable name `my_model`.

```python
# Define a model
my_model = RandomForestRegressor() # Your code here
```



### 1.2.4 Step 3: Submit your results

