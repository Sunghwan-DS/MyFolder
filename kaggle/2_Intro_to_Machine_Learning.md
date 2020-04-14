# Intro to Machine Learning

# 1. How Models Work

## 1.1 Introduction

Decision trees are easy to understand, and they are the basic building block for some of the best models in data science.

의사결정나무(Decision tree)

![image-20200413144808910](image-20200413144808910.png)

This step of capturing patterns from data is called **fitting** or **training** the model. The data used to **fit** the model is called the **training data**.

데이터로부터 패턴을 포착하는 이 단계를 fitting 또는 training the model 이라고 한다. 모델을 fit 하는데 사용되는 데이터를 training data라고 한다.



이해가 안가는 문장 : The details of how the model is fit (e.g. how to split up the data) is complex enough that we will save it for later.



## 1.2 Improving the Decision Tree

![image-20200413150004075](image-20200413150004075.png)

The biggest shortcoming of this model is that it doesn't capture most factors affecting home price, like number of bathrooms, lot size, location, etc.

You can capture more factors using a tree that has more "splits." These are called "deeper" trees.

![image-20200413150255500](image-20200413150255500.png)

The point at the bottom where we make a prediction is called a **leaf.**



# 2. Basic Data Exploration

## 2.1 Using Pandas to Get Familiar With Your Data

machine learning project

1. familiarize yourself with the data (Use the Pandas library)

   - Pandas is the primary tool data scientists use for exploring and manipulating data.

   - ```python
     import pandas as pd
     ```

   - The most important part of the Pandas library is the DataFrame.

   - ```python
     # save filepath to variable for easier access
     melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
     # read the data and store data in DataFrame titled melbourne_data
     melbourne_data = pd.read_csv(melbourne_file_path) 
     # print a summary of the data in Melbourne data
     melbourne_data.describe()
     ```

   - .csv = 엑셀 파일



## 2.2 Interpreting Data Description

![image-20200413152626375](image-20200413152626375.png)

이해가 안가는 문장 : The first number, the **count**, shows how many rows have non-missing values.

The second value is the mean, which is the average. Under that, std is the standard deviation, which measures how numerically spread out the values are. (mean : 평균, std : 표준편차)

To interpret the **min**, **25%**, **50%**, **75%** and **max** values, imagine sorting each column from lowest to highest value. The first (smallest) value is the min. If you go a quarter way through the list, you'll find a number that is bigger than 25% of the values and smaller than 75% of the values. That is the **25%** value (pronounced "25th percentile"). The 50th and 75th percentiles are defined analogously, and the **max** is the largest number.



## 2.3 Exercise: Explore Your Data

```python
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
```



### 2.3.1 Step 1: Loading Data

> Read the Iowa data file into a Pandas DataFrame called `home_data`.

```python
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = ____

# Call line below with no argument to check that you've loaded the data correctly
step_1.check()
```

```python
# Answer
home_data = pd.read_csv(iowa_file_path)
```



### 2.3.2 Step 2: Review The Data

> Use the command you learned to view summary statistics of the data. Then fill in variables to answer the following questions

```python
# Print summary statistics in next line
home_data.describe()
```

```python
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = ____

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = ____

# Checks your answers
step_2.check()
```

```python
# Answer
avg_lot_size = 10517
newest_home_age = 10
```



### 2.3.3 Think About Your Data

The newest house in your data isn't that new. A few potential explanations for this:

1. They haven't built new houses where this data was collected.
2. The data was collected a long time ago. Houses built after the data publication wouldn't show up.

If the reason is explanation #1 above, does that affect your trust in the model you build with this data? What about if it is reason #2?

How could you dig into the data to see which explanation is more plausible?





# 3. Your First Machine Learning Model

## 3.1 Selecting Data for Modeling

We'll start by picking a few variables using our intuition. Later courses will show you statistical techniques to automatically prioritize variables.

To choose variables/columns, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame (the bottom line of code below).

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
```

```
out:
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')
```

```python
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
```



There are many ways to select a subset of your data. The [Pandas Micro-Course](https://www.kaggle.com/learn/pandas) covers these in more depth, but we will focus on two approaches for now.

1. Dot notation, which we use to select the "prediction target"
2. Selecting with a column list, which we use to select the "features"



### 3.1.1 Selecting The Prediction Target

You can pull out a variable with **dot-notation**. This single column is stored in a **Series**, which is broadly like a DataFrame with only a single column of data.

We'll use the dot notation to select the column we want to predict, which is called the **prediction target**. By convention, the prediction target is called **y**. So the code we need to save the house prices in the Melbourne data is

```python
y = melbourne_data.Price
```



## 3.2 Choosing "Features"

We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).

Here is an example:

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
```

By convention, this data is called **X**.

```python
X = melbourne_data[melbourne_features]
```

Let's quickly review the data we'll be using to predict house prices using the `describe` method and the `head` method, which shows the top few rows.

```python
X.describe()
```

![image-20200414094119993](image-20200414094119993.png)

```python
X.head()
```

![image-20200414094301928](image-20200414094301928.png)

Visually checking your data with these commands is an important part of a data scientist's job. You'll frequently find surprises in the dataset that deserve further inspection.



## 3.3 Building Your Model

You will use the **scikit-learn** library to create your models. When coding, this library is written as **sklearn**, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.

The steps to building and using a model are:

- **Define:** What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- **Fit:** Capture patterns from provided data. This is the heart of modeling.
- **Predict:** Just what it sounds like
- **Evaluate**: Determine how accurate the model's predictions are.

Here is an example of defining a decision tree model with scikit-learn and fitting it with the features and target variable.

```python
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)
```

```
DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=1, splitter='best')
```



이해가 안가는 문장 : Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.



```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```

```
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
```





## 3.4 Exercise: Your First MAchine Learning Model

### 3.4.1 Recap

>So far, you have loaded your data and reviewed it with the following code. Run this cell to set up your coding environment where the previous step left off.

```python
# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")
```



### 3.4.2 Step 1: Specify Prediction Target

> Select the target variable, which corresponds to the sales price. Save this to a new variable called `y`. You'll need to print a list of the columns to find the name of the column you need.

```python
# print the list of columns in the dataset to find the name of the prediction target
home_data.columns
```

```
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'],
      dtype='object')
```

```python
# 가격은 SalePrice
y = home_data.SalePrice
```



### 3.4.3 Step 2: Create X

> Now you will create a DataFrame called `X` holding the predictive features.
>
> Since you want only some columns from the original data, you'll first create a list with the names of the columns you want in `X`.
>
> You'll use just the following columns in the list (you can copy and paste the whole list to save some typing, though you'll still need to add quotes):

```
* LotArea
* YearBuilt
* 1stFlrSF
* 2ndFlrSF
* FullBath
* BedroomAbvGr
* TotRmsAbvGrd
```

After you've created that list of features, use it to create the DataFrame that you'll use to fit the model.

```python
# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Check your answer
step_2.check()
```



#### Review Data

> Before building a model, take a quick look at **X** to verify it looks sensible

```python
# Review data
# print description or statistics from X
#print(X.describe())

# print the top few lines
#print(X.head())
```



### 3.4.4 Step 3: Specify and Fit Model

> Create a `DecisionTreeRegressor` and save it iowa_model. Ensure you've done the relevant import from sklearn to run this command.
>
> Then fit the model you just created using the data in `X` and `y` that you saved above.

```python
from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)

# Check your answer
step_3.check()
```



### 3.4.5 Step 4: Make Predictions

> Make predictions with the model's `predict` command using `X` as the data. Save the results to a variable called `predictions`.

```python
predictions = iowa_model.predict(X)
print(predictions)
```



### 3.4.6 Think About Your Results

> Use the `head` method to compare the top few predictions to the actual home values (in `y`) for those same homes. Anything surprising?

```python
# You can write code in this cell
Z = home_data[['SalePrice']]
print(Z.head())
print(y.head())
print(iowa_model.predict(X.head()))
```

```
   SalePrice
0     208500
1     181500
2     223500
3     140000
4     250000
0    208500
1    181500
2    223500
3    140000
4    250000
Name: SalePrice, dtype: int64
[208500. 181500. 223500. 140000. 250000.]
```





# 4. Model Validation

## 4.1 What is Model Validation

Many people make a huge mistake when measuring predictive accuracy. They make predictions with their *training data* and compare those predictions to the target values in the *training data*.

If you compare predicted and actual home values for 10,000 houses, you'll likely find mix of good and bad predictions. Looking through a list of 10,000 predicted and actual values would be pointless. We need to summarize this into a single metric.

There are many metrics for summarizing model quality, but we'll start with one called **Mean Absolute Error** (also called **MAE**). Let's break down this metric starting with the last word, error.

The prediction error for each house is:

```
error=actual−predicted
```

With the MAE metric, we take the absolute value of each error. We then take the average of those absolute errors. In plain English, it can be said as

```
On average, our predictions are off by about X.
```

To calculate MAE, we first need a model. 

```python
# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
```



Once we have a model, here is how we calculate the mean absolute error:

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```



## 4.2 The Problem with "In-Sample" Scores

The measure we just computed can be called an "in-sample" score. We used a single "sample" of houses for both building the model and evaluating it. Here's why this is bad.

Since this pattern was derived from the training data, the model will appear accurate in the training data.

But if this pattern doesn't hold when the model sees new data, the model would be very inaccurate when used in practice.

Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called **validation data**.



## 4.3 Coding It





# Intro to Machine Learning : Built-in Function 

```python
import pandas as pd

Name_file_path = '~~~.csv'
Name_data = pd.read_csr(Name_file_path)
Name_data.describe()
Name_data.columns
Name_data.Price
X.head()


from sklearn.tree import DecisionTreeRegressor

Name_model = DecisionTreeRegressor(random_state=1)
Name_model.fit(X, y)
Name_model.predict(X.head())


from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

