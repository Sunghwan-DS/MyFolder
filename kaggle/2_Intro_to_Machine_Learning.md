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

