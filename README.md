# Classification And Regression Projects

Classification is the process of predicting the class with your input data. There are mostly two types of classification in Machine Learning Algorithms, one is supervised learning other is unsupervised learning. In briefly, if your dataset includes your target (label) features and you are going to predict these target, this means supervised learning. If your dataset doesn’t include target (label) feature, this means unsupervised learning.

In this post, I am going to apply supervised learning because I try to predict a purchase that customer will reorder.
Dataset

I am going to use Instacart dataset from Kaggle. Instacart is an American company that operates a grocery delivery and pick-up service in the United States and Canada with headquarters in San Francisco. The company offers service via a website and mobile app in 5,500 cities in all 50 U.S. states and Canadian provinces in partnership with over 350 retailers that have more than 25,000 grocery stores.

Instacart Dataset consists of below tables

    Departments
    Products
    Aisle
    Orders
    Order Product

Exploratory Data Analysis

Exploratory Data Analysis helps you that,

    understanding your data ,
    exploring structure of data ,
    recognising relationship between variables,

Briefly, Exploratory Data Analysis tell us almost everything about data.

Loading file

First, we need to import important packages that we are going to use throughout the entire project:

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
import warnings 
import csv 
import os 
import seaborn as sns warnings.filterwarnings('ignore')

Then we can read the CSV file we download from Kaggle. I save the CSV file in the same subdirectory with the Jupyter Notebook file I am working on. If those two files are not located on the same level, your path will be different.

airbnb = pd.read_csv('AirBnBNewYork.csv')

We can use airbnb.head() to view the first 5 rows of the table. In order to see the certain number of rows, put a number inside the parenthesis. For example, airbnb.head(10) will show you the first 10 rows of the table.
Data Exploration

Before we can employ any machine learning models, we have to clean up the data set to make sure there are no-nonsense or NULL values. Here are some of the codes I use to get a clearer sense of the data set and all of its values.

airbnb.describe()

This will show you the count, mean, std, min, 25% 50%, 75%, and a max of each column in the table.

Next, we need to see how many non-null counts in each column. If the number of non-null values equal to the number of the rows on the table, there are no null values that need to be taken care of in that column. However, be aware that the column may still contain some nonsense values or values that are not supposed to be in it.

airbnb.info()<class 'pandas.core.frame.DataFrame'>
RangeIndex: 48895 entries, 0 to 48894
Data columns (total 16 columns):
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   id                              48895 non-null  int64  
 1   name                            48879 non-null  object 
 2   host_id                         48895 non-null  int64  
 3   host_name                       48874 non-null  object 
 4   neighbourhood_group             48895 non-null  object 
 5   neighbourhood                   48895 non-null  object 
 6   latitude                        48895 non-null  float64
 7   longitude                       48895 non-null  float64
 8   room_type                       48895 non-null  object 
 9   price                           48895 non-null  int64  
 10  minimum_nights                  48895 non-null  int64  
 11  number_of_reviews               48895 non-null  int64  
 12  last_review                     38843 non-null  object 
 13  reviews_per_month               38843 non-null  float64
 14  calculated_host_listings_count  48895 non-null  int64  
 15  availability_365                48895 non-null  int64  
dtypes: float64(3), int64(7), object(6)
memory usage: 6.0+ MB

The first line of code will show the number of non-null values while the second one will count the number of null values in each column.

airbnb.isnull().sum()id                                    0
name                                 16
host_id                               0
host_name                            21
neighbourhood_group                   0
neighbourhood                         0
latitude                              0
longitude                             0
room_type                             0
price                                 0
minimum_nights                        0
number_of_reviews                     0
last_review                       10052
reviews_per_month                 10052
calculated_host_listings_count        0
availability_365                      0
dtype: int64

Next, in this data set, I decide to drop duplicate values.

airbnb.duplicated().sum() 
airbnb.drop_duplicates(inplace=True)

Then, in order to replace null values, I replace them with values that are appropriate for each column. My goal is to make sure that no column contains no null values.

airbnb.fillna({'reviews_per_month':0}, inplace=True)
airbnb.fillna({'name':"No Name"}, inplace=True)
airbnb.fillna({'host_name':"No Host Name"}, inplace=True)
airbnb.fillna({'last_review':"No Review"}, inplace=True)

On a side note, if you want to fill in null values, use values that share the same type with the rest in the column. For example, in the column reviews_per_month, I use 0 to be consistent with other numbers in the column instead of using “Not Available.” Consistency is key.
Data Visualization

Data visualization helps us visualize the table and see the role of each column in the table.

First, I’ll plot the number of neighbourhood groups to see which has more AirBnB housing.

# Which neighborhood_group has the most AirBnB?
airbnb['neighbourhood_group'].value_counts().sort_index().plot.barh()

Image for post
Image for post
![image](https://user-images.githubusercontent.com/66491543/110878447-12f6d580-8290-11eb-9ae9-3afcbaafc877.png)

Manhattan and Brooklyn are two places with the most Airbnb houses. The reason behind this is that most touristy destinations are in those two main places, while in the three other neighbourhood groups (Queens, Bronx, Staten Island) have more residential areas and fewer places to visit.

The next step is to get the correlation between different values in the table. The goal is to see which feature variables will be important in determining the price of New York Airbnb.

corr = airbnb.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
airbnb.columns

Image for post
Image for post

There are other method in .corr(). They are Pearson, Kendall, spearman and I just found out while reading on the Internet that they have another method called “callable”. Personally, I’d like to stick with Kendall but you should explore some other options to see which colour theme that catches your eyes.

In the project, I do some more data visualization with the table. For example, I include the charts for the neighborhood in each neighbourhood group, room types, the correlation between neighbourhood group and room types.

Given the longitude and latitude, I use codes to draw the rough map of all the houses in AirBnB.

plt.figure(figsize=(10,7)) sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.availability_365) 
plt.ioff()

Image for post
Image for post

This is the map of all houses based on their longitude, latitude, and sorted with their availability. You can also create some more maps on their neighbourhood groups, room type, etc.
Simple Linear Regression Model

First of all, we need to import all the packages we are sure to use.

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

After that, we need to encode columns that contain non-numerical values. The reason is that machine learning models are not capable of interpreting words, a.k.a non-numerical values.

labelencoder = LabelEncoder()airbnb['neighbourhood_group'] = labelencoder.fit_transform(airbnb['neighbourhood_group'])airbnb['neighbourhood'] = labelencoder.fit_transform(airbnb['neighbourhood'])airbnb['room_type'] = labelencoder.fit_transform(airbnb['room_type'])

I use LabelEncoder() in this case but you can also use OneHotEncoder for faster results and experiment. Just keep in mind that OneHotEncoder label values as 0 or 1 and will widen the table.

Next step is to split the data and categorize which columns are feature variables and which are target variables.

x = airbnb.iloc[:,[0,7]]
y = airbnb['price']#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.3,random_state=353)

The size of testing data is 30%. We usually split data into 30–70 or 20–80, but there are exceptions.

We also need to instantiate and fit the model.

# instantiate, fit
linreg = LinearRegression()
linreg.fit(x_train, y_train)

We can calculate the accuracy score for now. Just keep in mind that for more complex data set, we might need to use confusion matrix to check if this data is imbalanced or not. For imbalanced data, the accuracy score might be high, but it does not reflect the true accuracy score of the table. If we ever to encounter such a problem, we need to use other metrics and calculate different scores.

y_pred = linreg.predict(x_test) 
print('Accuracy on test set: {}'.format(linreg.score(x_test, y_test)))

My accuracy score was around 0.01, which is 1% and this might imply that either the metric is not suitable for this data set or the data is not impressive enough.

Now, we can calculate our predictions after training the model using a simple linear regression model.

predictions = linreg.predict(x_test)
error=pd.DataFrame(np.array(y_test).flatten(),columns=['actual'])
error['prediction']=np.array(predictions)
error.head(10)

Here is the final result:
Image for post
Image for post

Using the linear regression, we can predict the housing price in NYC Airbnb. As you can see from the result, our predictions are far from the real price. Perhaps, this metric is not the best for this kind of data set. My recommendation would be to try Random Forest and even Gradient Boost to see if the accuracy score will increase and if we will be able to make better predictions.
