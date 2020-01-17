#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:28:20 2019

@author: Rifat
Problem Statement: 
    "The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities.
     Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each 
     product at a particular store.Using this model, BigMart will try to understand the properties of products and stores which play a key role in 
     increasing sales."
     
Datasource: https://datahack.analyticsvidhya.com/contest/practice-problem-bigmart-sales-prediction/#data_dictionary

"""

import pandas as pd
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 8

#Importing the dataset. 
dataset = pd.read_csv("BigMartDataset.csv")

## DATA OBSERVATION #######################################################
print (dataset.shape)

#Check the missing missing values:
dataset.apply(lambda x: sum(x.isnull()))

#Analyze the data summary:
dataset.describe()

#Compute the Number of unique values in each column:
dataset.apply(lambda x: len(x.unique()))

#Seperate the  categorical features
categorical_features = [f for f in dataset.columns if dataset[f].dtype == object]
numeric_features = [f for f in dataset.columns if dataset[f].dtype != object]

#Exclude ID cols:
categorical_features = [f for f in categorical_features if f not in ['Item_Identifier','Outlet_Identifier']]

#Print frequency of categories
for col in categorical_features:
    print ('\nFrequency of Categories for varible %s'%col)
    print (dataset[col].value_counts())
    

## DATA CLEANING #######################################################
    
#Get a boolean variable specifying missing Item_Weight values
miss_bool = dataset['Item_Weight'].isnull()
print ('Original item data without weight: %d'% sum(miss_bool) )
       
#set item weight 0 where the item identifier has no weight information
dataset["Item_Weight"].fillna(0, inplace=True)

#Calculate the average weight per item:
item_avg_weight = dataset.pivot_table(values='Item_Weight', index='Item_Identifier' )

#Impute item weight data
dataset.loc[miss_bool,'Item_Weight'] = dataset.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x,'Item_Weight'])

from scipy.stats import mode

#Determing the mode for each outlet type
outlet_size_mode = dataset.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
print ('\nMode for each Outlet_Type:\n', outlet_size_mode)

#Get a boolean variable specifying missing Item_Weight values
miss_bool = dataset['Outlet_Size'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print ('\nMissing values before computation: %d'%sum(miss_bool) )
dataset.loc[miss_bool,'Outlet_Size'] = dataset.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print ('\nMissing values after computation: %d' %sum(dataset['Outlet_Size'].isnull()) )


#Determine average visibility of a product
visibility_avg = dataset.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
miss_bool = (dataset['Item_Visibility'] == 0)

print ('Total Number of 0 values in Item_Visibility: %d'%sum(miss_bool))
dataset.loc[miss_bool,'Item_Visibility'] = dataset.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.at[x,'Item_Visibility'])
print ('Total Number of 0 values in Item_Visibility after modification: %d'%sum(dataset['Item_Visibility'] == 0))


## FEATURE ENGINEERING #######################################################

#from the problem statement, dataset collected is 2013 sales data 
dataset['Outlet_Years'] = 2013 - dataset['Outlet_Establishment_Year']
dataset['Outlet_Years'].describe()


# Data mapping of the column Item_Fat_Content:
print ('\nOriginal Categories:\n',dataset['Item_Fat_Content'].value_counts())

dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
print ('\nModified Categories:\n',dataset['Item_Fat_Content'].value_counts())


## Apply ONE-HOT CODING to create dummy variable for categorial variable

from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()

#New variable for outlet
dataset['Outlet'] = LabelEncoder_X.fit_transform(dataset['Outlet_Identifier'])

cat_features = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type','Outlet_Type','Outlet']

for i in cat_features:
    dataset[i] = LabelEncoder_X.fit_transform(dataset[i])
    
dataset = pd.get_dummies(dataset, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type','Outlet_Type','Outlet'])    

dataset.dtypes
# remove unnecessary columns 
dataset.drop(['Outlet_Establishment_Year'],axis=1,inplace=True)

numeric_features.remove('Outlet_Establishment_Year')
numeric_features.remove('Item_Outlet_Sales')


#Spliting the dataset into training and test data
from sklearn.model_selection import train_test_split

X = dataset.loc[:, dataset.columns != 'Item_Outlet_Sales']
y = dataset[['Item_Outlet_Sales']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)

# Feature Scaling on numeric columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(X_train[numeric_features])
scaled = sc.transform(X_train[numeric_features])

for i, col in enumerate(numeric_features):
       X_train[col] = scaled[:,i]


scaled = sc.fit_transform(X_test[numeric_features])

for i, col in enumerate(numeric_features):
      X_test[col] = scaled[:,i]
      
#**************** END OF DATA ENGINEERING ***********


#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

from sklearn import metrics

def modelfit(alg, Xtrain,  predictors, y_train, IDcol):
    alg.fit(Xtrain[predictors], y_train)
    Xtrain_predictions = alg.predict(Xtrain[predictors])
    print ("RMSE of training data: %.4g" % np.sqrt(metrics.mean_squared_error(y_train.values, Xtrain_predictions)))
  
 ## Run models and evaluate ##########################################################

#linear model 
from sklearn.linear_model import LinearRegression
predictors = [x for x in X_train.columns if x not in [target]+IDcol]
print ("\nLinear Regression Model Report:\n")
alg1 = LinearRegression(normalize=True)
modelfit(alg1, X_train, predictors, y_train, IDcol)
# Predicting the test result
y_pred = alg1.predict(X_test[predictors])
print ("RMSE of test data: %.4g" % np.sqrt(metrics.mean_squared_error(y_test.values, y_pred)))

 
## Ridge Regression
from sklearn.linear_model import Ridge
print ("\nRidge Regression Model Report:\n")
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, X_train, predictors, y_train, IDcol)
# Predicting the test result
y_pred = alg2.predict(X_test[predictors])
print ("RMSE of test data: %.4g" % np.sqrt(metrics.mean_squared_error(y_test.values, y_pred)))

## Decision Tree
from sklearn.tree import DecisionTreeRegressor
print ("\nDecision Tree Regression Model Report:\n")
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, X_train,  predictors, y_train, IDcol)
# Predicting the test result
y_pred = alg3.predict(X_test[predictors])
print ("RMSE of test data: %.4g" % np.sqrt(metrics.mean_squared_error(y_test.values, y_pred)))


##Random Forest
from sklearn.ensemble import RandomForestRegressor
print ("\Random Forest Regression Model Report:\n")
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, X_train, predictors, y_train, IDcol)
# Predicting the test result
y_pred = alg5.predict(X_test[predictors])
print ("RMSE of test data: %.4g" % np.sqrt(metrics.mean_squared_error(y_test.values, y_pred)))

## xgboost
import xgboost as xgb
print ("\XGBOOST Model Report\n")
alg6 = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=10,
                       min_child_weight=1.5,
                       n_estimators=8200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=32,
                       silent=1)
modelfit(alg6, X_train, predictors, y_train, IDcol)

y_pred = alg6.predict(X_test[predictors])
print ("RMSE of test data: %.4g" % np.sqrt(metrics.mean_squared_error(y_test.values, y_pred)))
#coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
#coef6.plot(kind='bar', title='Feature Importances')

 ## Findings ##########################################################
 """
In this project we used 75% data as training data and ran five algorithms to train the model.
After training the model, we test the results with the rest 25% dataset and compute the RMSE to 
calculate the acccuracy. For this dataset Random forest gave the lowest RMSE and 
xgboost have highest RMSE.So,for this perticular problem Random forest algorithm is the best option 
to use.

linear Regression Model Report:

RMSE  of training data: 1121
RMSE of test data: 1149

Ridge Regression Model Report:

RMSE  of training data: 1122
RMSE of test data: 1150

Decision Tree Regression Model Report:

RMSE  of training data: 1057
RMSE of test data: 1122

Random forest Regression Model Report:

RMSE  of training data: 1069
RMSE of test data: 1112

XGBOOST Model Report

RMSE  of training data: 66.03
RMSE of test data: 1350
 """