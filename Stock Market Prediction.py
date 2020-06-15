#!/usr/bin/env python
# coding: utf-8

# # Stock Market Prediction

# 
# ## In this Notebook we try to predict Stock prices based on Stock prices of previous days.

# ### For that purpose, we will see simple eight ways to predict the Stock prices. The various models to be used are:

# 1. Average
# 2. Weighted Average
# 3. Moving Average
# 4. Moving Weighted Average
# 5. Linear Regression
# 6. Weighted Linear Regression
# 7. Lasso Regression
# 8. Gradient Boosting Regressor

# ## Importing the Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse


# In[3]:


# importing the dataset.

df = pd.read_csv('dataset/infy_stock.csv')


# In[4]:


# checking the top 5 rows of the dataset.

df.head(5)


# In[5]:


# check the shape of the dataset.

df.shape


# In[7]:


# checking the info about the Dataset.

df.info()


# In[8]:


# checking the first and last Date in the Dataset.

print('First Data in the dataset. ', df.Date.min())
print('First Data in the dataset. ', df.Date.max())


# #### We can see that , we only have dataset of the working days, which means from 01.01.2015 to 31.12.2015, which counts to 248 records and from checking the info of the dataset, we can say that none of the attribute contains null values.

# In[ ]:





# In[10]:


# Creating a sub dataframe only contains the data as index and closing of the stock market as a colummn
df_sub = df = pd.read_csv("dataset//infy_stock.csv", usecols=['Date', 'Close'], parse_dates=['Date'],index_col='Date')


# In[11]:


df_sub.head()


# In[12]:


# Let's print the plot of Closing Prices on certain dates.

plt.figure(figsize=(17,5))
df.Close.plot()
plt.title("Closing Price", fontsize = 15)
plt.show()


# In[ ]:





# ### There is a huge drop on 15/06/2015. If we take this whole data, the prediction might not be as expected as there is a split in between!
# 
# ### So either We have to either drop the data or adjust the values before split. Since the split is 2 for 1, we can normalize the data prior to split by dividing them by 2. (Old shares are half that of today's share)

# In[14]:


# Plotting the Adjusted Price.

plt.figure(figsize=(17,5))
stock_price = pd.concat([df_sub.Close[:'2015-06-12']/2, df_sub.Close['2015-06-15':]]) 
plt.plot(stock_price)
plt.title("Closing Price Adjusted",fontsize=15)
plt.show()


# And now after adjusting the time series of Infosys stock prices.
# 
# Lets now Predict the Stock price based on various methods.
# 
# We will predict the values on last 68 days in the series.
# We will use Mean squared error as a metrics to calculate the error in our prediction.
# We will compare the results of various methods at the end.

# In[27]:


#helper function to plot the stock prediction

starting_values = stock_price.iloc[:180]
y_test = stock_price.iloc[180:]

def plot_pred(pred,title):
    plt.figure(figsize=(17,5))
    plt.plot(prev_values,label='Train')
    plt.plot(y_test,label='Actual')
    plt.plot(pred,label='Predicted')
    plt.ylabel("Stock prices")
    plt.title(title,fontsize=15)
    plt.legend()
    plt.show()


# In[28]:


# let's check the top 5 rows and shape of the stock price starting prices.

starting_values.head(), starting_values.shape


# In[25]:


# and check the price of # let's check the top 5 rows and shape of the stock price starting prices.

y_test.head() , y_test.shape


# In[ ]:





# ## Let's start with 1. Average First.

# In[17]:


y_av = pd.Series(np.repeat(prev_values.mean(), 68), index = y_test.index)
mse(y_av, y_test)


# In[18]:


plot_pred(y_av, "Average")


# ### 2. Weighted Mean

# We shall give more weightage to the data which are close to the last day in training data, while calculating the mean. The last day in the training set will get a weightage of 1(=180/180) and the first day will get a weightage of 1/180.

# In[29]:


weight = np.array(range(0,180))/180
weighted_train_data =np.multiply(prev_values,weight)

# weighted average is the sum of this weighted train data by the sum of the weight

weighted_average = sum(weighted_train_data)/sum(weight)
y_wa = pd.Series(np.repeat(weighted_average,68),index=y_test.index)

mse(y_wa,y_test)


# In[30]:


# Plotting the Weighted Average.

plot_pred(y_wa,"Weighted Average")


# For the other methods we will predict the value of stock price on a day based on the values of stock prices of 80 days prior to it. So in our series we will not consider the first eight days (since there previous eighty days is not in the series).
# We have to test the last 68 values. This would be based on the last 80 days stock prices of each day in the test data.
# Since we have neglected first 80 and last 68 is our test set, the train dataset will be between 80 and 180 (100 days).

# In[31]:


y_train = stock_price[80:180]
y_test = stock_price[180:]
print('y train:', y_train.shape, '\ny test:', y_test.shape)


# There are 100 days in training and 68 days in testing set. We will construct the features, that is the last 80 days stock for each date in the y_train and y_test. This would be our target variable.

# In[32]:


X_train = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100)],
                       columns=range(80,0,-1),index=y_train.index)
X_test = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100,168)],
                       columns=range(80,0,-1),index=y_test.index)

X_train


#  X_train is now a collection of 100 dates as index and a collection of stock prices of previous 80 days as features.
# 
#   Similarlily, X_test is now a collection of 68 dates as index and a collection of stock prices of previous 80 days as features.
# 
#   NOTE: Here 76 working days from '2015-05-04', the stock had a price of 986.725 and 77 working days from '2015-05-05', the stock has the same value. You can see the similarity of values along the diagonal. This is because consecutitive data will be similar to the previous except it drops the last value, shifts and has a new value.
# 
#   We will use these values for stock price prediction in the other four methods.

# ## 3.  Moving Average

# We have to predict the 68 values in data set and for each values we will get the average of previous 80 days.
# This will be a simple mean of each column in the y_test

# In[35]:


y_ma = X_test.mean(axis=1)

mse(y_ma, y_test)


# In[36]:


# Ploting the Moving Average

plot_pred(y_ma, 'Moving Average')


# ## 4. Weighted Moving Average

# We will obtain the stock price on the test date by calculating the weighted mean of past 80 days. The last of the 80 day will have a weightage of 1(=80/80) and the first will have a weightage of 1/80.

# In[37]:


weight = np.array(range(1, 81))/ 80

#weighted moving average
y_wma = X_test@weight / sum(weight)

mse(y_wma, y_test)


# In[38]:


# Ploting the Weighted Moving Average

plot_pred(y_wma,"Weighted Moving Average")


# ## 5. Linear regression

# In[40]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
y_lr = pd.Series(y_lr,index=y_test.index)

mse(y_test,y_lr)


# In[42]:


# Ploting the prediction with Linear Regression

plot_pred(y_lr,"Linear Regression")


# ## 6. Weighted Linear Regression

# In This approach we will give weight to the input data rather than the features.

# In[43]:


weight = np.array(range(1,101))/100
wlr = LinearRegression()

wlr.fit(X_train,y_train,weight)
y_wlr = wlr.predict(X_test)
y_wlr = pd.Series(y_wlr,index=y_test.index)

mse(y_test,y_wlr)


# In[44]:


# Ploting the outcome of Weighted Linear Regression
plot_pred(y_wlr, ' Weighted Linear Regression')


# ## 7. Lasso Regression 

# In[45]:


from sklearn.linear_model import Lasso
lasso = Lasso()

las = lasso.fit(X_train,y_train)
y_las = las.predict(X_test)
y_las = pd.Series(y_las,index = y_test.index)

mse(y_las,y_test)


# In[46]:


# Plotting the output of Lassor Regressor

plot_pred(y_las,"Lasso Regression")


# In[ ]:





# ## 8. Gradient Boosting Regressor.

# In[47]:



from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Initializing Gradient Boosting

gb = GradientBoostingRegressor(max_depth = 7, n_estimators=200, learning_rate = 0.01)

# making a list of params for the Gradient Search CV
param = [{'min_samples_split': [5, 9, 13], 'max_leaf_nodes': [3, 5, 7, 9], 'max_features': [8, 10, 15, 18]}]

# initializing the GradientSearchCV with the params
gs = GridSearchCV(gb, param, cv = 5, scoring = 'neg_mean_squared_error')


# In[48]:


gs.fit(X_train,y_train)


# In[49]:


gb = gs.best_estimator_


# In[50]:


y_gb = gb.predict(X_test)
y_gb = pd.Series(y_gb,index = y_test.index)

mse(y_gb, y_test)


# In[51]:


# Plotting the output of Gradrient Boosting Regressor

plot_pred(y_gb, "Gradient Boosting Regression")


# In[ ]:




