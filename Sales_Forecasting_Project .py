#!/usr/bin/env python
# coding: utf-8

# # 1. Data Preprocessing and Data analysis

# In[1]:


#Importing python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading dataset
data=pd.read_csv('sales_data.csv')
data


# In[3]:


# Convert the date column to datetime
data['date']=pd.to_datetime(data['date'])
data


# In[4]:


#Set the 'date' column as an index
data.set_index('date',inplace=True)


# In[5]:


data


# In[6]:


#plot the sales data
plt.figure(figsize=(10,5))
plt.plot(data['sales'])
plt.title('Monthly_sale')
plt.xlabel('Date')
plt.ylabel('sales')
plt.show()


# In[7]:


data.isnull().sum()


# # 2.Feature Engineering

# In[8]:


#Create Additional time_based features
data['month']=data.index.month
data['quarter']= data.index.quarter
data['year']=data.index.year


# In[9]:


# lag features
data['lag_1'] = data['sales'].shift(1)
data['lag_2'] = data['sales'].shift(2)


# In[10]:


data


# In[11]:


#Moving Average feature
data['rolling_mean_3'] = data['sales'].rolling(window=3).mean()
data['rolling_mean_6'] = data['sales'].rolling(window=6).mean()


# In[12]:


data


# In[13]:


#Drop rows with missing values created by shift/rolling 
data.dropna(inplace=True)


# In[14]:


data


# # 3. Split the data into training and testing sets

# In[15]:


from sklearn.model_selection import train_test_split

# Define the feature and target
features = ['month','quarter','year','lag_1','lag_2','rolling_mean_3','rolling_mean_6']
target = 'sales'

#split the data into training and testing set
x = data[features]
y = data[target]

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# # 4. Machine learning Models

# # # Linear Regression Model

# In[16]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Train the linear regression model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)


# In[17]:


#Make prediction
y_pred_linear_model = linear_model.predict(x_test)


# In[18]:


#Evaluate the model
linear_model_rmse = mean_squared_error(y_test, y_pred_linear_model, squared=False)
print('Linear Regression RMSE:',linear_model_rmse)


# In[19]:


#Plot the result
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual Sales')
plt.plot(y_test.index, y_pred_linear_model, label='Linear Regression Forecast', color='red')
plt.legend()
plt.xlabel('Date', color='green')
plt.ylabel('Sales', color='green')
plt.show()


# # # Random Forest Model

# In[20]:


from sklearn.ensemble import RandomForestRegressor

#Train the random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)


# In[21]:


#Make Prediction
y_pred_rf = rf_model.predict(x_test)


# In[22]:


#Evaluate the model
rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
print('Random Forest RMSE:',rf_rmse)


# In[23]:


# Compare RMSE of different model
print('Linear Regression RMSE: ',linear_model_rmse)
print('Random Forest RMSE: ',rf_rmse)

