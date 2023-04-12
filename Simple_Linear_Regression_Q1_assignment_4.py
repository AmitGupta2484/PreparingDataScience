#!/usr/bin/env python
# coding: utf-8

# 1) Delivery_time -> Predict delivery time using sorting time 
# 2) Salary_hike -> Build a prediction model for Salary_hike
# 
# Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


# import dataset
dataset=pd.read_csv('delivery_time.csv')
dataset


# In[4]:


dataset.info()


# In[5]:


# cleaning data Renaming Columns
dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# In[9]:


sns.distplot(dataset['delivery_time'])


# In[10]:


sns.distplot(dataset['sorting_time'])


# In[12]:


dataset.corr()


# In[18]:


sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])


# # Model Building ( simple linear regression)

# In[20]:


model_slr=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# # Model Testing

# In[22]:


# Finding Coefficient parameters
model_slr.params


# In[23]:


# Finding tvalues and pvalues
model_slr.tvalues , model_slr.pvalues


# In[24]:


# Finding R_Squared Values
model_slr.rsquared , model_slr.rsquared_adj


# # Model Prediction

# In[25]:


# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time


# In[26]:


# Automatic Prediction for say sorting time 5, 8
Auto_pred=pd.Series([5,8])
Auto_pred


# In[27]:


data_pred=pd.DataFrame(Auto_pred,columns=['sorting_time'])
data_pred


# In[29]:


model_slr.predict(data_pred)


# In[ ]:




