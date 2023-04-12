#!/usr/bin/env python
# coding: utf-8

# 2) Salary_hike -> Build a prediction model for Salary_hike
# 
# Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


salary_data=pd.read_csv('Salary_Data.csv')
salary_data


# EDA and Data Visualization

# In[3]:


salary_data.info()


# In[4]:


sns.distplot(salary_data['YearsExperience'])


# In[5]:


sns.distplot(salary_data['Salary'])


# # Correlation Analysis

# In[6]:


salary_data.corr()


# In[7]:


sns.regplot(x=salary_data['YearsExperience'],y=salary_data['Salary'])


# ### Model Building

# In[17]:


model_slr=smf.ols("Salary~YearsExperience",data=salary_data).fit()


# ### Model Testing

# In[18]:


# Finding Cefficient Parameters
model_slr.params


# In[19]:


# Finding Pvalues and tvalues
model_slr.tvalues, model_slr.pvalues


# In[20]:


# Finding Rsquared values
model_slr.rsquared , model_slr.rsquared_adj


# ### Model Predictions

# In[21]:


# Manual prediction for say 3 Years Experience
Salary = (25792.200199) + (9449.962321)*(3)
Salary


# In[22]:


# Automatic Prediction for say 3 & 5 Years Experience 

new_data=pd.Series([3,5])
new_data


# In[23]:


data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred


# In[25]:


model_slr.predict(data_pred)


# In[ ]:




