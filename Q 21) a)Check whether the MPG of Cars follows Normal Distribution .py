#!/usr/bin/env python
# coding: utf-8

# Q 21) Check whether the data follows normal distribution
# a)	Check whether the MPG of Cars follows Normal Distribution 
#         Dataset: Cars.csv

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[3]:


carsdata=pd.read_csv('cars.csv')


# In[4]:


carsdata.head


# In[5]:


carsdata['MPG'].mean()


# In[7]:


carsdata['MPG'].median()


# In[8]:


sns.distplot(carsdata['MPG'])
plt.grid(True)
plt.show()


# In[10]:


carsdata['MPG'].skew()


# In[11]:


carsdata['MPG'].kurt()


# In[ ]:




