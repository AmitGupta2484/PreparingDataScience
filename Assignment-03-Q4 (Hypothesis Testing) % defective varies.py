#!/usr/bin/env python
# coding: utf-8
# TeleCall uses 4 centers around the globe to process customer order forms. They audit a certain %  of the customer order forms. Any error in order form renders it defective and has to be reworked before processing.  The manager wants to check whether the defective %  varies by centre. Please analyze the data at 5% significance level and help the manager draw appropriate inferences

Minitab File: CustomerOrderForm.mtw
# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[3]:


data=pd.read_csv('Costomer+OrderForm.csv')
data


# In[4]:


data.Phillippines.value_counts()
 


# In[5]:


data.Indonesia.value_counts()


# In[6]:


data.Malta.value_counts()


# In[7]:


data.India.value_counts()


# In[8]:


# Make a contingency table
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs


# In[9]:


# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)

chi2_contingency(obs)


# Inference: As (p_value = 0.2771) > (α = 0.05); Accept Null Hypthesis i.e. Independence of categorical variables Thus, customer order forms defective % does not varies by centre.
