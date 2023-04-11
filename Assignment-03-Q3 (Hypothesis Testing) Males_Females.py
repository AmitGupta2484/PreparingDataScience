#!/usr/bin/env python
# coding: utf-8
# Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions.

Ho= All proportions are equal

Ha= Not all proportions are equal 

Check p-value
If p-Value < alpha, we reject Null Hypothesis

# In[5]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


# In[3]:


datasales=pd.read_csv('BuyerRatio.csv')
datasales.head()


# In[4]:


datasales1=datasales.iloc[:,1:6]
datasales1


# In[5]:


datasales1.values


# In[7]:


val=stats.chi2_contingency(datasales1)
val


# In[9]:


rows=len(datasales1.iloc[0:2,0])
columns=len(datasales1.iloc[0,0:4])
degree_of_f=(rows-1)*(columns-1)
print('degree_of_f=', degree_of_f)


# In[10]:


exp_value=val[3]
exp_value


# In[6]:


observed = np.array([50, 142, 131, 70, 435, 1523, 1356, 750])
expected = np.array([42.76531299,  146.81287862,  131.11756787, 72.30424052, 442.23468701, 1518.18712138, 1355.88243213, 747.69575948])


# In[7]:


statistics, p_value = stats.chisquare(observed, expected, ddof = 3)
print("Statistics = ",statistics,"\n",'P_Value = ', p_value)


# p_value is greater than alpha
# we failed to reject Null Hypothesis because of lack of evidence
# Therefore, there is no association or dependency between male-female buyers rations and are similar across regions. Hence, Independent samples
