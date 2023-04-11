#!/usr/bin/env python
# coding: utf-8

# A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions.
# 
# Minitab File : Cutlets.mtw
# 
# Assume Null hyposthesis as Ho: μ1 = μ2 (There is no difference in diameters of cutlets between two units).
# 
# Thus Alternate hypothesis as Ha: μ1 ≠ μ2 (There is significant difference in diameters of cutlets between two units) 2 Sample 2 Tail test applicable
# 

# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


# In[3]:


dataF=pd.read_csv('Cutlets.csv')
dataF.head()


# In[5]:


unitA=pd.Series(dataF.iloc[:,0])
unitA


# In[6]:


unitB=pd.Series(dataF.iloc[:,1])
unitB


# In[7]:


# 2-sample 2-tail ttest:   stats.ttest_ind(array1,array2)     # ind -> independent samples
p_value=stats.ttest_ind(unitA,unitB)
p_value


# In[8]:


p_value[1]


# # P-value(0.4722)>0.05 and hence P High and Ho Fly.
# 
# The inference is that there is no significant difference in the diameters of Unit A and Unit B.
