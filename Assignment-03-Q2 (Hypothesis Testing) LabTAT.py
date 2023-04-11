#!/usr/bin/env python
# coding: utf-8

# A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch.
#    
#   Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level.
# 
# Minitab File: LabTAT.mtw
# 
# Ho= Average TAT for all the samples is the same
# 
# Ha= Averages TAT for all the samples is not the same

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


# In[2]:


dataF=pd.read_csv('LabTAT.csv')
dataF


# In[3]:


# Anova ftest statistics: stats.f_oneway(column-1,column-2,column-3,column-4)
p_value=stats.f_oneway(dataF.iloc[:,0],dataF.iloc[:,1],dataF.iloc[:,2],dataF.iloc[:,3])
p_value


# we can see that P-value is < 0.05
# 
# Hence there is a significant difference in the average TAT for all the labs
