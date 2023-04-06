#!/usr/bin/env python
# coding: utf-8

# Q 23) Calculate the t scores of 95% confidence interval, 96% confidence interval, 99% confidence interval for sample size of 25

# In[1]:


from scipy import stats
from scipy.stats import norm


# In[2]:


# t scores of 95% confidence interval for sample size of 25
stats.t.ppf(0.975,24)  # df = n-1 = 24


# In[3]:


# t scores of 96% confidence interval for sample size of 25
stats.t.ppf(0.98,24)


# In[4]:


# t scores of 99% confidence interval for sample size of 25
stats.t.ppf(0.995,24)


# In[ ]:




