#!/usr/bin/env python
# coding: utf-8

#   Q 24)   A Government  company claims that an average light bulb lasts 270 days. A researcher randomly selects 18 bulbs for testing. The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. If the CEO's claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days
# Hint:  rcode   pt(tscore,df)  
#  df  degrees of freedom

# In[1]:


from scipy import stats
from scipy.stats import norm


# In[ ]:


# Assume Null Hypothesis is: Ho = Avg life of Bulb >= 260 days
# Alternate Hypothesis is: Ha = Avg life of Bulb < 260 days


# In[2]:


# find t-scores at x=260; t=(s_mean-P_mean)/(s_SD/sqrt(n))
tscore=(260-270)/(90/18**0.5)
tscore


# In[ ]:


# Find P(X>=260) for null hypothesis


# In[3]:


# p_value=1-stats.t.cdf(abs(t_scores),df=n-1)... Using cdf function
p_value=1-stats.t.cdf(abs(-0.4714),df=17)
p_value


# In[5]:


#  OR p_value=stats.t.sf(abs(t_score),df=n-1)... Using sf function
p_value_sf=stats.t.sf(abs(-0.4714),df=17)
p_value_sf


# In[ ]:




