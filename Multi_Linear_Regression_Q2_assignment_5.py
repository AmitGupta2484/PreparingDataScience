#!/usr/bin/env python
# coding: utf-8

# Prepare a prediction model for profit of 50_startups data.
# Do transformations for getting better predictions of profit and
# make a table containing R^2 value for each prepared model.
# 
# R&D Spend -- Research and devolop spend in the past few years
# Administration -- spend on administration in the past few years
# Marketing Spend -- spend on Marketing in the past few years
# State -- states from which data is collected
# Profit  -- profit of each state in the past few years

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[2]:


data=pd.read_csv('50_Startups.csv')
data


# In[3]:


#EDA
data.info()


# In[4]:


data1=data.rename({'R&D Spend':'RND','Administration':'ADMIN','Marketing Spend':'MKTSPD'},axis=1)
data1


# In[5]:


data1[data1.duplicated()] # No duplicated data


# In[6]:


data1.describe().T

#### Correlation Analysis
# In[7]:


data1.corr()


# In[8]:


sns.set_style(style='darkgrid')
sns.pairplot(data1)


# #### Model Building

# In[10]:


modelMLR=smf.ols("Profit~RND+ADMIN+MKTSPD", data=data1).fit()


# ###### Model Testing

# In[12]:


# Finding Coefficient parameters
modelMLR.params


# In[13]:


# Finding tvalues and pvalues
modelMLR.tvalues , np.round(modelMLR.pvalues,5)


# In[16]:


# Finding rsquared values
modelMLR.rsquared , modelMLR.rsquared_adj  # Model accuracy is 94.75%


# In[ ]:


# Build SingleLinearRegression and MultiLinearRegression models for insignificant variables 'ADMIN' and 'MKTSPD'
# Also find their tvalues and pvalues


# In[19]:


SLR_ADMIN=smf.ols("Profit~ADMIN", data=data1).fit()


# In[20]:


SLR_ADMIN.tvalues , SLR_ADMIN.pvalues


# #### ADMIN has in-significant pvalue

# In[23]:


SLR_MKTSPD=smf.ols("Profit~MKTSPD", data=data1).fit()


# In[25]:


SLR_MKTSPD.tvalues , SLR_MKTSPD.pvalues


# #### Marketing Spend has significant pvalue

# In[27]:


MLR_ADMIN_MKT=smf.ols("Profit~ADMIN+MKTSPD",data=data1).fit()
MLR_ADMIN_MKT.tvalues , MLR_ADMIN_MKT.pvalues 


# #### Variables has significant pvalue

# ## Model Validation
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis 

# #### 1) Collinearity Problem Check
#  Calculate VIF = 1/(1-Rsquare) for all independent variables
#  & 2. Residual Analysis

# In[29]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_r=smf.ols("RND~ADMIN+MKTSPD",data=data1).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMIN~RND+MKTSPD",data=data1).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKTSPD~RND+ADMIN",data=data1).fit().rsquared
vif_m=1/(1-rsq_m)

# Putting the values in Dataframe format
d1={'Variables':['RND','ADMIN','MKTSPD'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(d1)
Vif_df

# None variable has VIF>10, No Collinearity, so consider all varaibles in Regression equation
# In[33]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)

sm.qqplot(modelMLR.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[36]:


# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) : return (vals-vals.mean())/vals.std()  # User defined z = (x - mu)/sigma


# In[39]:


plt.scatter(standard_values(modelMLR.fittedvalues),standard_values(modelMLR.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 

#### Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)    # exog = x-variable & endog = y-variable
# In[41]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(modelMLR,'RND',fig=fig)
plt.show()


# #### Model Deletion Diagnostics (checking Outliers or Influencers)                                                                                                        Two Techniques : 1. Cook's Distance & 2. Leverage value

# In[44]:


# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
modelMLR.influence= modelMLR.get_influence()
(c,_)=modelMLR.get_influence().cooks_distance
c


# In[45]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data1)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[46]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[47]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
influence_plot(modelMLR)
plt.show()


# In[48]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=data1.shape[1]
n=data1.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[49]:


data1[data1.index.isin([49])] 


# In[50]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
data2=data1.drop(data1.index[[49]],axis=0).reset_index(drop=True)
data2


# #### Model Deletion Diagnostics and Final Model

# In[51]:


while np.max(c)>0.5 :
    model=smf.ols("Profit~RND+ADMIN+MKTSPD",data=data2).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    data2=data2.drop(data2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    data2
else:
    final_model=smf.ols("Profit~RND+ADMIN+MKTSPD",data=data2).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)


# In[52]:


final_model.rsquared


# #### Model Predictions

# In[54]:


# say New data for prediction is
new_data=pd.DataFrame({'RND':70000,"ADMIN":90000,"MKTSPD":140000},index=[0])
new_data


# In[55]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[56]:


# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(data2)
pred_y


# In[57]:


d2={'Prep_Models':['ModelMLR','Final_Model'],'Rsquared':[modelMLR.rsquared,final_model.rsquared]}
table=pd.DataFrame(d2)
table


# In[ ]:




