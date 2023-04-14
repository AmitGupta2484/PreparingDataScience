#!/usr/bin/env python
# coding: utf-8

# Consider only the below columns and prepare a prediction model for predicting Price.
# 
# Corolla<-Corolla c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
# 

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[2]:


# import dataset
dtoyo=pd.read_csv('ToyotaCorolla.csv',encoding='latin1')
dtoyo


# ### EDA

# In[3]:


dtoyo.info()


# In[4]:


dtoyo2=pd.concat([dtoyo.iloc[:,2:4],dtoyo.iloc[:,6:7],dtoyo.iloc[:,8:9],dtoyo.iloc[:,12:14],dtoyo.iloc[:,15:18]],axis=1)
dtoyo2


# In[5]:


dtoyo3=dtoyo2.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
dtoyo3


# In[6]:


dtoyo3[dtoyo3.duplicated()]


# In[7]:


dtoyo4=dtoyo3.drop_duplicates().reset_index(drop=True)
dtoyo4


# In[8]:


dtoyo4.describe().T


# Correlation Analysis

# In[9]:


dtoyo4.corr()


# In[10]:


sns.set_style(style='darkgrid')
sns.pairplot(dtoyo4)


# #### Model Building

# In[13]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight', data=dtoyo4).fit()


# #### Model Testing

# In[14]:


# Finding Coefficient parameters
model.params


# In[15]:


# Finding tvalues and pvalues
model.tvalues , np.round(model.pvalues,5)


# In[18]:


# Finding rsquared values
model.rsquared , model.rsquared_adj   # Model accuracy is 86.17%


# In[ ]:


# Building SLR and MLR models for insignificant variables 'CC' and 'Doors'
# Also finding their tvalues and pvalues


# In[22]:


slr_CC=smf.ols('Price~CC', data=dtoyo4).fit()
slr_CC.tvalues , slr_CC.pvalues


# In[23]:


slr_doors=smf.ols('Price~Doors', data=dtoyo4).fit()
slr_doors.tvalues , slr_doors.pvalues


# In[25]:


mlr_cc_door=smf.ols('Price~CC+Doors',data=dtoyo4).fit()
mlr_cc_door.tvalues , mlr_cc_door.pvalues # CC & Doors have significant pvalue


# ###### Model Validation Techniques
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis

# In[26]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=dtoyo4).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=dtoyo4).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=dtoyo4).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=dtoyo4).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=dtoyo4).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=dtoyo4).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=dtoyo4).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=dtoyo4).fit().rsquared
vif_WT=1/(1-rsq_WT)


# In[27]:


# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df

# None variable has VIF>10, No Collinearity, so consider all varaibles in Regression equation
# In[28]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[29]:


list(np.where(model.resid>6000))  # outliar detection from above QQ plot of residuals


# In[30]:


list(np.where(model.resid<-6000))


# In[31]:


# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) : return (vals-vals.mean())/vals.std()  # User defined z = (x - mu)/sigma


# In[32]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 

# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)    # exog = x-variable & endog = y-variable
# In[33]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()


# In[34]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()


# In[35]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()


# In[36]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()


# In[37]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()


# In[38]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()


# In[41]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()


# In[44]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight', fig=fig)
plt.show()


# #Model Deletion Diagnostics (checking Outliers or Influencers)
# #Two Techniques : 1. Cook's Distance & 2. Leverage value

# In[45]:


# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[46]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(dtoyo4)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[47]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[52]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[51]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=dtoyo4.shape[1]
n=dtoyo4.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[53]:


dtoyo4[dtoyo4.index.isin([80])] 


# ###### Improving the Model

# In[57]:


# Creating a copy of data so that original dataset is not affected
toyo_improved=dtoyo4.copy()
toyo_improved


# In[58]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
dtoyo5=toyo_improved.drop(toyo_improved.index[[80]],axis=0).reset_index(drop=True)
dtoyo5


# ##### Model Deletion Diagnostics and Final Model

# In[60]:


while np.max(c)>0.5 :
   model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=dtoyo5).fit()
   (c,_)=model.get_influence().cooks_distance
   c
   np.argmax(c) , np.max(c)
   dtoyo5=dtoyo5.drop(dtoyo5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
   dtoyo5
else:
   final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=dtoyo5).fit()
   final_model.rsquared , final_model.aic
   print("Thus model accuracy is improved to",final_model.rsquared)


# In[61]:


dtoyo5


# ### Model Predictions

# In[62]:


#  New data for prediction is
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data


# In[63]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[64]:


# Automatic Prediction of Price with 90.02% accurcy
pred_price=final_model.predict(dtoyo5)
pred_price


# In[ ]:




