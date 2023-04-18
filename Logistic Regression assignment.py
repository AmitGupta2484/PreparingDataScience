#!/usr/bin/env python
# coding: utf-8

# # Output variable -> y
# y -> Whether the client has subscribed a term deposit or not 
# Binomial ("yes" or "no")
# 
# Attribute information For bank dataset
# 
#    Input variables:
#    # bank client data:
#    1 - age (numeric)
#    2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
#                                        "blue-collar","self-employed","retired","technician","services") 
#    3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
#    4 - education (categorical: "unknown","secondary","primary","tertiary")
#    5 - default: has credit in default? (binary: "yes","no")
#    6 - balance: average yearly balance, in euros (numeric) 
#    7 - housing: has housing loan? (binary: "yes","no")
#    8 - loan: has personal loan? (binary: "yes","no")
#    # related with the last contact of the current campaign:
#    9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
#   10 - day: last contact day of the month (numeric)
#   11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#   12 - duration: last contact duration, in seconds (numeric)
#    # other attributes:
#   13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#   14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
#   15 - previous: number of contacts performed before this campaign and for this client (numeric)
#   16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
# 
#   Output variable (desired target):
#   17 - y - has the client subscribed a term deposit? (binary: "yes","no")
# 
# 8. Missing Attribute Values: None

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# To enable plotting graphs in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#Sklearn package's data splitting function which is based on random function
from sklearn.model_selection import train_test_split

#For logistic Regression model
from sklearn.linear_model import LogisticRegression # Focused Algorithm 

import statsmodels.api as sm
# calculate accuracy measures and confusion matrix
from sklearn import metrics


# In[2]:


#Load the data set
bank = pd.read_csv("bank-full.csv")
bank


# # Exploratory Data Analysis

# In[3]:


bank.info()


# In[4]:


# One-Hot Encoding of categrical variables, adding columns
data1=pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome','month'])
data1


# In[5]:


# To see all columns in one output
pd.set_option("display.max.columns", None)
data1


# In[6]:


data1.info()


# In[7]:


# Custom Binary Encoding (yes=1, no=0) of Binary o/p variables 
data1['default'] = np.where(data1['default'].str.contains("yes"), 1, 0)
data1['housing'] = np.where(data1['housing'].str.contains("yes"), 1, 0)
data1['loan'] = np.where(data1['loan'].str.contains("yes"), 1, 0)
data1['y'] = np.where(data1['y'].str.contains("yes"), 1, 0)
data1


# In[8]:


data1.info()


# # Model Building

# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[19]:


bank.shape


# In[20]:


X_train.shape


# In[21]:


y_train.shape


# In[22]:


45211*0.70 #column used for train


# In[23]:


X_test.shape


# In[24]:


y_test.shape


# In[25]:


model = LogisticRegression()


# In[26]:


model.fit(X_train, y_train)


# In[28]:


y_train_pred = model.predict(X_train)


# In[30]:


y_train_pred


# In[31]:


metrics.accuracy_score(y_train, y_train_pred) # accuracy of model in during train


# In[32]:


y_test_pred = model.predict(X_test)


# In[33]:


y_test_pred


# In[34]:


metrics.accuracy_score(y_test, y_test_pred) # accuracy of model in during test


# In[35]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


# In[36]:


confusion_matrix(y_test, y_test_pred) 


# In[42]:


tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()


# In[43]:


print(tn, fp, fn, tp)


# In[45]:


recall_score(y_test, y_test_pred)


# In[46]:


precision_score(y_test, y_test_pred)


# In[47]:


print(classification_report(y_test, y_test_pred))


# In[48]:


y_test_pred = model.predict(X_test)
print(y_test_pred)


# In[50]:


y_pred_prob = model.predict_proba(X_test)
print(y_pred_prob)


# In[51]:


#AUC Value
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, y_test_pred)
logit_roc_auc


# In[54]:


#Plotting the ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_test_pred)
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')

print('auc accuracy:',logit_roc_auc)


# In[ ]:




