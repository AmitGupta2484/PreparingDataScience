#!/usr/bin/env python
# coding: utf-8

# 1) Prepare a classification model using Naive Bayes 
# for salary data 
# 
# Data Description:
# 
# age -- age of a person
# workclass	-- A work class is a grouping of work 
# education	-- Education of an individuals	
# maritalstatus -- Marital status of an individulas	
# occupation	 -- occupation of an individuals
# relationship -- 	
# race --  Race of an Individual
# sex --  Gender of an Individual
# capitalgain --  profit received from the sale of an investment	
# capitalloss	-- A decrease in the value of a capital asset
# hoursperweek -- number of hours work per week	
# native -- Native of an individual
# Salary -- salary of an individual
# 

# In[1]:


#import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings('ignore')

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import statsmodels.api as sm

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# In[2]:


#Read Dataset
salarydata_train = pd.read_csv('SalaryData_Train.csv')
salarydata_train.head()


# In[3]:


salarydata_test = pd.read_csv('SalaryData_Test.csv')
salarydata_test.head()


# #Exploratory data analysis

# In[5]:


salarydata_train.shape


# We can see that there are 30161 instances and 14 attributes in the training data set.

# In[6]:


salarydata_test.shape


# We can see that there are 15060 instances and 14 attributes in the training data set.

# # View summary of dataset

# In[7]:


salarydata_test.info()


# In[8]:


salarydata_test.describe()


# In[9]:


#Finding the special characters in the data frame 
salarydata_test.isin(['?']).sum(axis=0)


# In[10]:


#Finding the special characters in the data frame 
salarydata_train.isin(['?']).sum(axis=0)


# In[11]:


print(salarydata_train[0:5])


# In[12]:


# find categorical variables

categorical = [var for var in salarydata_train.columns if salarydata_train[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)


# In[13]:


# view the categorical variables

salarydata_train[categorical].head()


# In[14]:


# check missing values in categorical variables
salarydata_train[categorical].isnull().sum()


# In[15]:


# view frequency counts of values in categorical variables

for var in categorical: 
    
    print(salarydata_train[var].value_counts())


# In[16]:


# view frequency distribution of categorical variables

for var in categorical: 
    
    print(salarydata_train[var].value_counts()/np.float(len(salarydata_train)))


# In[17]:


# check labels in workclass variable

salarydata_train.workclass.unique()


# In[18]:


# check frequency distribution of values in workclass variable

salarydata_train.workclass.value_counts()


# In[19]:


# check labels in occupation variable

salarydata_train.occupation.unique()


# In[20]:


# check frequency distribution of values in occupation variable

salarydata_train.occupation.value_counts()


# In[21]:


# check labels in native_country variable

salarydata_train.native.unique()


# In[22]:


# check frequency distribution of values in native_country variable

salarydata_train.native.value_counts()


# In[23]:


# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(salarydata_train[var].unique()), ' labels')


# In[24]:


# find numerical variables

numerical = [var for var in salarydata_train.columns if salarydata_train[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[25]:


# view the numerical variables

salarydata_train[numerical].head()


# In[26]:


# check missing values in numerical variables

salarydata_train[numerical].isnull().sum()


# In[27]:


X = salarydata_train.drop(['Salary'], axis=1)

y = salarydata_train['Salary']


# In[28]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[29]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[31]:


X_train.dtypes


# In[32]:


X_test.dtypes


# In[33]:


# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[34]:


# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[35]:


X_train[categorical].head()


# In[36]:


pip install category_encoders


# In[37]:


# import category encoders

import category_encoders as ce


# In[38]:


# encode remaining variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[39]:


X_train.head()


# In[40]:


X_train.shape


# In[41]:


X_test.head()


# In[42]:


X_test.shape


# In[43]:


cols = X_train.columns


# In[44]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[45]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[46]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[47]:


X_train.head()


# # Model training

# In[48]:


# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)


# In[49]:


#Predict the results

y_pred = gnb.predict(X_test)

y_pred


# In[50]:


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[51]:


#Compare the train-set and test-set accuracy

y_pred_train = gnb.predict(X_train)

y_pred_train


# # Check for overfitting and underfitting

# In[53]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# # Compare model accuracy with null accuracy

# In[54]:


# check class distribution in test set

y_test.value_counts()


# In[55]:


# check null accuracy score

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# # Confusion matrix

# In[56]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[57]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[58]:


Classification metrices
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[59]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[60]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# In[61]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# In[62]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# In[63]:


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# In[64]:


#False Positive Rate

false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# In[65]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[66]:


# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(X_test)[0:10]

y_pred_prob


# In[67]:


# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])

y_pred_prob_df


# In[68]:


# print the first 10 predicted probabilities for class 1 - Probability of >50K

gnb.predict_proba(X_test)[0:10, 1]


# In[70]:


# store the predicted probabilities for class 1 - Probability of >50K

y_pred1 = gnb.predict_proba(X_test)[:, 1]


# In[71]:


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of salaries >50K')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


# In[72]:


# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = '>50K')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# In[73]:


# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# In[74]:


#k-Fold Cross Validation
# Applying 10-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))


# In[75]:


# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))


# In[ ]:




