#!/usr/bin/env python
# coding: utf-8

# Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.
# 
# Data Description:
# Murder -- Muder rates in different places of United States
# 
# Assualt- Assualt rate in different places of United States
# 
# UrbanPop - urban population in different places of United States
# 
# Rape - Rape rate in different places of United States

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[2]:


# Import Dataset
crime=pd.read_csv('crime_data.csv')
crime


# In[3]:


crime.info()


# In[4]:


crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime


# In[5]:


# Normalize dataset
crime_norm=StandardScaler().fit_transform(crime)
crime_norm


# In[6]:


# DBSCAN Clustering
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(crime_norm)


# In[7]:


dbscan.labels_

#Noisy samples represented as label -1


# In[8]:


# Adding clusters to dataset
crime['clusters']=dbscan.labels_
crime


# In[9]:


crime.groupby('clusters').agg(['mean']).reset_index()


# In[10]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(crime['clusters'],crime['UrbanPop'], c=dbscan.labels_) 


# In[ ]:




