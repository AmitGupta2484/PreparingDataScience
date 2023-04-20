#!/usr/bin/env python
# coding: utf-8

# Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters. 
# Draw the inferences from the clusters obtained.
# 
# Data Description:
#  
# The file EastWestAirlinescontains information on passengers who belong to an airline’s frequent flier program. For each passenger the data include information on their mileage history and on different ways they accrued or spent miles in the last year. The goal is to try to identify clusters of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers
# 
# ID --Unique ID
# 
# Balance--Number of miles eligible for award travel
# 
# Qual_mile--Number of miles counted as qualifying for Topflight status
# 
# cc1_miles -- Number of miles earned with freq. flyer credit card in the past 12 months:
# cc2_miles -- Number of miles earned with Rewards credit card in the past 12 months:
# cc3_miles -- Number of miles earned with Small Business credit card in the past 12 months:
# 
# 1 = under 5,000
# 2 = 5,000 - 10,000
# 3 = 10,001 - 25,000
# 4 = 25,001 - 50,000
# 5 = over 50,000
# 
# Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months
# 
# Bonus_trans--Number of non-flight bonus transactions in the past 12 months
# 
# Flight_miles_12mo--Number of flight miles in the past 12 months
# 
# Flight_trans_12--Number of flight transactions in the past 12 months
# 
# Days_since_enrolled--Number of days since enrolled in flier program
# 
# Award--whether that person had award flight (free flight) or not

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[2]:


#Read_dataset
ewairline=pd.read_csv('EastWestAirlines.csv')
ewairline


# In[3]:


ewairline.info()


# In[4]:


#Dropping ID column as it is not needed
ewairline1=ewairline.drop(['ID#'],axis=1)
ewairline1


# In[5]:


# Normalize dataset
ewairline2_norm=pd.DataFrame(normalize(ewairline1),columns=ewairline1.columns)
ewairline2_norm


# In[6]:


# Create Dendrograms
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(ewairline2_norm,'complete'))


# In[7]:


# create clusters
cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')


# In[8]:


# save clusters for chart
y_cluster = pd.DataFrame(cluster.fit_predict(ewairline2_norm),columns=['clustersid'])
y_cluster['clustersid'].value_counts()


# In[12]:


# Adding clusters to dataset
ewairline1['clustersid']=cluster.labels_
ewairline1


# In[14]:


ewairline1.sort_values("clustersid")


# In[16]:


### Let us relook at the data using different methods

ewairline['clustersid'] = y_cluster
ewairline.head()


# In[18]:


ewairline.sort_values("clustersid")


# In[19]:


df1 = ewairline.drop(['ID#'],axis=1)
df_clust = df1.groupby('clustersid').mean()
df_clust


# In[ ]:




