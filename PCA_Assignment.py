#!/usr/bin/env python
# coding: utf-8

# # Perform Principal component analysis and perform clustering using first 
# 3 principal component scores (both heirarchial and k mean clustering(scree plot or elbow curve) and obtain 
# optimum number of clusters and check whether we have obtained same number of clusters with the original data 
# (class column we have ignored at the begining who shows it has 3 clusters)df

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 


# In[2]:


#Read the dataset
data=pd.read_csv('wine.csv')
data.head()


# In[4]:


data.describe().T


# In[5]:


data['Type'].value_counts()


# In[6]:


#dropping column first
data2= data.iloc[:,1:]
data2


# In[8]:


data2.shape


# In[9]:


data2.info()


# In[10]:


# Converting data into numpy array
Data_ary=data2.values
Data_ary


# In[11]:


# Normalizing the numerical data 
data_normal = scale(Data_ary)


# In[12]:


data_normal


# In[13]:


#applying PCA fit to data
pca = PCA()
pca_values = pca.fit_transform(data_normal)


# In[14]:


pca_values


# In[15]:


pca_values.shape


# In[16]:


# PCA Components matrix or convariance Matrix
pca.components_


# In[17]:


# The amount of variance that each PCA has
variance = pca.explained_variance_ratio_
variance


# In[18]:


# Cummulative variance of each PCA
Var_cum = np.cumsum(np.round(variance,decimals= 4)*100)
Var_cum


# In[20]:


plt.plot(Var_cum,color="blue");


# In[21]:


#Concating Wine type and PCA values
wine_pca=pd.concat([data['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
wine_pca


# In[22]:


# Visualization of PCAs
import seaborn as sns
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=wine_pca);


# In[27]:


sns.scatterplot(data=wine_pca, x='PC1', y='PC2', hue='Type');


# In[28]:


#PCA1 values
pca_values[: ,0:1]


# In[29]:


x= pca_values[:,0:1]
y= pca_values[:,1:2]
plt.scatter(x,y);


# # Checking with other Clustering Algorithms
# Hierarchical Clustering

# In[30]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[31]:


# use normalized data to create Dendrograms
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(data_normal,'complete'))


# In[32]:


# Create Clusters
h_clusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
h_clusters


# In[33]:


y=pd.DataFrame(h_clusters.fit_predict(data_normal),columns=['clustersid'])
y['clustersid'].value_counts()


# In[37]:


# Adding clusters to dataset
wine_data=data.copy()
wine_data['clustersid']=h_clusters.labels_
wine_data


# # K-Means Clustering

# In[38]:


from sklearn.cluster import KMeans


# In[39]:


# As we already have normalized data
# Use Elbow Graph to find optimum number of  clusters (K value) from K values range
# The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion WCSS 
# random state can be anything from 0 to 42, but the same number to be used everytime,so that the results don't change.


# In[42]:


# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(data_normal)
    wcss.append(kmeans.inertia_)


# In[43]:


# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');


# # Build Cluster algorithm using K-3

# In[45]:


# Cluster algorithm using K=3
clusters3=KMeans(3,random_state=30).fit(data_normal)
clusters3


# In[46]:


clusters3.labels_


# In[48]:


# Assign clusters to the data set
wine3=data.copy()
wine3['clusters3id']=clusters3.labels_
wine3


# In[49]:


wine3['clusters3id'].value_counts()


# In[ ]:




