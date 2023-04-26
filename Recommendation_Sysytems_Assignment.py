#!/usr/bin/env python
# coding: utf-8

# # Problem statement.
# 
# Build a recommender system by using cosine simillarties score.Book dataset
# 

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings("ignore")


# In[9]:


books= pd.read_csv('book.csv',  encoding='latin-1')
books


# In[11]:


books1= books.drop(['Unnamed: 0'], axis=1)


# In[12]:


#Renaming columns
books1=books1.rename({'User.ID':'userid','Book.Title':'booktitle','Book.Rating':'bookrating'},axis = 1)
books1


# In[13]:


len(books1['userid'].unique())


# In[16]:


array_user = books1['userid'].unique()
array_user


# In[15]:


# extracting unique values from Book.Title column, also getting the count of unique books

len(books1['booktitle'].unique())


# In[17]:


# Creating Collaborative matrix using PIVOT function

books_df1 = books1.pivot_table(index = 'userid',
                        columns = 'booktitle',
                        values = 'bookrating').reset_index(drop = True)


# In[18]:


books_df1


# In[19]:


#Mapping Original userid to dataframe

books_df1.index = books1.userid.unique()


# In[20]:


books_df1


# In[21]:


# replacing all the null values with 0.0 ,so that we can calculate the similarities

books_df1.fillna(0, inplace = True)
books_df1


# In[22]:


# Calculating similarties using cosine metrics

user_sim = 1 - pairwise_distances(books_df1.values, metric = 'cosine')


# In[23]:


user_sim


# In[24]:


# Converting the calculated similarities in dataframe
user_sim_df = pd.DataFrame(user_sim)   

# Displaying first 5 records
user_sim_df.iloc[0:5,0:5]                       


# In[25]:


# Mapping the index with original userids
user_sim_df.index = books1.userid.unique()    

# Mapping the column_names with original userids
user_sim_df.columns = books1.userid.unique()      


# In[26]:


user_sim_df.iloc[0:5,0:5]


# In[27]:


# As the similarities between the same userids will be always one  
#so as we are required to see the similarities between different users so filling the diagonal values by 0

np.fill_diagonal(user_sim,0) 


# In[28]:


user_sim_df.iloc[0:5,0:5]


# In[29]:


# Finding out which users are having maximum simlarities between them

user_sim_df.idxmax(axis = 1)


# In[31]:


# Displaying records of the users having similarities

books1[(books1['userid'] == 276744) | (books1['userid'] == 276726)]


# In[32]:


user_1 = books1[books1['userid'] == 162129]


# In[33]:


user_2 = books1[books1['userid'] == 276726]


# In[34]:


pd.merge(user_1,user_2, on = 'booktitle', how = 'outer')

