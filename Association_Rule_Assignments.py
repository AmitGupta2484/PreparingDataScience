#!/usr/bin/env python
# coding: utf-8

# # Prepare rules for the all the data sets 
# 1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values
# 2) Change the minimum length in apriori algorithm
# 3) Visulize the obtained rules using different plots 

# In[14]:


get_ipython().system('pip install mlxtend')


# In[15]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[17]:


#Read data
book=pd.read_csv('book.csv')
book

# Data preprocessing not required as it is already in transaction format

Apriori Algorithm
1) Association rules with 10% Support and 70% confidence
# In[18]:


# With 10% Support
frequent_itemsets=apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[19]:


# with 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules


# In[20]:


rules.sort_values('lift',ascending=False)


# In[21]:


# Lift Ratio > 1 is a good association rule in selecting the associated transactions
rules[rules.lift>1]


# In[22]:


# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[23]:


# now with 20% Support
frequent_itemsets2=apriori(book,min_support=0.20,use_colnames=True)
frequent_itemsets2


# In[24]:


# With 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2


# In[25]:


# plot graph of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[26]:


# Now With 5% Support
frequent_itemsets3=apriori(book,min_support=0.05,use_colnames=True)
frequent_itemsets3


# In[27]:


# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3


# In[28]:


# plot graph of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# # Prepare rules for the all the data sets
# 1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values
# 2) Change the minimum length in apriori algorithm
# 3) Visulize the obtained rules using different plots (Data=My Movies)

# In[29]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[31]:


#import dataset
movie=pd.read_csv("my_movies.csv")
movie


# In[32]:


movie.info()


# In[33]:


#Processing Data
movie2=movie.iloc[:,5:]
movie2


# Apriori Algorithm
# 1. Association rules with 10% Support and 70% confidence

# In[34]:


# with 10% support
frequent_itemsets=apriori(movie2,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[35]:


# 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules


# In[36]:


# Lift Ratio > 1 is a good association rule in selecting the associated transactions
rules[rules.lift>1]


# In[37]:


# Plotting graph for visualization 
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# 2. Association rules with 5% Support and 90% confidence

# In[38]:


# with 5% support
frequent_itemsets2=apriori(movie2,min_support=0.05,use_colnames=True)
frequent_itemsets2


# In[39]:


# 90% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.9)
rules2


# In[40]:


# Lift Ratio > 1 is a good association rule in selecting the associated transactions
rules2[rules2.lift>1]


# In[41]:


# Plotting graph for visualization
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[ ]:




