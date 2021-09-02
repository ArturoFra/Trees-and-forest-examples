#!/usr/bin/env python
# coding: utf-8

# # Arboles de Regresión

# In[1]:


import pandas as pd 
import numpy as np


# In[28]:


data = pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/boston/Boston.csv")
data.head()


# In[3]:


data.shape


# In[6]:


colnames = data.columns.values.tolist()
predictors = colnames[:13]
target= colnames[13]
X=data[predictors]
Y=data[target]


# In[7]:


from sklearn.tree import DecisionTreeRegressor 


# In[8]:


regtree= DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10)


# In[24]:


regtree.fit(X,Y)


# In[ ]:





# In[25]:


preds = regtree.predict(data[predictors])


# In[26]:


data["preds"]= preds


# In[ ]:





# In[27]:


data[["preds", "medv"]]


# In[ ]:





# In[13]:


from sklearn import tree
import matplotlib.pyplot as plt


# In[14]:


tree.plot_tree(regtree)


# In[15]:


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (30,30), dpi=900)
tree.plot_tree(regtree, filled = True);
fig.savefig('imageregtree.png')


# In[29]:


from sklearn.model_selection import KFold   
from sklearn.model_selection import cross_val_score


# In[ ]:





# In[30]:


cv = KFold( n_splits= 10, shuffle = True, random_state=1)


# In[ ]:





# In[31]:


scores = cross_val_score(regtree, X, Y, scoring="accuracy", cv = cv, n_jobs=1)
scores


# In[ ]:





# In[ ]:





# In[ ]:


score=np.mean(scores)
print(score)


# In[32]:


list(zip(predictors, regtree.feature_importances_))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




