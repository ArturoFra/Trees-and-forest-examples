#!/usr/bin/env python
# coding: utf-8

# # Árbol de decisión para especies de flores

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


data = pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/iris/iris.csv")
data.head()


# In[ ]:





# In[3]:


data.shape


# In[ ]:





# In[4]:


plt.hist(data.Species)


# In[ ]:





# In[5]:


colnames=data.columns.values.tolist()
predictors=colnames[:4]
target = colnames[4]


# In[6]:


data["is_train"] = np.random.uniform(0,1, len(data))<=0.75


# In[7]:


plt.hist(data.is_train.astype(int))


# In[8]:


train, test = data[data["is_train"]==True], data[data["is_train"]==False]


# In[ ]:





# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:





# In[33]:


tree = DecisionTreeClassifier(criterion = "entropy", min_samples_split=20, random_state=99).fit(train[predictors], train[target])


# In[43]:


clf=tree


# In[ ]:





# In[11]:


preds = tree.predict(test[predictors])


# In[12]:


pd.crosstab(test[target], preds, rownames=["Actual"], colnames=["Predicciones"])


# In[ ]:





# ### Visualizción del árbol de decisión 

# In[13]:


from sklearn.tree import export_graphviz


# In[14]:


with open("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/notebooks/resources/iris_dtree.dot", "w") as dotfile:
    export_graphviz(tree, out_file=dotfile, feature_names=predictors)
    dotfile.close()


# In[15]:


import os 
from graphviz import Source


# In[32]:


file=open("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/notebooks/resources/iris_dtree.dot", "r")
text = file.read()
text
Source(text)


# In[ ]:





# In[28]:


from pydotplus import graph_from_dot_data


# In[31]:


graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')


# In[36]:


dot_data = export_graphviz(tree, feature_names=predictors)


# In[37]:


graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')


# In[39]:


import matplotlib.pyplot as plt


# In[42]:


fig = plt.figure(figsize=(25,20))
fig=tree.plot_tree(tree, feature_names=predictors,filled=True)


# In[50]:


from sklearn import tree


# In[51]:


tree.plot_tree(clf)


# In[52]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[54]:


dot_data = tree.export_graphviz(clf,out_file="tree.dot",feature_names = predictors, filled = True)


# In[62]:


Source(text)


# # Cross validation para la poda

# In[63]:


X= data[predictors]
Y= data[target]


# In[64]:


arbol = DecisionTreeClassifier(criterion = "entropy", min_samples_split=20, random_state=99)


# In[65]:


arbol.fit(X,Y)


# In[67]:


arbol = DecisionTreeClassifier(criterion = "entropy", max_depth=5, min_samples_split=20, random_state=99)


# In[68]:


arbol.fit(X,Y)


# In[69]:


from sklearn.model_selection import KFold   


# In[73]:


cv = KFold( n_splits= 10, shuffle = True, random_state=1)


# In[74]:


from sklearn.model_selection import cross_val_score


# In[77]:


scores = cross_val_score(arbol, X, Y, scoring="accuracy", cv = cv, n_jobs=1)
scores


# In[78]:


score=np.mean(scores)


# In[79]:


score


# In[ ]:





# In[82]:


for i in range(1,11):
    arbol = DecisionTreeClassifier(criterion = "entropy", max_depth=i, min_samples_split=20, random_state=99)
    arbol.fit(X,Y)
    cv = KFold( n_splits= 10, shuffle = True, random_state=1)
    scores = cross_val_score(arbol, X, Y, scoring="accuracy", cv = cv, n_jobs=1)
    score=np.mean(scores)
    print("Score para i = ",i,"es de: ",score)
    print(" ",arbol.feature_importances_)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




