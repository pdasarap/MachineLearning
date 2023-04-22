#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from subprocess import call
from Assignment2_bcd import *
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[26]:


for depth in range(1,6,2):

    #Load the train data
    M = np.genfromtxt('./Diabeties.csv', missing_values=0, skip_header=1, delimiter=',', dtype='int', autostrip='True')
    y = M[:, -1]
    X = M[:, :-1]

    avgs={}
    avgs[0]=3.84
    avgs[1]=120.89
    avgs[2]=69.10
    avgs[3]=20.54
    avgs[4]=79.80
    avgs[5]=32.0
    avgs[6]=0.472
    avgs[7]=33.24

    for key,val in avgs.items():
        for i in range(X.shape[0]):
            if(X[i,key]>val):
                X[i,key]=1
            else:
                X[i,key]=0  

    
    tst_frac = 0.3  # Fraction of examples to sample for the test set
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)


    attrvalpairs = []
    for i in range(X_trn.shape[1]):
        set_values = set()
        for j in X_trn[:,i]:
            if j not in set_values:
                set_values.add(j)
                attrvalpairs.append((i,j))


    tree = id3(X_trn, y_trn, attrvalpairs, max_depth=depth)
    y_dt=np.array([predict_example(x,tree) for x in X_tst])
    print('Confusion matrix for depth {}'.format(depth))
    print(confusion_matrix(y_tst, y_dt))
    dot_str = to_graphviz(tree)
    render_dot_file(dot_str, './tree_depth{}'.format(depth))


# # <font color='#556b2f'>**Part D**</font>

# In[10]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# ### Depth = 1

# In[27]:


decisionTree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
decisionTree1 = decisionTree1.fit(X_trn, y_trn)
ydt1=decisionTree1.predict(X_tst)
err_1=compute_error(y_tst,ydt1)
confusion_matrix(y_tst, ydt1)


# In[28]:


export_graphviz(decisionTree1,out_file ="DiabetiesTree1.dot",filled=True,rounded=True)
call(['dot', '-T', 'png', 'DiabetiesTree1.dot', '-o', 'DiabetiesTree1.png'])


# ### Depth = 3

# In[29]:


decisionTree3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
decisionTree3 = decisionTree3.fit(X_trn, y_trn)
ydt3=decisionTree3.predict(X_tst)
err_3=compute_error(y_tst,ydt3)
confusion_matrix(y_tst, ydt3)


# In[30]:


export_graphviz(decisionTree3,out_file ="DiabetiesTree3.dot",filled=True,rounded=True)
call(['dot', '-T', 'png', 'DiabetiesTree3.dot', '-o', 'DiabetiesTree3.png'])


# ### Depth = 5

# In[31]:


decisionTree5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
decisionTree5 = decisionTree5.fit(X_trn, y_trn)
ydt5=decisionTree5.predict(X_tst)
err_5=compute_error(y_tst,ydt5)
confusion_matrix(y_tst, ydt5)


# In[32]:


export_graphviz(decisionTree5,out_file ="DiabetiesTree5.dot",filled=True,rounded=True)
call(['dot', '-T', 'png', 'DiabetiesTree5.dot', '-o', 'DiabetiesTree5.png'])

