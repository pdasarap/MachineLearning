#!/usr/bin/env python
# coding: utf-8

# # <font color='#556b2f'>**Part B**</font>

# In[12]:


import numpy as np
from DecisionTree import *
import matplotlib.pyplot as plt


# In[13]:


#for monk in range(1,4):
train_loss = []
test_loss = []
for depth in range(1,11):
    M = np.genfromtxt('monks-'+str(3)+'.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    M = np.genfromtxt('monks-'+str(3)+'.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    attrvalpairs = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:,i]:
            if j not in set_values:
                set_values.add(j)
                attrvalpairs.append((i,j))
    decision_tree = id3(Xtrn, ytrn, attrvalpairs, max_depth=depth)
    y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
    y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
    test_err = compute_error(ytst, y_pred_tst)
    train_err = compute_error(ytrn, y_pred_trn)
    train_loss.append(train_err)
    test_loss.append(test_err)
    
plt.figure(figsize=(10,8))
plt.plot([i for i in range(1,11)],train_loss, label="train loss")
plt.plot([i for i in range(1,11)],test_loss, label="test loss")
#plt.plot(train_loss, linewidth=3)
#plt.plot(test_loss, linewidth=3)
depth = []
for i in range(1,11):
    depth.append(i)

plt.title('Monk-3 Data')
plt.xlabel('Tree depth "d"', fontsize=16)
plt.ylabel('Error', fontsize=16)

plt.xticks(list(depth), fontsize=12)
plt.legend(['Train Error', 'Test Error'], fontsize=16,loc='upper right')     


# # <font color='#556b2f'>**Part C**</font>

# In[14]:


import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from subprocess import call

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'


# In[18]:


for depth in range(1,6,2):
    #Load the train data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    N = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = N[:, 0]
    Xtst = N[:, 1:]

    #depths=[i for i in range(1,6,2)]

    attrvalpairs = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:,i]:
            if j not in set_values:
                set_values.add(j)
                attrvalpairs.append((i,j))


    
    tree1 = id3(Xtrn, ytrn, attrvalpairs, max_depth=depth)
    y_dt=np.array([predict_example(x,tree1) for x in Xtst])
    print('Confusion matrix for depth {}'.format(depth))
    print(confusion_matrix(ytst, y_dt))
    dot_str = to_graphviz(tree1)
    render_dot_file(dot_str, './my_learned_tree_depth{}'.format(depth))   


# In[5]:





# # <font color='#556b2f'>**Part D**</font>

# In[11]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# ### Depth = 1

# In[12]:


decision_tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
decision_tree1 = decision_tree1.fit(Xtrn, ytrn)
y_dt1=decision_tree1.predict(Xtst)
err1=compute_error(ytst,y_dt1)
confusion_matrix(ytst, y_dt1)


# In[13]:


export_graphviz(decision_tree1,out_file ="myDecTree1.dot",filled=True,rounded=True)
call(['dot', '-T', 'png', 'myDecTree1.dot', '-o', 'myDecTree1.png'])


# ### Depth = 3

# In[14]:


decision_tree3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
decision_tree3 = decision_tree3.fit(Xtrn, ytrn)
y_dt3=decision_tree3.predict(Xtst)
err3=compute_error(ytst,y_dt3)
confusion_matrix(ytst, y_dt3)


# In[15]:


export_graphviz(decision_tree3,out_file ="myDecTree3.dot",filled=True,rounded=True)
call(['dot', '-T', 'png', 'myDecTree3.dot', '-o', 'myDecTree3.png'])


# ### Depth = 5

# In[16]:


decision_tree5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
decision_tree5 = decision_tree5.fit(Xtrn, ytrn)
y_dt5=decision_tree5.predict(Xtst)
err5=compute_error(ytst,y_dt5)
confusion_matrix(ytst, y_dt5)


# In[17]:


export_graphviz(decision_tree5,out_file ="myDecTree5.dot",filled=True,rounded=True)
call(['dot', '-T', 'png', 'myDecTree5.dot', '-o', 'myDecTree5.png'])

