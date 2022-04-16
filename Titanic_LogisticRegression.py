#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train_data = pd.read_csv("train.csv") #reading training data set

test_data = pd.read_csv("test.csv") #reading test data set
test_id = test_data["PassengerId"]
train_data.head(20)
#test_data.head()
###trials###
#print(train_data.to_string())
#titanic_data = pd.read_csv(r'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
#print(titanic_data.head())
############


# In[2]:


def clean(data):
    data = data.drop(["Ticket", "Name", "PassengerId", "Cabin" ], axis = 1)
    
    cols = [ "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace = True)
        
    data.Embarked.fillna("U", inplace = True)
    return data
data = clean(train_data)
test = clean(test_data)
data.head(20)


# In[3]:


from sklearn import preprocessing
l_enc = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked"]

for col in cols:
    data[col] = l_enc.fit_transform(data[col])
    test[col] = l_enc.transform(test[col])
    print(l_enc.classes_)
    
data.head()


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = data["Survived"]
X = data.drop("Survived", axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[5]:


clf = LogisticRegression(random_state =0, max_iter = 1000).fit(X_train, y_train)


# In[6]:


pred = clf.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val, pred)


# In[7]:


sub_pred = clf.predict(test)


# In[11]:


df = pd.DataFrame({"PassengerId" : test_id.values,
                   "Survived" : sub_pred})


# In[12]:


df.to_csv("Submssion.csv", index = False)


# In[ ]:




