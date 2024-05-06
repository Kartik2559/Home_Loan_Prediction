#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier


# In[7]:


train=pd.read_csv(r"C:\Users\kartik.1\Documents\home\train.csv")
test=pd.read_csv(r"C:\Users\kartik.1\Documents\home\test.csv")


# In[8]:


train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)


# In[9]:


train.dropna(inplace = True)
test.dropna(inplace = True)


# In[10]:


to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}

# adding the new numeric values from the to_numeric variable to both datasets
train = train.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
test = test.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)


# In[15]:


# convertind the Dependents column
Dependents_ = pd.to_numeric(train.Dependents)
Dependents__ = pd.to_numeric(test.Dependents)

# dropping the previous Dependents column
train.drop(['Dependents'], axis = 1, inplace = True)
test.drop(['Dependents'], axis = 1, inplace = True)

# concatination of the new Dependents column with both datasets
train = pd.concat([train, Dependents_], axis = 1)
test = pd.concat([test, Dependents__], axis = 1)


# In[20]:


x = train.drop('Loan_Status', axis = 1)
y = train['Loan_Status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[21]:


rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

y_predict = rfc.predict(x_test)

#  prediction Summary by species
print(classification_report(y_test, y_predict))

# Accuracy score
rfc_accuracy = accuracy_score(y_predict,y_test)
print(f"{round(rfc_accuracy*100,2)}% Accurate")


# In[22]:


import pickle

# Save the trained model to a file
with open('home_app_model.pkl', 'wb') as file:
    pickle.dump(rfc, file)


# In[ ]:




