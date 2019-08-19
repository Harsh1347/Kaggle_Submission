#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('Data Set',data.shape)
print('Test Set',test.shape)


# In[2]:


print('Data Set \n',data.isnull().sum())
print('---------------')
print('Test Set \n',test.isnull().sum())


# In[3]:


data.head()


# In[4]:


test.head()


# In[ ]:





# # Visualization

# In[5]:


sns.set()


# In[6]:


def bar_ch(para,tf):
    survived = data[data['Survived']==1][para].value_counts()
    dead = data[data['Survived']==0][para].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index=['Survived','Dead']
    df.plot(kind='bar',stacked = tf,figsize =(10,5))


# In[7]:


bar_ch('Sex',True)


# Females are more likely to Survive than male

# In[8]:


bar_ch('Pclass',False)


# Out Of survived 1st Class has more Survival Chance and 2nd has the least.
# 3rd class are most likely to have died

# In[9]:


bar_ch('SibSp',False)


# In[10]:


bar_ch('Parch',False)


# In[11]:


bar_ch('Embarked',False)


# In[12]:


data.Sex.replace(['male','female'],[1,0],inplace = True)
test.Sex.replace(['male','female'],[1,0],inplace = True)
data.head()


# In[ ]:





# In[13]:


data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'].value_counts()
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'].value_counts()


# In[14]:


bar_ch('Title',True)


# In[15]:


data.Title.replace(['Mr','Miss','Mrs','Master','Dr','Rev','Col','Major','Mlle','Don','Ms','Lady','Capt','Sir','Mme','Countess','Jonkheer'],[0,1,2,3,0,4,4,4,4,4,1,2,4,4,4,4,4],inplace = True)
test.Title.replace(['Mr','Miss','Mrs','Master','Dr','Rev','Col','Major','Mlle','Don','Ms','Lady','Capt','Sir','Mme','Countess','Jonkheer','Dona'],[0,1,2,3,0,4,4,4,4,4,1,2,4,4,4,4,4,4],inplace = True)

0: Mr,Dr
1:Miss,Ms
2:Mrs,Lady
3:Master
4:Rev,Col,Major,Mlle,Don,Cap,Sir,Mme,Countess,Jonkheer,Dona
# In[16]:


data.drop('Name',axis = 1,inplace = True)
test.drop('Name',axis = 1,inplace = True)
data.drop('Ticket',axis = 1,inplace = True)
test.drop('Ticket',axis = 1,inplace = True)


# In[17]:


data.Embarked.replace(['S','C','Q'],[0,1,2],inplace = True)
test.Embarked.replace(['S','C','Q'],[0,1,2],inplace = True)
data.head()

S:0
C:1
Q:2
# In[18]:


data['Age'].fillna(data.groupby('Title')['Age'].transform('median'),inplace = True)
test['Age'].fillna(data.groupby('Title')['Age'].transform('median'),inplace = True)


# In[19]:


data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'),inplace = True)
test['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'),inplace = True)


# In[20]:


data['Embarked'].fillna(3,inplace = True)
test['Embarked'].fillna(3,inplace = True)


# In[21]:


data.isnull().sum()


# In[22]:


data['Cabin']= data['Cabin'].str[:1]
data['Cabin']


# In[23]:


test['Cabin']= test['Cabin'].str[:1]
data.Cabin.value_counts()


# In[24]:


bar_ch('Cabin',True)


# In[25]:


data.Cabin.replace(['A','B','C','D','E','F','G','T'],[0,0.4,0.8,1.2,1.6,2.0,2.4,2.8], inplace = True)
data['Cabin'].fillna(data.groupby('Title')['Cabin'].transform('median'),inplace = True)
test.Cabin.replace(['A','B','C','D','E','F','G','T'],[0,0.4,0.8,1.2,1.6,2.0,2.4,2.8], inplace = True)
test['Cabin'].fillna(data.groupby('Title')['Cabin'].transform('median'),inplace = True)
data.info()


# In[ ]:





# In[ ]:





# In[26]:


print('Data Set \n',data.isnull().sum())
print('-------xxxx-------')
print('Test Set \n',test.isnull().sum())
data.Fare


# In[27]:


train_test_data = [data, test]
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 18, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 20, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 50), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 50) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:





# In[ ]:





# In[28]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[29]:


data.head()


# In[30]:


#data['Embarked'].fillna(0,inplace =True)
data['Embarked'].isnull().sum()


# In[31]:


data.corr()


# In[32]:


## sib parch not req


# ## KNN

# In[33]:


x_train,x_test,y_train,y_test = train_test_split(data[['Pclass','Sex','Age','Fare','Cabin','Embarked','Title']] ,data['Survived'] ,test_size = 0.35,random_state = 4)


# In[34]:


#res = []
#for n in range(3,30):
knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
#acc =metrics.accuracy_score(knn.predict(x_test),y_test)
#res.append(acc)
#plt.plot([x for x in range(3,30)],res)
metrics.accuracy_score(knn.predict(x_test),y_test)


# 

# In[95]:


prediction = knn.predict(test[['Pclass','Sex','Age','Fare','Cabin','Embarked','Title']])


# In[96]:


sol = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })


# In[97]:


sol


# In[98]:


sol.to_csv('sol6.csv', index=False)


# # LOGISTIC REGRESSION

# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
prediction3 = lr.predict(test[['Pclass','Sex','Age','Fare','Cabin','Embarked','Title']])
metrics.accuracy_score(lr.predict(x_test),y_test)


# In[106]:


prediction3


# In[107]:


sol4 = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction3
    })


# In[108]:


sol4.to_csv('sol.csv',index=False)


# ## SVM

# In[37]:


sm = SVC()
sm.fit(data[['Pclass','Sex','Age','Fare','Cabin','Embarked','Title']],data['Survived'])


# In[38]:


pred_svm = sm.predict(test[['Pclass','Sex','Age','Fare','Cabin','Embarked','Title']])
metrics.accuracy_score(sm.predict(x_test),y_test)


# In[40]:


subb = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred_svm
    })
subb.to_csv('submission1.csv', index=False)


# In[37]:


subb


# ## SVM model gave the hightest accuracy(86.538)
# ## hence svm model was submitted in the competition which gave us a score of 0.799 and a rank of #1592 at the time of submission

# In[ ]:




