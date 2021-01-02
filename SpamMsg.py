#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# In[6]:


ds = pd.read_csv('spam.csv',encoding='latin-1')


# In[8]:


ds.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1, inplace= True)


# In[9]:


ds.head()


# In[12]:


Encoder =LabelEncoder()
ds['v1'] = Encoder.fit_transform(ds['v1'])


# In[15]:


ds.groupby('v1').size().plot(kind='bar')


# In[16]:


ds.isnull().sum()


# In[20]:


import nltk
nltk.download('wordnet')


# In[21]:


#Creating corpus

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
ps=WordNetLemmatizer()
corpus=[]
for i in range(0,len(ds)):
    reviews=re.sub('[^a-zA-Z]',' ',ds['v2'][i])
    reviews=reviews.lower()
    reviews=reviews.split()
    reviews=[ps.lemmatize(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews=' '.join(reviews)
    corpus.append(reviews)


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()


# In[23]:


x


# In[49]:


y=ds.iloc[:,[0]].values


# In[50]:


y


# In[60]:


x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=25)


# In[70]:


LinReg=LinearRegression(x,)


# In[71]:


LinReg.fit(x_train,y_train)


# In[72]:


LinReg.score(x_train,y_train)


# In[73]:


y_pred= LinReg.predict(x_test)


# In[74]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[75]:


cm =confusion_matrix(y_test,y_pred)
cm


# In[77]:


accuracy_score(y_test,y_pred)


# In[78]:


from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(n_estimators=500,random_state=0)


# In[81]:


RFC.fit(x_train,y_train)
y_pred_2=RFC.predict(x_test)


# In[82]:


cm=confusion_matrix(y_test,y_pred_2)
cm


# In[83]:


accuracy_score(y_test,y_pred_2)

