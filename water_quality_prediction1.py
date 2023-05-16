#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\ADMIN\Downloads\water_potability.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df['Sulfate'].mean()


# In[8]:


df.fillna(df.mean(), inplace=True)
df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.Potability.value_counts()


# In[13]:


df.Potability.value_counts().plot(kind="bar", color=["brown", "salmon"])
plt.show()


# In[14]:


sns.distplot(df['ph'])


# In[15]:


df.hist(figsize=(14,14))
plt.show()


# In[16]:


sns.pairplot(df,hue='Potability')


# In[17]:


sns.scatterplot(df['Hardness'])


# In[ ]:


sns.heatmap(df.corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[19]:


df.boxplot(figsize=(14,7))


# In[20]:


df['Solids'].describe()


# In[21]:


X = df.drop('Potability',axis=1)


# In[22]:


Y= df['Potability']


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101,shuffle=True)


# In[24]:


Y_train.value_counts()


# In[25]:


Y_test.value_counts()


# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 10, splitter= 'best')
dt.fit(X_train,Y_train)


# In[27]:


prediction=dt.predict(X_test)
accuracy_dt=accuracy_score(Y_test,prediction)*100
accuracy_dt


# In[28]:


print("Accuracy on training set: {:.3f}".format(dt.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, Y_test)))


# In[29]:


accuracy_score(prediction,Y_test)


# In[30]:


print("Feature importances:\n{}".format(dt.feature_importances_))


# In[31]:


confusion_matrix(prediction,Y_test)


# In[32]:


X_DT=dt.predict([[5.735724, 158.318741,25363.016594,7.728601,377.543291,568.304671,13.626624,75.952337,4.732954]])


# In[33]:


print(*X_DT)


# In[34]:


X_DT=dt.predict([[0,0,0,0,0,0,0,0,0]])


# In[35]:


print(*X_DT)


# In[ ]:


a=float(input("Enter ph:"))
b=float(input("Enter Hardness:"))
c=float(input("Enter Solids:"))
d=float(input("Enter Chloramines:"))
e=float(input("Enter Sulfate:"))
f=float(input("Enter Conductivity:"))
g=float(input("Enter Organic_carbon:"))
h=float(input("Enter Trihalomethanes:"))
i=float(input("Enter Turbidity:"))


# In[ ]:


X_DT=dt.predict([[a,b,c,d,e,f,g,h,i]])


# In[ ]:


print(*X_DT)


# In[ ]:




