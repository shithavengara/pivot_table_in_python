#!/usr/bin/env python
# coding: utf-8

# EDA

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') #Importing libraries


# In[15]:


df = pd.read_csv('train.csv') #Read file
df.head() 


# In[3]:


df.drop(['PassengerId','Ticket','Name'],inplace=True,axis=1)  #Droping few features


# In[4]:


table = pd.pivot_table(data=df,index=['Sex']) # ‘Sex’ column as the index,single index
table


# In[5]:


table.plot(kind='bar');


# In[6]:


table = pd.pivot_table(df,index=['Sex','Pclass']) #multiple indexes
table


# In[16]:


table = pd.pivot_table(df,index=['Sex','Pclass'],aggfunc={'Age':np.mean,'Survived':np.sum}) #aggregate functions
table


# In[8]:


table = pd.pivot_table(df,index=['Sex','Pclass'],values=['Survived'], aggfunc=np.mean)
table


# In[9]:


table.plot(kind='bar');


#  Relationship between features with columns parameter

# In[10]:


table = pd.pivot_table(df,index=['Sex'],columns=['Pclass'],values=['Survived'],aggfunc=np.sum)
table


# In[12]:


table.plot(kind='bar')


#  missing data

# In[13]:


table = pd.pivot_table(df,index=['Sex','Survived','Pclass'],columns=['Embarked'],values=['Age'],aggfunc=np.mean) #nullvalues
table


# In[17]:


table = pd.pivot_table(df,index=['Sex','Survived','Pclass'],columns=['Embarked'],values=['Age'],aggfunc=np.mean,fill_value=np.mean(df['Age'])) #replacing the NaN values with the mean value from the ‘Age’ column
table


# In[ ]:




