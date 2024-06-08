#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[38]:


df=pd.read_csv(r"C:\Users\uniqu\Downloads\archive\hiring.csv")
df


# In[39]:


experience_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                      'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                      'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
                      'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
                      'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'twenty-one': 21,
                      'twenty-two': 22, 'twenty-three': 23, 'twenty-four': 24,
                      'twenty-five': 25}

df['experience'] = df['experience'].map(experience_mapping)


# In[40]:


df.dropna(inplace=True)
df


# In[41]:


df


# In[42]:


model=linear_model.LinearRegression()
model.fit(df.drop('salary($)', axis='columns'), df['salary($)'])


# In[45]:


model.predict([[20,9,8]])


# In[ ]:




