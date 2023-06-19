#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_path = 'D:\Project\Students\Wine_quality.csv'


# In[4]:


wine = pd.read_csv(file_path)
wine


# In[7]:


wine.info()


# In[10]:


wine.columns


# In[11]:


wine.isnull().sum()


# In[12]:


mean_fixed_acidity = np.mean(wine['fixed acidity'])
mean_fixed_acidity


# In[14]:


wine['fixed acidity'].fillna(mean_fixed_acidity, inplace = True)


# In[15]:


wine['fixed acidity']


# In[16]:


wine['fixed acidity'].isnull().sum()


# In[19]:


wine.isna().sum()


# In[18]:


wine['volatile acidity'].describe()


# In[22]:


wine.hist(bins=60, figsize=(20,10))


# In[25]:


wine[['type', 'volatile acidity']].plot(kind='scatter', y='type', x='volatile acidity', figsize=(20,10))


# In[26]:


mean_volatile_acidity = np.mean(wine['volatile acidity'])
mean_volatile_acidity


# In[27]:


wine['volatile acidity'].fillna(mean_volatile_acidity, inplace = True)


# In[28]:


wine[['type', 'volatile acidity']].plot(kind='scatter', y='type', x='volatile acidity', figsize=(20,10))


# In[29]:


wine['volatile acidity'].hist(bins=60, figsize=(20,10))


# In[32]:


wine['fixed acidity'].plot.box()


# In[36]:


q1 = wine['fixed acidity'].quantile(0.25)
q3 = wine['fixed acidity'].quantile(0.75)
iqr =q3 - q1
lower_bound = q1 - 1.5*iqr
upper_bound = q3 + 1.5*iqr
wine['fixed acidity'] = np.where(wine['fixed acidity'] < lower_bound, lower_bound, wine['fixed acidity'])
wine['fixed acidity'] = np.where(wine['fixed acidity'] > upper_bound, upper_bound, wine['fixed acidity'])
wine['fixed acidity'].plot.box()


# In[37]:


numerical_features = wine.select_dtypes(include = 'number').columns.tolist()
categorical_features = wine.select_dtypes(exclude = 'number').columns.tolist()


# In[38]:


numerical_features


# In[39]:


categorical_features


# In[40]:


def treat_outlier(numerical_features):
    for column in numerical_features:
        q1 = wine[column].quantile(0.25)
        q3 = wine[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        
        wine[column] = np.where(wine[column] < lower_bound, lower_bound, wine[column])
        wine[column] = np.where(wine[column] > upper_bound, upper_bound, wine[column])
        
treat_outlier(numerical_features)


# In[41]:


wine.plot.box()


# In[43]:


wine.hist(bins = 20, figsize = (20,10))


# In[49]:


new_path = 'D:\\Project\Students\\cleaned_wine_quality.csv'
wine.to_csv(new_path, index = False, header = True)


# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure()
ax = fig.add_subplot(111)
wine['volatile acidity'].plot.density(color = 'red')


# In[ ]:





# In[ ]:




