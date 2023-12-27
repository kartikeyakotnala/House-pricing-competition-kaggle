#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# In[25]:


df=pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\Python\housing\train.csv")


# In[26]:


dfpp=df[['MSSubClass','LotFrontage','LotArea',
        'OverallQual','OverallCond','YearBuilt',
        'YearRemodAdd','MasVnrArea','BsmtFinSF1',
        'BsmtUnfSF','TotalBsmtSF','1stFlrSF',
        '2ndFlrSF','GrLivArea','BsmtFullBath',
        'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
        'KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
        'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
        'MoSold','YrSold','SalePrice']].copy()


# In[27]:


dfpp.info()


# In[28]:


dfpp.dropna(subset=['SalePrice'])
dfpp['LotFrontage']=dfpp['LotFrontage'].fillna(dfpp['LotFrontage'].median())
dfpp['MasVnrArea']=dfpp['MasVnrArea'].fillna(dfpp['MasVnrArea'].median())


# In[29]:


dfpp.isna().sum()


# In[30]:


y=dfpp['SalePrice']


# In[33]:


dfpp.drop('SalePrice',inplace=True,axis='columns')


# In[36]:


X=dfpp


# In[77]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)


# In[78]:


model=LinearRegression()


# In[79]:


model.fit(X_train,y_train)


# In[80]:


model.score(X_test,y_test)


# In[ ]:


#prediction

