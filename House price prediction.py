#!/usr/bin/env python
# coding: utf-8

# In[133]:


import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


# In[134]:


df=pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\Python\housing\train.csv")


# In[140]:


dfppord=df[['MSSubClass','LotArea',
        'OverallQual','OverallCond','YearBuilt',
        'YearRemodAdd','BsmtFinSF1',
        'BsmtUnfSF','TotalBsmtSF','1stFlrSF',
        '2ndFlrSF','GrLivArea','BsmtFullBath',
        'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
        'KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
        'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
        'MoSold','YrSold','SalePrice']].copy()
#'LotFrontage' MasVnrArea


# In[141]:


#testing outliers here
T=dfppord[df['LotArea']<df['GrLivArea']].copy()
pd.DataFrame(T)


# In[262]:


dfppcat=df[['MSZoning','LotConfig','Neighborhood',
            'RoofStyle','ExterQual','ExterCond',
            'Foundation','SalePrice']].copy()
#'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1' 'RoofMatl','Exterior1st','Exterior2nd'


# In[263]:


dfppord.info()


# In[264]:


dfppord.dropna(subset=['SalePrice'])
#dfppord['LotFrontage']=dfppord['LotFrontage'].fillna(dfppord['LotFrontage'].median())
#dfppord['MasVnrArea']=dfppord['MasVnrArea'].fillna(dfppord['MasVnrArea'].median())


# In[265]:


#handling NaN values
dfppcat.dropna(subset=['SalePrice'])
#dfppcat['BsmtQual']=dfppcat['BsmtQual'].fillna(dfppcat['BsmtQual'].mode()[0])
#dfppcat['BsmtCond']=dfppcat['BsmtCond'].fillna(dfppcat['BsmtCond'].mode()[0])
#dfppcat['BsmtExposure']=dfppcat['BsmtExposure'].fillna(dfppcat['BsmtExposure'].mode()[0])
#dfppcat['BsmtFinType1']=dfppcat['BsmtFinType1'].fillna(dfppcat['BsmtFinType1'].mode()[0])
dfppcat.info()


# In[266]:


#creating Dummies
dfppcatdummies=pd.get_dummies(dfppcat,columns=['MSZoning','LotConfig','Neighborhood',
            'RoofStyle','ExterQual','ExterCond',
            'Foundation'])
dfppcatdummies.drop(['SalePrice'],axis='columns',inplace=True)
#'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1' 'RoofMatl','Exterior1st',   'Exterior2nd'


# In[267]:


dfppcatdummies.info()


# In[268]:


dfp=pd.concat([dfppord,dfppcatdummies],axis='columns')


# In[ ]:





# In[269]:


dfp.info()


# In[270]:


#removingOutliers


# In[271]:


dfp.drop(dfp[dfp['1stFlrSF']<(dfp['2ndFlrSF']*0.7)].index,inplace=True)
dfp.info()


# In[272]:


#assinging X and y


# In[273]:


y=dfp['SalePrice']
dfp.drop('SalePrice',inplace=True,axis='columns')


# In[274]:


X=dfp


# In[275]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05)


# In[276]:


model=LinearRegression()


# In[277]:


model.fit(X_train,y_train)


# In[278]:


model.score(X_test,y_test)


# In[279]:


cross_val_score(LinearRegression(),X,y,cv=7)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[280]:


#building prediction model


# In[281]:


dftest=pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\Python\housing\test.csv")


# In[282]:


dfppordtest=dftest[['MSSubClass','LotArea',
        'OverallQual','OverallCond','YearBuilt',
        'YearRemodAdd','BsmtFinSF1',
        'BsmtUnfSF','TotalBsmtSF','1stFlrSF',
        '2ndFlrSF','GrLivArea','BsmtFullBath',
        'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
        'KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
        'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
        'MoSold','YrSold']].copy()


# In[283]:


dfppcattest=dftest[['MSZoning','LotConfig','Neighborhood',
            'RoofStyle','ExterQual','ExterCond',
            'Foundation']].copy()
#'RoofMatl','Exterior1st', 'Exterior2nd'


# In[284]:


dfppordtest.info()


# In[285]:


dfppordtest['BsmtFinSF1']=dfppordtest['BsmtFinSF1'].fillna(dfppordtest['BsmtFinSF1'].mean())
dfppordtest['BsmtUnfSF']=dfppordtest['BsmtUnfSF'].fillna(dfppordtest['BsmtUnfSF'].mean())
dfppordtest['TotalBsmtSF']=dfppordtest['TotalBsmtSF'].fillna(dfppordtest['TotalBsmtSF'].mean())
dfppordtest['BsmtFullBath']=dfppordtest['BsmtFullBath'].fillna(dfppordtest['BsmtFullBath'].mean())
dfppordtest['BsmtHalfBath']=dfppordtest['BsmtHalfBath'].fillna(dfppordtest['BsmtHalfBath'].mean())
dfppordtest['GarageCars']=dfppordtest['GarageCars'].fillna(dfppordtest['GarageCars'].mean())
dfppordtest['GarageArea']=dfppordtest['GarageArea'].fillna(dfppordtest['GarageArea'].mean())


# In[286]:


dfppcattest.info()


# In[288]:


dfppcattest['MSZoning']=dfppcattest['MSZoning'].fillna(dfppcattest['MSZoning'].mode()[0])
#dfppcattest['Exterior1st']=dfppcattest['Exterior1st'].fillna(dfppcattest['Exterior1st'].mode()[0])
#dfppcattest['Exterior2nd']=dfppcattest['Exterior2nd'].fillna(dfppcattest['Exterior2nd'].mode()[0])


# In[289]:


dfppcatdummiestest=pd.get_dummies(dfppcattest,columns=['MSZoning','LotConfig','Neighborhood',
            'RoofStyle','ExterQual','ExterCond',
            'Foundation'])
#,'RoofMatl','Exterior1st', 'Exterior2nd'


# In[290]:


dfptest=pd.concat([dfppordtest,dfppcatdummiestest],axis='columns')


# In[291]:


ypredicted=model.predict(dfptest)


# In[292]:


dfptest['SalePrice']=pd.DataFrame(ypredicted,columns=['SalePrice'])


# In[294]:


result=pd.concat([dftest['Id'],dfptest['SalePrice']],axis='columns')


# In[295]:


result.to_csv(r"C:\Users\karti\OneDrive\Desktop\Python\housing\resulttest.csv")


# In[ ]:





# In[ ]:




