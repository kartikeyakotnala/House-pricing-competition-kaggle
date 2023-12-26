#!/usr/bin/env python
# coding: utf-8

# In[72]:


from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[73]:


df=pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\Python\housing\train.csv")


# In[74]:


df.head()


# In[61]:


df.isna().count()


# In[62]:


df.info()


# In[307]:


y=df.SalePrice
y


# In[ ]:





# In[679]:


X_df=df[['MSSubClass','MSZoning','LotFrontage',
        'PoolArea','Foundation','Neighborhood',
        'YrSold','SaleCondition','BldgType', 'Street', 'LotArea',
        'OverallQual','OverallCond','ExterQual',
        'ExterCond','GrLivArea','BedroomAbvGr','RoofStyle',
        'KitchenAbvGr','SalePrice','LotConfig','HouseStyle','Condition1',
    ]].copy()
#'' 'Condition2','HeatingQC''KitchenQual','TotRmsAbvGrd',


# In[680]:


X_df.info()


# In[681]:


le=LabelEncoder()


# In[682]:


X_df.MSZoning =le.fit_transform(X_df.MSZoning)
X_df.SaleCondition = le.fit_transform(X_df.SaleCondition)
X_df.BldgType =le.fit_transform(X_df.BldgType)
X_df.ExterQual = le.fit_transform(X_df.ExterQual)
X_df.ExterCond =le.fit_transform(X_df.ExterCond)
X_df.Foundation =le.fit_transform(X_df.Foundation)
X_df.LotConfig =le.fit_transform(X_df.LotConfig)
X_df.HouseStyle =le.fit_transform(X_df.HouseStyle)
X_df.Street =le.fit_transform(X_df.Street)
X_df.Neighborhood =le.fit_transform(X_df.Neighborhood)
X_df.Condition1 =le.fit_transform(X_df.Condition1)
#X_df.Condition2 =le.fit_transform(X_df.Condition2)
X_df.RoofStyle =le.fit_transform(X_df.RoofStyle)
#X_df.HeatingQC =le.fit_transform(X_df.HeatingQC)
#X_df.KitchenQual =le.fit_transform(X_df.KitchenQual)


# In[ ]:





# In[683]:


print(X_df.shape)


# In[684]:


X_df.dropna(inplace=True)


# In[685]:


print(X_df.shape)


# In[686]:


pca=PCA(0.98)


# In[687]:


model=LinearRegression()


# In[688]:


X_pca=X_df.drop(['SalePrice'],axis='columns')


# In[689]:


X_test,X_train,y_test,y_train=train_test_split(X_pca,X_df['SalePrice'])


# In[690]:


model.fit(X_train,y_train)


# In[691]:


ypredict=model.predict(X_test)
ypredict=pd.DataFrame(ypredict,columns=['SalePrice'])


# In[692]:


model.score(X_test,y_test)


# In[693]:


#predicitng


# In[709]:


df1=pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\Python\housing\test.csv")


# In[710]:


X_df1=df1[['MSSubClass','MSZoning','LotFrontage',
        'PoolArea','Foundation','Neighborhood',
        'YrSold','SaleCondition','BldgType', 'Street', 'LotArea',
        'OverallQual','OverallCond','ExterQual',
        'ExterCond','GrLivArea','BedroomAbvGr','RoofStyle',
        'KitchenAbvGr','LotConfig','HouseStyle','Condition1',
    ]].copy()
#'' 'Condition2','HeatingQC''KitchenQual','TotRmsAbvGrd',


# In[711]:


le=LabelEncoder()
X_df1.info()


# In[713]:


X_df1.MSZoning =le.fit_transform(X_df1.MSZoning)
X_df1.SaleCondition = le.fit_transform(X_df1.SaleCondition)
X_df1.BldgType =le.fit_transform(X_df1.BldgType)
X_df1.ExterQual = le.fit_transform(X_df1.ExterQual)
X_df1.ExterCond =le.fit_transform(X_df1.ExterCond)
X_df1.Foundation =le.fit_transform(X_df1.Foundation)
X_df1.LotConfig =le.fit_transform(X_df1.LotConfig)
X_df1.HouseStyle =le.fit_transform(X_df1.HouseStyle)
X_df1.Street =le.fit_transform(X_df1.Street)
X_df1.Neighborhood =le.fit_transform(X_df1.Neighborhood)
X_df1.Condition1 =le.fit_transform(X_df1.Condition1)
#X_df.Condition2 =le.fit_transform(X_df.Condition2)
X_df1.RoofStyle =le.fit_transform(X_df1.RoofStyle)
#X_df.HeatingQC =le.fit_transform(X_df.HeatingQC)
#X_df.KitchenQual =le.fit_transform(X_df.KitchenQual)


# In[ ]:





# In[724]:


X_df1.MSZoning.fillna(X_df1.MSZoning.median(),inplace=True)
X_df1.SaleCondition.fillna(X_df1.SaleCondition.median(),inplace=True)
X_df1.BldgType.fillna(X_df1.BldgType.median(),inplace=True)
X_df1.ExterQual.fillna(X_df1.ExterQual.median(),inplace=True)
X_df1.fillna(X_df1.ExterCond.median(),inplace=True)
X_df1.Foundation.fillna(X_df1.Foundation.median(),inplace=True)
X_df1.LotConfig.fillna(X_df1.LotConfig.median(),inplace=True)
X_df1.HouseStyle.fillna(X_df1.HouseStyle.median(),inplace=True)
X_df1.Street.fillna(X_df1.Street.median(),inplace=True)
X_df1.Neighborhood.fillna(X_df1.median(),inplace=True)
X_df1.Condition1.fillna(X_df1.Condition1.median(),inplace=True)
#X_df.Condition2 =le.fit_transform(X_df.Condition2)
X_df1.RoofStyle.fillna(X_df1.RoofStyle.median(),inplace=True)
X_df1.MSSubClass.fillna(X_df1.MSSubClass.median(),inplace=True)
X_df1.LotFrontage.fillna(X_df1.LotFrontage.median(),inplace=True)
X_df1.PoolArea.fillna(X_df1.PoolArea.median(),inplace=True)       
X_df1.YrSold.fillna(X_df1.YrSold.median(),inplace=True)     
X_df1.LotArea.fillna(X_df1.LotArea.median(),inplace=True)        
X_df1.OverallQual.fillna(X_df1.OverallQual.median(),inplace=True)   
X_df1.OverallCond.fillna(X_df1.OverallCond.median(),inplace=True)    
X_df1.GrLivArea.fillna(X_df1.GrLivArea.median(),inplace=True)    
X_df1.BedroomAbvGr.fillna(X_df1.BedroomAbvGr.median(),inplace=True)   
X_df1.KitchenAbvGr.fillna(X_df1.KitchenAbvGr.median(),inplace=True)   


# In[725]:


X_df1.info()


# In[726]:


y_df1=model.predict(X_df1)

X_df1['SalesPrice']=pd.DataFrame(y_df1, columns=['SalesPrice'])


# In[727]:


result=pd.concat([df1['Id'],X_df1['SalesPrice']],axis='columns')


# In[728]:


result


# In[730]:


result.to_csv(r"C:\Users\karti\OneDrive\Desktop\Python\housing\trainmodel.csv")


# In[ ]:




