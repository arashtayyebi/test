#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import sklearn



from sklearn.ensemble import RandomForestRegressor


# Define the URL of the raw dataset on GitHub
url = "https://github.com/arashtayyebi/ss/raw/main/UND%20Data9.XLSX"

# Load the dataset from the URL
df = pd.read_excel(url)

#df = pd.read_excel('C:/Users/arash.tayyebi/Desktop/scale/data/done/UND Data9.xlsx')

Y = df.iloc[:,19:31]

X = df.iloc[:,0:19]

#data split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=991)
#test size is 20% and traing size is 80%
#linear regression
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

print (model.score(X_train, Y_train))

Y_pred_test = model.predict(X_test)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test,squared=False))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))

import pickle
pickle.dump(model, open('model19x11y.pkl','wb'))
#
#model=pickle.load (open('model.pkl','rb'))
#print (model.predict([[6.14,9.385,500,391.6,17.5,337.233,14970,1674280,204.3,6605,1727,13.38,83500,5.745,116.9,350.7,54.21,1089,14.85
#]]))


# In[61]:


print (model.predict([[6.14,9.385,391.6,17.5,337.233,14970,1674280,0,204.3,6605,1727,13.38,83500,5.745,116.9,350.7,54.21,1089,14.85]]))


# In[ ]:





# In[ ]:





# In[ ]:




