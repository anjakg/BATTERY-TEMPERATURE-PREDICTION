#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


train= pd.read_csv('/Users/anjugeorge/Desktop/Train_Dataset_NCACell6_allTemp_PH30_TH20_3inputs_T_Q_AmbT.csv',
                    names=['Temp','current','Ambient Temp','Final Temp'], header=None)


# In[ ]:


#train['time'] = range(1, len(train)+1)


# In[ ]:


train


# In[ ]:


train.shape


# In[ ]:


train.isna().sum()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.info()


# In[ ]:


#EDA


# In[ ]:


plt.figure(figsize=(15,3))
#plt.grid()
plt.plot(train['Temp'],  label='TEMPERATURE')

plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(15,3))
#plt.grid()
plt.plot(train['current'], label='current')
plt.legend()
plt.show()


# In[ ]:


test= pd.read_csv('/Users/anjugeorge/Desktop/test_data.csv',
                    names=['Temp','current','Ambient Temp','Final Temp'], header=None)
#test['time'] = range(1, len(test)+1)


# In[ ]:


test.shape


# In[ ]:


test


# In[ ]:



test.head()


# In[ ]:


test.tail()


# In[ ]:


#preprocessing


# In[ ]:


X_train= train.drop(['Final Temp'], axis=1)

y_train = train['Final Temp']


# In[ ]:


X_test= test.drop(['Final Temp'], axis=1)

y_test = test['Final Temp']


# In[ ]:


#ALGORITHMS


# In[ ]:


#SUPPORT VECTOR REGRESSION
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train,y_train)

y_pred = regr.predict(X_test)
y_pred
print(np.shape(y_pred))


# In[ ]:


a=pd.DataFrame(y_test)
a=a.reset_index()
a=a.drop(columns='index')
a


prediction=pd.DataFrame(y_pred)
prediction=prediction.reset_index()
prediction=prediction.drop(columns='index')
prediction
output=pd.DataFrame()
output['current_temperature']=a
output['final_temp 30s_Prediction']=prediction[0]
output['Difference']=output['current_temperature']-output['final_temp 30s_Prediction']
output['% Change']=abs(output['Difference']/output['current_temperature'])*100
output


# In[ ]:



mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("max_error for support vector regression;  ", max_error(y_test, y_pred))
print( "mean_absolute_error for  support vector regression; ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error for support vector regression: ", mse)
print("Root Mean Squared Error of support vector regression: ", rmse)
print("R2_score of support vector regression ",r2_score(y_test, y_pred))
avg_err = output['Difference'].mean()
print ('The average error is:   ',avg_err)


# In[ ]:


#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression().fit(X_train,y_train)
y_predict = lin_reg.predict(X_test)
y_predict


# In[ ]:


a=pd.DataFrame(y_test)
a=a.reset_index()
a=a.drop(columns='index')
a

prediction=pd.DataFrame(y_predict)
prediction=prediction.reset_index()
#prediction=prediction.drop(columns='index')
prediction
output=pd.DataFrame()
output['current_temperature']=a
output['final_temp 30s_Prediction']=prediction[0]
output['Difference']=output['current_temperature']-output['final_temp 30s_Prediction']
output['% Change']=abs(output['Difference']/output['current_temperature'])*100
output


# In[ ]:


mse = mean_squared_error(y_test, y_predict)
rmse = mse ** 0.5
print("MSE: ", mse)
print("RMSE: ", rmse)
from sklearn.metrics import max_error
print("max_error;  ", max_error(y_test, y_predict))
from sklearn.metrics import mean_absolute_error
print( "mean_absolute_error; ", mean_absolute_error(y_test, y_predict))
print("R2_score. ",r2_score(y_test, y_predict))
avg_err_lr = output['Difference'].mean()
print ('The average error is:    ',avg_err_lr)


# In[ ]:


#DECISION TREE REGRESSOR

dtr = DecisionTreeRegressor().fit(X_train,y_train)
#DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                      #max_features=None, max_leaf_nodes=None,
                      #min_impurity_decrease=0.0, min_impurity_split=None,
                      #min_samples_leaf=1, min_samples_split=2,
                      #min_weight_fraction_leaf=0.0, presort='deprecated',
                      #random_state=None, splitter='best') 
 
y_p = dtr.predict(X_test)
y_p


# In[ ]:


a=pd.DataFrame(y_test)
a=a.reset_index()
a=a.drop(columns='index')
a

prediction=pd.DataFrame(y_p)
prediction=prediction.reset_index()
#prediction=prediction.drop(columns='index')
prediction
output=pd.DataFrame()
output['current_temperature']=a
output['final_temp 30s_Prediction']=prediction[0]
output['Difference']=output['current_temperature']-output['final_temp 30s_Prediction']
output['% Change']=abs(output['Difference']/output['current_temperature'])*100
output


# In[ ]:


from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(y_test, y_p)
rmse = mse ** 0.5
print("MSE: ", mse)
print("RMSE: ", rmse)
print("max_error;  ", max_error(y_test, y_p))
print( "mean_absolute_error; ", mean_absolute_error(y_test, y_p))
print("R2_score. ",r2_score(y_test, y_p))
avg_err_dt = output['Difference'].mean()
print ('The average error is:    ',avg_err_dt)


# In[ ]:


#RIDGE REGRESSION

rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train) 
pred_test_rr= rr.predict(X_test)
pred_test_rr


# In[ ]:


X_test


# In[ ]:


a=pd.DataFrame(y_test)
a=a.reset_index()
a=a.drop(columns='index')
a

prediction=pd.DataFrame(pred_test_rr)
prediction=prediction.reset_index()
#prediction=prediction.drop(columns='index')
prediction
output=pd.DataFrame()
output['current_temperature']=a
output['final_temp 30s_Prediction']=prediction[0]
output['Difference']=output['current_temperature']-output['final_temp 30s_Prediction']
output['% Change']=abs(output['Difference']/output['current_temperature'])*100
output


# In[ ]:


mse = mean_squared_error(y_test,pred_test_rr)
rmse = mse ** 0.5 
rmse = mse ** 0.5
print("MSE: ", mse)
print("RMSE: ", rmse)
print("max_error;  ", max_error(y_test, pred_test_rr))
print( "mean_absolute_error; ", mean_absolute_error(y_test, pred_test_rr))
print("R2_score. ",r2_score(y_test, pred_test_rr))
avg_err_rr = output['Difference'].mean()
print ('The average error is:    ',avg_err_rr)


# In[ ]:


#ELASTICNET REGRESSION

model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(X_train, y_train) 
pred_test_enet= model_enet.predict(X_test)
pred_test_enet


# In[ ]:


a=pd.DataFrame(y_test)
a=a.reset_index()
a=a.drop(columns='index')
a

prediction=pd.DataFrame(pred_test_enet)
prediction=prediction.reset_index()
#prediction=prediction.drop(columns='index')
prediction
output=pd.DataFrame()
output['current_temperature']=a
output['final_temp 30s_Prediction']=prediction[0]
output['Difference']=output['current_temperature']-output['final_temp 30s_Prediction']
output['% Change']=abs(output['Difference']/output['current_temperature'])*100
output


# 

# In[ ]:


mse = mean_squared_error(y_test,pred_test_enet)
rmse = mse ** 0.5 
rmse = mse ** 0.5
print("MSE: ", mse)
print("RMSE: ", rmse)
print("max_error;  ", max_error(y_test, pred_test_enet))
print( "mean_absolute_error; ", mean_absolute_error(y_test, pred_test_enet))
print("R2_score. ",r2_score(y_test, pred_test_enet))
avg_err_rr = output['Difference'].mean()
print ('The average error is:    ',avg_err_rr)


# In[ ]:


#PLOTTING THE GRAPHS


# In[ ]:


X_test['time'] = range(1, len(X_test)+1)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.scatter(X_test['time'],y_pred , c='red', label ='predicted')
plt.scatter(X_test['time'],y_test , c='green',label ='True')
plt.legend(loc="upper right", fontsize = 9)
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR SVR",fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Temperature', fontsize=15)


plt.show()


# In[ ]:


plt.figure(figsize=(15, 5))
plt.scatter(X_test['time'][:1000], y_pred[:1000], c='red', label='predicted')
plt.scatter(X_test['time'][:1000], y_test[:1000], c='green', label='true')
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR SVR REGRESSION FOR 1000s", fontsize=15)
plt.legend(loc="upper right", fontsize=9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Temperature', fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
plt.scatter(X_test['time'],y_predict, c='red',label ='predicted')
plt.scatter(X_test['time'],y_test , c='green',label ='true')
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR LINEAR REGRESSION",fontsize=15)
plt.legend(loc="upper right", fontsize = 9)
plt.xlabel('Time', fontsize=15)
plt.ylabel(' Temperature', fontsize=15)


plt.show()


# In[ ]:


plt.figure(figsize=(15, 5))
plt.scatter(X_test['time'][:1000], y_predict[:1000], c='red', label='predicted')
plt.scatter(X_test['time'][:1000], y_test[:1000], c='green', label='true')
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR LINEAR REGRESSION FOR 1000s", fontsize=15)
plt.legend(loc="upper right", fontsize=9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Temperature', fontsize=15)
plt.show()


# 

# In[ ]:


plt.figure(figsize=(15,5))
plt.scatter(X_test['time'],y_p, c='red',label ='predicted')
plt.scatter(X_test['time'],y_test , c='green',label ='true')
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR DECISION TREE REGRESSION",fontsize=15)
plt.legend(loc="upper right", fontsize = 9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Temperature', fontsize=15)


plt.show()


# In[ ]:


plt.figure(figsize=(15, 5))
plt.scatter(X_test['time'][:1000], y_p[:1000], c='red', label='predicted')
plt.scatter(X_test['time'][:1000], y_test[:1000], c='green', label='true')
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR DECISION TREE REGRESSION FOR 1000s", fontsize=15)
plt.legend(loc="upper right", fontsize=9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Temperature', fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
plt.scatter(X_test['time'],pred_test_rr, c='red',label ='Predicted')
plt.scatter(X_test['time'],y_test , c='green',label ='True')

plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR RIDGE REGRESSION",fontsize=15)
plt.legend(loc="upper right", fontsize = 9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Predicted temperature', fontsize=15)


plt.show()


# In[ ]:


plt.figure(figsize=(15, 5))
plt.scatter(X_test['time'][:1000], pred_test_rr[:1000], c='red', label='predicted')
plt.scatter(X_test['time'][:1000], y_test[:1000], c='green', label='true')

plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR RIDGE REGRESSION FOR 1000s", fontsize=15)
plt.legend(loc="upper right", fontsize=9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Temperature', fontsize=15)
plt.show()


# In[ ]:





# In[ ]:


plt.figure(figsize=(15,5))
plt.scatter(X_test['time'],pred_test_enet, c='red',label ='predicted')
plt.scatter(X_test['time'],y_test , c='green',label ='true')
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR ELASTICNET REGRESSION",fontsize=15)
plt.legend(loc="upper right", fontsize = 9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Predicted temperature', fontsize=15)


plt.show()


# In[ ]:


plt.figure(figsize=(15, 5))
plt.scatter(X_test['time'][:1000], pred_test_enet[:1000], c='red', label='predicted')
plt.scatter(X_test['time'][:1000], y_test[:1000], c='green', label='true')
plt.title("COMPARISON BETWEEN THE TEMPERATURE FOR ELASTICNET REGRESSION FOR 100s", fontsize=15)
plt.legend(loc="upper right", fontsize=9)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Temperature', fontsize=15)
plt.show()

