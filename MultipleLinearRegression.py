import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("C:\\Users\\hp\\Desktop\\Courses\\Machine Learning\\Multiple Linear Regression #4\\odev_tenis.csv")
#Data preprocessing
#result = df.info()
#result = df.describe()
#Label Encoding
from sklearn.preprocessing import LabelEncoder
tempDf = df[['outlook','windy','play']]
tempDf = tempDf.apply(LabelEncoder().fit_transform)

#One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
outlook = tempDf.iloc[:,:1]
outlook = ohe.fit_transform(outlook).toarray()
#Creating new dataframe
outlook = pd.DataFrame(data=outlook, index=range(14), columns=['overcast','rainy','sunny'])
newDf = pd.concat([outlook,df.iloc[:,1:3],tempDf.iloc[:,1:3]],axis=1)
#Train preprocessing
y = newDf['humidity']
x = newDf.drop('humidity',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#lr.fit(x_train,y_train)
#y_predict = lr.predict(x_test)
#print(y_predict)
#print(y_test)

#Backward Elimination
import statsmodels.api as sm
array = np.append(arr=np.ones((14,1)).astype(int),values=newDf, axis=1)
#Converting x to Df
newDf = pd.DataFrame(data=array,index=range(14),columns=['const','overcast','rainy','sunny','temperature','humidity','windy','play'])

#x_list = newDf.drop('humidity',axis=1).values #Independent variables
#model = sm.OLS(y,x_list).fit()
#print(model.summary()) #Windy's p-value is high. We gonna eliminate it.


x_list = newDf.drop(['humidity','windy'],axis=1).values #Independent variables
model = sm.OLS(y,x_list).fit()

x_train = x_train.iloc[:,[0,1,2,3,5]]
x_test= x_test.iloc[:,[0,1,2,3,5]]
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)
print(y_test,y_predict)