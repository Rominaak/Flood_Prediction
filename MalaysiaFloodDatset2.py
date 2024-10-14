import numpy as np
import pandas as pd

data = pd.read_csv('MalaysiaFloodDatset.csv')
df= pd.DataFrame(data)
x = df.iloc[1:825,4:16]#features
y = df.iloc[1:825,16]#label


print(x)
print(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
df2 = pd.DataFrame({'ytest':ytest,'ypred':ypred})
print(df2)

from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))
