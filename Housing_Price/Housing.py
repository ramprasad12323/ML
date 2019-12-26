import pandas as pd

def nullMng(Data,Null_values='mean'):
    for i in range(Data.shape[1]-1):
        if Data[Data.columns[i]].dtype=='O':
            Data[Data.columns[i]][Data[Data.columns[i]].isnull()]='012789'
        else:
            if Null_values=='mean':
                Data[Data.columns[i]][Data[Data.columns[i]].isnull()]=Data[Data.columns[i]].mean()
            elif Null_values=='mode':
                Data[Data.columns[i]][Data[Data.columns[i]].isnull()]=Data[Data.columns[i]].mode()
            elif Null_values=='median':
                Data[Data.columns[i]][Data[Data.columns[i]].isnull()]=Data[Data.columns[i]].median()
            else:
                Data[Data.columns[i]][Data[Data.columns[i]].isnull()]=Null_values
    Data=Data.dropna()
    return Data
                
            

def dataEncoding(Data):
    Dumies=pd.DataFrame()
    for i in range(Data.shape[1]-1):
        if Data[Data.columns[i]].dtype=='O':
            dum=pd.get_dummies(data=Data[Data.columns[i]])
            if '012789' in dum.columns:
                dum=dum.drop(['012789'],'columns')
            else:
                dum=dum.drop(dum.columns[0],'columns')
            Dumies=pd.concat([Dumies,dum],'columns')
        else:
            pass
        
    y=Data[Data.columns[Data.shape[1]-1]]
    Data=Data.drop(Data.columns[Data.shape[1]-1],'columns')
    i=0
    while(i<Data.shape[1]):
        if Data[Data.columns[i]].dtype=='O':
            Data=Data.drop(Data.columns[i],'columns')
            i=i-1
        i=i+1
    
    Data=pd.concat([Data,Dumies,y],'columns')
    return Data
    
def nullElimination(Data,Threshold=.15):
    i=0
    while(i<Data.shape[1]-1):
        thr=Data[Data.columns[i]].shape[0]*Threshold
        if Data[Data.columns[i]][Data[Data.columns[i]].isnull()].shape[0]>thr:
            Data=Data.drop(Data.columns[i],'columns')
            i=i-1
        i=i+1
    return Data
    
def inAprenter(Data):
    Data = Data.applymap(lambda s:s.lower() if type(s) == str else s)
    return Data

Data=pd.read_csv('train.csv')
Data=inAprenter(Data)
Data=nullElimination(Data,0)
Data=nullMng(Data,Null_values='mean')
Data=dataEncoding(Data)

import matplotlib.pyplot as plt

plt.scatter(Data.values[:,0],Data.values[:,Data.shape[1]-1])
plt.show()

print(Data.shape)

for i in range(Data.shape[0]-1):
        if Data[Data.columns[Data.shape[1]-1]][i]>400000:
            Data=Data.drop([i],axis=0)
            
plt.scatter(Data.values[:,0],Data.values[:,Data.shape[1]-1])
plt.show()

print(Data.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
model=LinearRegression()
y=Data.values[:,Data.shape[1]-1]
X=Data.values[:,1:Data.shape[1]-1]

X,x,Y,y=train_test_split(X,y,test_size=0.25)

model.fit(X,Y)

print(((model.predict(x)-abs(model.predict(x)-y))/y).mean())
