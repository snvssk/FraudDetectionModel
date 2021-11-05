import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix,accuracy_score,classification_report

import seaborn as sns
import matplotlib.pyplot as plt

# Loading dataset
df = pd.read_csv('payments.csv')
print(df.shape)
print(df.head())


#Checking for nulls
print(df.isnull().sum())

print(df.describe().T)

print(df.info())


# count number of fraud and not fraud data
print(df.isFraud.value_counts())


# Count number of data point in each type of transaction
print(df.type.value_counts())


# Investigate variable "isFlaggedFraud"
pd.crosstab(df.isFraud,df.isFlaggedFraud)


#Groupby type
print(df.groupby('type')['isFraud','isFlaggedFraud'].sum())


# Feature extraction
data = df.copy()

# Merchant flag for source and dist
data['OrigC']=data['nameOrig'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
data['DestC']=data['nameDest'].apply(lambda x: 1 if str(x).find('C')==0 else 0)

# flag for transfer and cashout from type feature
data['TRANSFER']=data['type'].apply(lambda x: 1 if x=='TRANSFER' else 0)
data['CASH_OUT']=data['type'].apply(lambda x: 1 if x=='CASH_OUT' else 0)



# Calculating Amount error
data['OrigAmntErr']=(abs(data.oldbalanceOrg-data.newbalanceOrig)-data.amount)

#print result
def model_result(clf,x_test,y_test):
    y_prob=clf.predict_proba(x_test)
    y_pred=clf.predict(x_test)
    print('AUPRC :', (average_precision_score(y_test, y_prob[:, 1])))
    print('F1_score :',(f1_score(y_test,y_pred)))
    print('Confusion_matrix : ')
    print(confusion_matrix(y_test,y_pred))
    print("accuracy_score")
    print(accuracy_score(y_test,y_pred))
    print("classification_report")
    print(classification_report(y_test,y_pred))


# droping those feature which are id type category and those which used for feature extraction
droplist=['isFlaggedFraud','type','nameDest','nameOrig']


from sklearn.ensemble import RandomForestClassifier

def stkflod_RF(X_train,y_train,X_test,y_test):   
    r_clf=RandomForestClassifier()
    r_clf.fit(X_train,y_train)
    print ('Test')
    print(model_result(r_clf,X_test,y_test))


from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)


# spliting data into X_train, X_test, y_train, y_test
# Creating X and y for spliting dataset into test and train
MLData=data.drop(labels=droplist,axis=1).head(100000)
X=MLData.drop('isFraud',axis=1)
y=MLData.isFraud


print("No of Splits")
print(cv.get_n_splits(X, y))

print("Fold Details:")
print(cv)


print("Size of the Dataframe")
len(X)


for train_index, test_index in cv.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    len(X_train)
    len(X_test)
    stkflod_RF(X_train,y_train,X_test,y_test)

