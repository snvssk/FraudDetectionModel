# importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix,accuracy_score,classification_report
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier

# Loading dataset
df = pd.read_csv('gs://245_project/PS_20174392719_1491204439457_log.csv')

# Feature extraction
data = df.copy() #full data
#data = df.sample(frac=0.1) # Sampled data

# Merchant flag for source and dist
data['OrigC']=data['nameOrig'].apply(lambda x: 1 if str(x).find('C')==0 else 0)
data['DestC']=data['nameDest'].apply(lambda x: 1 if str(x).find('C')==0 else 0)

# flag for transfer and cashout from type feature
data['TRANSFER']=data['type'].apply(lambda x: 1 if x=='TRANSFER' else 0)
data['CASH_OUT']=data['type'].apply(lambda x: 1 if x=='CASH_OUT' else 0)

# Calculating Amount error
data['OrigAmntErr']=(abs(data.oldbalanceOrg-data.newbalanceOrig)-data.amount)

# droping those feature which are id type category and those which used for feature extraction
droplist=['isFlaggedFraud','type','nameDest','nameOrig']
MLData=data.drop(labels=droplist,axis=1)

#saving to CSV
MLData.to_csv('TransformedCreditCardFraudData.csv')

#data input
inputfile = 'TransformedCreditCardFraudData.csv'

class Model:
    def __init__(self, datafile = inputfile, model_type = None):
        self.df = pd.read_csv(datafile)
        if model_type == 'lof':
            fraud = self.df[self.df["isFraud"] == 1] # Number of fraudulent transactions
            valid = self.df[self.df["isFraud"] == 0] # Number of valid transactions
            try : # to handle divide by zero error
                outlier_fraction = len(fraud)/float(len(valid))
            except ZeroDivisionError:
                outlier_fraction = 0
            self.user_defined_model = LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction)
        else:
            self.user_defined_model = RandomForestClassifier()
            
    def split(self, test_size):
        X = np.array(self.df.drop('isFraud',axis=1))
        y = np.array(self.df.isFraud)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size,random_state=42, shuffle=False)
    
    def fit(self):
        self.model = self.user_defined_model.fit(self.X_train, self.y_train)

    
    def model_result(self, model_type = None):
        if model_type == 'lof':
            y_pred=self.user_defined_model.fit_predict(self.X_test)
            y_pred[y_pred == 1] = 0 # Valid transactions are labelled as 0.
            y_pred[y_pred == -1] = 1 # Fraudulent transactions are labelled as 1.
            print('******** LOF ******')
            print('Confusion_matrix : ')
            print(confusion_matrix(self.y_test,y_pred))
            print("accuracy_score")
            print(accuracy_score(self.y_test,y_pred))
            print("errors")
            print((y_pred != self.y_test).sum()) # Total number of errors is calculated.
            print("classification_report")
            print(classification_report(self.y_test,y_pred))
            
            
        else:
            y_prob=self.user_defined_model.predict_proba(self.X_test)
            y_pred=self.user_defined_model.predict(self.X_test) 
            print('******** RF ******')
            print('AUPRC :', (average_precision_score(self.y_test, y_prob[:, 1])))
            print('F1_score :',(f1_score(self.y_test,y_pred)))
            print('Confusion_matrix : ')
            print(confusion_matrix(self.y_test,y_pred))
            print("accuracy_score")
            print(accuracy_score(self.y_test,y_pred))
            print("classification_report")
            print(classification_report(self.y_test,y_pred))

if __name__ == '__main__':
    model_instance1 = Model(model_type = 'rf')
    model_instance1.split(0.2)
    model_instance1.fit()   
    model_instance1.model_result()

    model_instance2 = Model(model_type = 'lof')
    model_instance2.split(0.2)
    model_instance2.fit()
    model_instance2.model_result(model_type = 'lof')
