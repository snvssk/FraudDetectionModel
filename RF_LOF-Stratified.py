# importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix,accuracy_score,classification_report
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import logging

FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'

logging.basicConfig(filename='RF_Model_log.log',
                            filemode='a',
                            format=FORMAT,
                            datefmt='%Y-%b-%d %X%z',
                            level=logging.INFO)
logging.info('Info message')

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
        self.model_type = model_type
        logging.info('Data loaded, filename : {0}, rows : {1}, columns : {2}'.format(datafile,self.df.shape[0],self.df.shape[1]))
        if self.model_type == 'lof':   
            logging.info('******** LOF ******')
            fraud = self.df[self.df["isFraud"] == 1] # Number of fraudulent transactions
            valid = self.df[self.df["isFraud"] == 0] # Number of valid transactions
            try : # to handle divide by zero error
                outlier_fraction = len(fraud)/float(len(valid))
            except ZeroDivisionError:
                logginf.error('Divide by Zero error at outlier fraction calculation')
                outlier_fraction = 0
            self.user_defined_model = LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction)
            logging.info('Model : {}'.format(self.user_defined_model))
        else:
            logging.info('******** RF ******')
            self.user_defined_model = RandomForestClassifier()
            logging.info('Model : {}'.format(self.user_defined_model))
            
    def split(self, test_size):
        X = np.array(self.df.drop('isFraud',axis=1))
        y = np.array(self.df.isFraud)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size,random_state=42, shuffle=False)
        logging.info('data split complete')
    
    def fit(self):
        self.model = self.user_defined_model.fit(self.X_train, self.y_train)

    
    def model_result(self):
        if self.model_type == 'lof':
            y_pred=self.user_defined_model.fit_predict(self.X_test)
            y_pred[y_pred == 1] = 0 # Valid transactions are labelled as 0.
            y_pred[y_pred == -1] = 1 # Fraudulent transactions are labelled as 1.
            logging.info('Confusion_matrix : ')
            logging.info('{}'.format(confusion_matrix(self.y_test,y_pred)))
            logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,y_pred)))
            logging.info('errors : {}'.format((y_pred != self.y_test).sum())) # Total number of errors is calculated.
            logging.info('classification_report : {}'.format(classification_report(self.y_test,y_pred)))
            
            
        else:
            y_prob=self.user_defined_model.predict_proba(self.X_test)
            y_pred=self.user_defined_model.predict(self.X_test) 
            logging.info('AUPRC : {}'.format(average_precision_score(self.y_test, y_prob[:, 1])))
            logging.info('F1_score : {}'.foramt(f1_score(self.y_test,y_pred)))
            logging.info('Confusion Matrix : {}'.format(confusion_matrix(self.y_test,y_pred)))
            logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,y_pred)))
            logging.info('classification_report : {}'.format(classification_report(self.y_test,y_pred)))

            
    def kfoldValidation(self):
        logging.info('*******KfoldValidation****** : {}'.format(self.user_defined_model))
        cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        X=self.df.drop('isFraud',axis=1)
        y=self.df.isFraud
        logging.info('number of splits : {}'.format(cv.get_n_splits(X, y)))
        i=1
        for train_index, test_index in cv.split(X, y):
            logging.info('Fold : {}'.format(i))
            logging.info('TRAIN : {0}, TEST : {1}'.format(train_index,test_index))
            #Pick Selected Training and Testing index(row numbers) and create Traing and Testing Folds
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            i+=1
            Model.stkflod_RF(self,X_train,y_train,X_test,y_test)
            
    def stkflod_RF(self,X_train,y_train,X_test,y_test):   
        Model.split(self,0.2)
        Model.fit(self)
        Model.model_result(self)

if __name__ == '__main__':
    model_instance1 = Model(model_type = 'rf')
    model_instance1.split(0.2)
    model_instance1.fit()   
    model_instance1.model_result()

    model_instance2 = Model(model_type = 'lof')
    model_instance2.split(0.2)
    model_instance2.fit()
    model_instance2.model_result(model_type = 'lof')
