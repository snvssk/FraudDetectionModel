# importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix,accuracy_score,classification_report
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import faiss
from datetime import datetime
from pathlib import Path
import pickle 
import logging
import warnings
import sys
from datetime import date
import gcsfs
from google.cloud import bigquery
import json
import time
import math
import os
sys.path.append("../conf")
from gcp_conf import *

warnings.filterwarnings("ignore")

table_id = "ml_project.metric" 

client = bigquery.Client()


FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'

logging.basicConfig(filename='FraudDetectionModel.log',
                            filemode='a',
                            format=FORMAT,
                            datefmt='%Y-%b-%d %X%z',
                            level=logging.INFO)
logging.info('Info message')

#Take the date
today = date.today()
#Format the date
todaydate = today.strftime('%d%m%Y')

#processed data input
processedfile = 'gs://'+ml_data_bucket+'/'+ml_processed_data_folder_name+"/"+todaydate+"/" +ml_processed_data_file_name


#Packaged Model
model_storage = 'gs://'+ml_model_store_bucket_name+"/"

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k
    #IndexFlatL2 is Euclidean distance
    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y
    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


class Model:
    def __init__(self, datafile = processedfile, model_type = None):
        self.df = pd.read_csv(datafile)
#       self.df = self.df.sample(frac=0.01) # Size of data frame is reduced
        self.model_type = model_type
        logging.info('Data loaded, filename : {0}, rows : {1}, columns : {2}'.format(datafile,self.df.shape[0],self.df.shape[1]))
        if self.model_type == 'lof':   
            logging.info('******** LOF ******')
            fraud = self.df[self.df["isFraud"] == 1] # Number of fraudulent transactions
            valid = self.df[self.df["isFraud"] == 0] # Number of valid transactions
            try : # to handle divide by zero error
                outlier_fraction = len(fraud)/float(len(valid))
            except ZeroDivisionError:
                logging.error('Divide by Zero error at outlier fraction calculation')
                outlier_fraction = 0
            self.user_defined_model = LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction,novelty=True)
            logging.info('Model : {}'.format(self.user_defined_model))
        elif self.model_type == 'rf':
            logging.info('******** RF ******')
            self.user_defined_model = RandomForestClassifier()
            logging.info('Model : {}'.format(self.user_defined_model))
        elif self.model_type == 'knn':
            logging.info('******** KNN ******')
            self.user_defined_model = FaissKNeighbors(3)
            logging.info('Model : {}'.format(self.user_defined_model))
        elif self.model_type == 'svm':
            logging.info('******* SVM *******')
            self.user_defined_model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, verbose=1))
            logging.info('Model : {}'.format(self.user_defined_model))
            
    def fit(self):
        logging.info('Fitting Training Data : {}') 
        if self.model_type == 'knn':
            label_df = self.y_train.to_frame()
            fraud_training_indexes = label_df[label_df['isFraud'] == 1].index
            valid_training_indexes = label_df[label_df['isFraud'] == 0].index
            for i in range(10):
                random_fraud_index = np.random.choice(fraud_training_indexes, 50)
                random_valid_index = np.random.choice(valid_training_indexes, 5000)
                train_subset_index = np.concatenate((random_fraud_index.astype(int), random_valid_index.astype(int)), axis=0)
                self.model = self.user_defined_model.fit(np.ascontiguousarray(self.X_train.loc[train_subset_index.astype(int)]), np.ascontiguousarray(self.y_train.loc[train_subset_index.astype(int)]))
        else:
            self.model = self.user_defined_model.fit(self.X_train, self.y_train)

    
    def model_result(self):
        if self.model_type == 'lof':
            self.y_pred=self.user_defined_model.decision_function(self.X_test)
            self.y_pred[self.y_pred >= 0] = 0 # Valid transactions are labelled as 0.
            self.y_pred[self.y_pred < 0] = 1 # Fraudulent transactions are labelled as 1.
            self.auprc = 0 #for uniformity as lof does not have auprc function
            logging.info('Confusion_matrix : ')
            logging.info('{}'.format(confusion_matrix(self.y_test,self.y_pred)))
            logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,self.y_pred)))
            logging.info('errors : {}'.format((self.y_pred != self.y_test).sum())) # Total number of errors is calculated.
            logging.info('classification_report : {}'.format(classification_report(self.y_test,self.y_pred)))
            
            
        elif self.model_type == 'rf':
            self.y_prob=self.user_defined_model.predict_proba(self.X_test)
            self.y_pred=self.user_defined_model.predict(self.X_test)
            self.auprc = average_precision_score(self.y_test,self.y_prob[:, 1])
            logging.info('AUPRC : {}'.format(average_precision_score(self.y_test, self.y_prob[:, 1])))
            logging.info('F1_score : {}'.format(f1_score(self.y_test,self.y_pred)))
            logging.info('Confusion Matrix : {}'.format(confusion_matrix(self.y_test,self.y_pred)))
            logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,self.y_pred)))
            logging.info('classification_report : {}'.format(classification_report(self.y_test,self.y_pred)))

        elif self.model_type == 'knn':
            self.y_pred=self.user_defined_model.predict(np.ascontiguousarray(self.X_test))
            self.auprc = -1.0
            logging.info('F1_score : {}'.format(f1_score(self.y_test,self.y_pred)))
            logging.info('Confusion Matrix : {}'.format(confusion_matrix(self.y_test,self.y_pred)))
            logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,self.y_pred)))
            logging.info('classification_report : {}'.format(classification_report(self.y_test,self.y_pred)))
            
        elif self.model_type == 'svm':
            self.y_pred=self.user_defined_model.predict(self.X_test)
            self.auprc = -1.0
            logging.info('F1_score : {}'.format(f1_score(self.y_test,self.y_pred)))
            logging.info('Confusion Matrix : {}'.format(confusion_matrix(self.y_test,self.y_pred)))
            logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,self.y_pred)))
            logging.info('classification_report : {}'.format(classification_report(self.y_test,self.y_pred)))
            
    def kfoldValidation(self):
        logging.info('*******KfoldValidation****** : {}'.format(self.user_defined_model))
        cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        X=self.df.drop('isFraud',axis=1)
        y=self.df.isFraud
        logging.info('number of splits : {}'.format(cv.get_n_splits(X, y)))
        i=1
        for train_index, test_index in cv.split(X, y):
            self.foldNumber = i 
            logging.info('Fold : {}'.format(i))
            logging.info('TRAIN : {0}, TEST : {1}'.format(train_index,test_index))
            #Pick Selected Training and Testing index(row numbers) and create Traing and Testing Folds
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]
            i+=1
            self.training_starting_time = round(time.time())
            Model.stkflod_RF(self)
            
    def stkflod_RF(self):
        logging.info('Entered into Stratified Kfold Fitting process : {}')  
        Model.fit(self)

        Model.model_result(self)
        tn, fp, fn, tp = confusion_matrix(self.y_test,self.y_pred).ravel()
        data = [{
                    'timestamp' : str(Model.current_milli_time()),
                    'modelName' : self.model_type,
                    'foldNumber' : self.foldNumber,
                    'fold_exec_starttime' : str(self.training_starting_time),
                    'testDataSetSize': len(self.X_test),
                    'trainDataSetSize' : len(self.X_train),
                    'accuracy' : accuracy_score(self.y_test,self.y_pred),
                    'confusionMatrix' : {
                        'truePositive' : int(tp),
                        'trueNegative' : int(tn),
                        'falsePositive' : int(fp),
                        'falseNegative' : int(fn)
                    },
                    'auprc' : self.auprc
             }]

        logging.info('big query insertion : {}'.format(str(json.dumps(data))))
        Model.writeDataToBigQuery(table_id, json.loads(str(json.dumps(data))))
        Model.packagingModel(self)
        
    def current_milli_time():
        return round(time.time())


    def writeDataToBigQuery(tableName, jsonData):
        client.insert_rows_json(tableName, jsonData)
         
    def packagingModel(self):
        Path('../ModelPackages/' + todaydate).mkdir(parents=True, exist_ok=True)
        filename = '../ModelPackages/' + todaydate + "/"+ datetime.now().strftime("%Y-%m-%d_%H:%M") +'_'+ str(self.model_type) +'_fold_'+ str(self.foldNumber) +'_model.pkl'
        
        with open(filename, 'wb') as model_file:
            pickle.dump(self.user_defined_model, model_file)
        logging.info('Model Saved, file name : {}'.format(filename))
        #upload to GCS
        os.system('gsutil cp -r '+'../ModelPackages/' + todaydate+ ' ' +model_storage)

if __name__ == '__main__':
    model_instance1 = Model(model_type = 'lof')        
    model_instance1.kfoldValidation()
    

