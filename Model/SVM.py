# importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix,accuracy_score,classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import pickle 
import logging
from datetime import date
import time
import json
import sys
import os
from google.cloud import bigquery
sys.path.append("../conf")
from gcp_conf import *


#Take the date
today = date.today()
#Format the date
todaydate = today.strftime('%d%m%Y')

#processed data input
processedfile = 'gs://'+ml_data_bucket+'/'+ml_processed_data_folder_name+"/"+todaydate+"/" +ml_processed_data_file_name


#Packaged Model
model_storage = 'gs://'+ml_model_store_bucket_name+"/"


client = bigquery.Client()

FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'

logging.basicConfig(filename='svm.log',
                            filemode='w+',
                            format=FORMAT,
                            datefmt='%Y-%b-%d %X%z',
                            level=logging.INFO)

class Model:
    def __init__(self, datafile = processedfile, model_type = None):
        logging.info('Loading data from:' + datafile)
        self.df = pd.read_csv(datafile)
        self.model_type = model_type
        logging.info('Data loaded, filename : {0}, rows : {1}, columns : {2}'.format(datafile,self.df.shape[0],self.df.shape[1]))
        self.user_defined_model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, verbose=1))
    
    def fit(self):
        self.model = self.user_defined_model.fit(self.X_train, self.y_train)

    def model_result(self):
        y_pred=self.user_defined_model.predict(self.X_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test,y_pred).ravel()
        logging.info('F1_score : {}'.format(f1_score(self.y_test,y_pred)))
        logging.info('Confusion Matrix : {}'.format(confusion_matrix(self.y_test,y_pred)))
        logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,y_pred)))
        logging.info('classification_report : {}'.format(classification_report(self.y_test,y_pred)))
        data = [{
            'timestamp' : str(Model.current_milli_time()),
            'modelName' : self.model_type,
            'foldNumber' : self.foldNumber,
            'testDataSetSize': len(self.X_test),
            'trainDataSetSize' : len(self.X_train),
            'accuracy' : accuracy_score(self.y_test,y_pred),
            'confusionMatrix' : {
                'truePositive' : int(tp),
                'trueNegative' : int(tn),
                'falsePositive' : int(fp),
                'falseNegative' : int(fn)
            }
        }]

        logging.info('big query insertion : {}'.format(str(json.dumps(data))))
        Model.writeDataToBigQuery('ml_project.metric2', json.loads(str(json.dumps(data))))


    def current_milli_time():
        return round(time.time())
            
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
            Model.stkflod_RF(self)
            
    def stkflod_RF(self):   
        Model.fit(self)
        Model.model_result(self)

    def packagingModel(self):
        os.mkdir ('../ModelPackages/' + todaydate)
        filename = '../ModelPackages/' + todaydate + "/"+ datetime.now().strftime("%Y-%m-%d %H:%M") +'_'+ str(self.model_type) + '_model.pkl' 
        
        with open(filename, 'wb') as model_file:
            pickle.dump(self.user_defined_model, model_file)
        logging.info('Model Saved, file name : {}'.format(filename))
        #upload to GCS
        os.system('gsutil cp -r '+'../ModelPackages/' + todaydate+ ' ' +model_storage)

    def writeDataToBigQuery(tableName, jsonData):
        client.insert_rows_json(tableName, jsonData)

if __name__ == '__main__':
    model_instance = Model(model_type = 'svm')
    model_instance.kfoldValidation()
    model_instance.packagingModel()
