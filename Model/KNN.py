import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix,accuracy_score,classification_report
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import logging
from datetime import date
import faiss
from datetime import datetime
import pickle 
import time
import json
import sys
import os
from google.cloud import bigquery
sys.path.append("../conf")
from gcp_conf import *

client = bigquery.Client()

FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'

logging.basicConfig(filename='knn.log',
                            filemode='w+',
                            format=FORMAT,
                            datefmt='%Y-%b-%d %X%z',
                            level=logging.INFO)

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
        self.model_type = model_type
        self.df = pd.read_csv(datafile)
        logging.info('Data loaded, filename : {0}, rows : {1}, columns : {2}'.format(datafile,self.df.shape[0],self.df.shape[1]))
        self.user_defined_model = FaissKNeighbors(3)

    def fit(self):
        label_df = self.y_train.to_frame()
        fraud_training_indexes = label_df[label_df['isFraud'] == 1].index
        valid_training_indexes = label_df[label_df['isFraud'] == 0].index
        for i in range(10):
            random_fraud_index = np.random.choice(fraud_training_indexes, 50)
            random_valid_index = np.random.choice(valid_training_indexes, 5000)
            train_subset_index = np.concatenate((random_fraud_index.astype(int), random_valid_index.astype(int)), axis=0)
            self.model = self.user_defined_model.fit(np.ascontiguousarray(self.X_train.loc[train_subset_index.astype(int)]), np.ascontiguousarray(self.y_train.loc[train_subset_index.astype(int)]))

    def model_result(self):
        y_pred=self.user_defined_model.predict(np.ascontiguousarray(self.X_test))
        logging.info('F1_score : {}'.format(f1_score(self.y_test,y_pred)))
        logging.info('Confusion Matrix : {}'.format(confusion_matrix(self.y_test,y_pred)))
        logging.info('accuracy_score : {}'.format(accuracy_score(self.y_test,y_pred)))
        logging.info('classification_report : {}'.format(classification_report(self.y_test,y_pred)))
        tn, fp, fn, tp = confusion_matrix(self.y_test,y_pred).ravel()
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
    model_instance = Model(model_type = 'knn')
    model_instance.kfoldValidation()
    model_instance.packagingModel()