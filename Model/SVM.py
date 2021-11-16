# importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix,accuracy_score,classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import logging

FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'

logging.basicConfig(filename='svm.log',
                            filemode='w+',
                            format=FORMAT,
                            datefmt='%Y-%b-%d %X%z',
                            level=logging.DEBUG)

logging.info('Loading data from temporary location')
#data input
inputfile = 'data_01112021.csv'

class Model:
    def __init__(self, datafile = inputfile, model_type = None):
        self.df = pd.read_csv(datafile)
        self.model_type = model_type
        logging.info('Data loaded, filename : {0}, rows : {1}, columns : {2}'.format(datafile,self.df.shape[0],self.df.shape[1]))
        self.user_defined_model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, verbose=1))
    
    def fit(self):
        self.model = self.user_defined_model.fit(self.X_train, self.y_train)

    def model_result(self):
        y_pred=self.user_defined_model.predict(self.X_test) 
        logging.info('F1_score : {}'.format(f1_score(self.y_test,y_pred)))
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
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]
            i+=1
            Model.stkflod_RF(self)
            
    def stkflod_RF(self):   
        Model.fit(self)
        Model.model_result(self)

if __name__ == '__main__':
    model_instance = Model(model_type = 'svm')
    model_instance.kfoldValidation()
