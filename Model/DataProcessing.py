import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import LabelEncoder
import logging
import sys
import time
import os
import json
from google.cloud import bigquery
sys.path.append("../conf")
from gcp_conf import *

client = bigquery.Client()

inputfile = 'gs://'+ml_data_bucket+'/'+ml_raw_data_folder_name+"/20211116/" +ml_raw_data_file_name
outputfolder = 'gs://'+ml_data_bucket+'/'+ml_processed_data_folder_name+"/"

class Logger():
    def logger_funct(x_): 
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        today = date.today()
        todaydate = today.strftime('%d%m%Y')

        handler = logging.FileHandler("logfile{}".format(todaydate))
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        logger.info(x_)

class Str_checker():
    def check_str(x_, y_):
        x_[y_] = x_[y_].astype('str')
        try:
            cs = "Check string function is being deployed"
            Logger.logger_funct(cs)
            for i in range(len(x_)):
                if (type(x_[y_][i]) == type("str")) == True:
                    cs_ = "Value strings"
                    #Logger.logger_funct(cs)
                    #return True
                else:
                    cs_ = "invalid strings"
                    Logger.logger_funct(cs)
                    #return False
        except ValueError as e:
            Logger.logger_funct(e)  
        except TypeError as e:
            Logger.logger_funct(e)  

class TransferType_check():
    def check_valid_type(x_):
        """
        Returns 
        """
        valid_type = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        try:
            if (x_.type.isin(valid_type)).all() == True:
                cvt_ = 'All Type values are valid'
                Logger.logger_funct(cvt_)
                return True
            else:
                invalid_ = x_[x_.type.isin(valid_type) == False].index.tolist()
                cvt_ = "Type has an invalid value. Returning indices with invalid values: {}".format(invalid_)
                Logger.logger_funct(cvt_)
                return x_[x_.type.isin(valid_type)]
                
        except ValueError as e:
            Logger.logger_funct(e)  
        except TypeError as e:
            Logger.logger_funct(e)  

class Account_checker():
    def account_check(x_, y_):
        valid_str = ['C', 'M']
        valid_tuple = tuple(valid_str)
        try:
            if (((x_[y_].str.startswith(valid_tuple)).all() == True)):
                ac_ = 'Account information for {} is properly named'.format(y_)
                Logger.logger_funct(ac_)
                return True
            else:
                invalid_ = x_[~x_[y_].str.startswith(valid_tuple)].index.tolist()
                ac_ = "{} is missing the correct information. Returning indices with invalid information : {}".format(y_, invalid_)
                Logger.logger_funct(ac_)
                return x_[x_[y_].str.startswith(valid_tuple)]
        except ValueError as e:
            Logger.logger_funct(e)  
        except TypeError as e:
            Logger.logger_funct(e)   

class Positive_checker():
    def check_positive(x_, y_):
        try:
            if((x_[y_] >= 0).all() == True):
                cp_ = 'All values for {} is nonnegative'.format(y_)
                Logger.logger_funct(cp_)
                return True 
            else:
                invalid_ = x_[(x_[y_] < 0) | (x_[y_].isnull())].index.tolist()
                cp_ = 'There are values for {} that are not postive. Returning indices with nonpositive values : {}'.format(y_, invalid_)
                Logger.logger_funct(cp_)
                #return x_[(x_[y_] < 0) | (x_[y_].isnull())]
                return x_[~(x_[y_] < 0) | (x_[y_].isnull())]
        except ValueError as e:
            Logger.logger_funct(e)  
        except TypeError as e:
            Logger.logger_funct(e)  
    
class Difference_name():
    def orig_dest_diff(x_, y_, z_):
        try:
            if(x_[y_] != x_[z_]).all() == True:
                odd_ = "There contains no account names in {} that are the same as {}".format(y_, z_)
                Logger.logger_funct(odd_)
                return True
            else:
                invalid_ = x_[~(x_[y_] != x_[z_])].index.tolist()
                odd_ = "There are name accounts that are the same. Returning indices with same bank account names : {}".format(invalid_)
                Logger.logger_funct(odd_)
                return x_[~(x_[y_] != x_[z_])]
        except ValueError as e:
            Logger.logger_funct(e)  
        except TypeError as e:
            Logger.logger_funct(e)  

class Null_checker():
    def check_null(x_):
        """
        Checks the dataframe to see whether there are any missing values, 
        then removes the rows with missing values
        """
        if(x_.isnull().values.any()) == False:
            null_ = "No nulls"
            Logger.logger_funct(null_)
            return x_
        else:
            #Data Quality Report - TBA
            null_ = "Dropping nulls"
            x_ = x_.dropna()
            Logger.logger_funct(null_)
            return x_

class EncodeData():
    def type_convert(x_):
        """
        Convert columns that contain "string" into numeric by using an encoder
        """
        
        valid_type = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        encodertype = LabelEncoder()
        encodertype.fit(valid_type)

        encoderfunc = LabelEncoder()

        x_['transaction_type'] = encodertype.transform(x_['type'])
        x_['nameorig_enc'] = encoderfunc.fit_transform(x_['nameOrig'])
        x_['namedest_enc'] = encoderfunc.fit_transform(x_['nameDest'])

        x_ = x_.drop(columns = ['type', 'nameOrig', 'nameDest', 'step', 'isFlaggedFraud'], axis =1)

        x_ = x_.rename(columns = {'oldbalanceOrg': 'oldbalanceOrig'})

        convert_ = "Applying the converter"

        Logger.logger_funct(convert_)
        return x_

class FeatureValidation():
    def balance_diff(x_):
        x_['balance_difference'] = round(x_['oldbalanceOrig'] - x_['newbalanceOrig'], 2).ne(x_['amount'])
        x_["balance_difference"] = x_["balance_difference"].astype(int)

        bd_= "Apply balance difference"

        Logger.logger_funct(bd_)
        return x_

class CSVHandler():
    def export_csv(x_):
        """
        Creates a datetime object for the current date (in which it was processed), 
        then converts the datetime object into a string. 
        Append the datetime string to the file name 
        and exports the PANDAs dataframe as a csv file.
        """

        #Take the date
        today = date.today()
        #Format the date
        todaydate = today.strftime('%d%m%Y')
        #Create file name
        #filename = 'data_{}.csv'.format(todaydate)
        folder_path  = outputfolder+ todaydate
        os.mkdir(todaydate)
        filename = "processed_payments.csv"


        #export file as CSV

        ecsv_ = "Exporting as CSV"
        Logger.logger_funct(ecsv_)


        #Local Folder with processed CSV
        x_.to_csv(todaydate+"/"+filename, index = False)

        processed_records = len(x_.index)

        #upload to GCS
        os.system('gsutil cp -r '+todaydate+ ' ' +outputfolder)


        data_processing_info = [{
            'timestamp' : str(round(time.time())),
            'numOfInputRecords' : input_data_records,
            'numOfProcessedRecords' : processed_records,
            'inputDataLink' : inputfile,
            'processedDataLink' : (outputfolder+ todaydate+"/"+filename),
            'data_prep_starttime' : str(data_prep_starttime)
        }]

        logging.info('big query insertion : {}'.format(str(json.dumps(data_processing_info))))
        client.insert_rows_json('ml_project.data_preparation', json.loads(str(json.dumps(data_processing_info))))
        
        

if __name__ == '__main__':
   
    
    df = pd.read_csv(inputfile)

    input_data_records = len(df.index)
    data_prep_starttime = round(time.time())

    def DataValidation(data_, amt_, obo, nbo, obd, nbd, no_, nd_):

        #Str_checker.check_str(df, 'nameOrig')
        
        valid_func = TransferType_check.check_valid_type(data_)
        account_func = Account_checker.account_check(data_, no_)
        amount_func = Positive_checker.check_positive(data_, amt_)
        obo_func = Positive_checker.check_positive(data_, obo)
        nbo_func = Positive_checker.check_positive(data_, nbo)
        obd_func = Positive_checker.check_positive(data_, obd)
        nbd_func = Positive_checker.check_positive(data_, nbd) 
        diff_func = Difference_name.orig_dest_diff(data_, no_, nd_)
        pos_list = [valid_func,account_func,amount_func, obo_func, nbo_func, obd_func, nbd_func,diff_func]

        if all(pos_list) == True:
            return True
        else:
            return False

    def funcs(x_):
        """
        Apply functions in a single instance
        """

        prep = DataValidation(x_, 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest','nameOrig', 'nameDest')

        if prep == True:
            x = Null_checker.check_null(x_)
            
            x = EncodeData.type_convert(x)

            x = FeatureValidation.balance_diff(x)

            CSVHandler.export_csv(x)

            return x
        else:
            issue_ = "Data preparation function found an error"
            Logger.logger_funct(issue_)
            
    funcs(df)