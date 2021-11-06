import pandas as pd
import numpy as np
import os 
from datetime import date

from sklearn.preprocessing import LabelEncoder

import collections
import logging

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

class Valid_checker():
    def check_valid_type(x_):
        valid_type = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        try:
            if (x_.type.isin(valid_type)).all() == True:
                cvt_ = 'All Types are valid'
                Logger.logger_funct(cvt_)
                return True
            else:
                cvt_ = "Type has a misspelling"
                Logger.logger_funct(cvt_)
                return x_[~x_.type.isin(valid_type)]
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
                ac_ = "{} is missing the correct information".format({y_})
                Logger.logger_funct(ac_)
                return x_[~x_[y_].str.startswith(valid_tuple)]
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
                cp_ = 'There are values for {} that are not postive'.format(y_)
                Logger.logger_funct(cp_)
                return x_[(x_[y_] < 0) | (x_[y_].isnull())]
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
                odd_ = "There are name accounts that are the same"
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

class Convert_checker():
    def type_convert(x_):
        """
        Convert columns that contain "string" into numeric by using an encoder
        """

        #training already tested, keep reverse mapping - CHECK

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

class Balance_checker():
    def balance_diff(x_):
        x_['balance_difference'] = round(x_['oldbalanceOrig'] - x_['newbalanceOrig'], 2).ne(x_['amount'])
        x_["balance_difference"] = x_["balance_difference"].astype(int)

        bd_= "Apply balance difference"

        Logger.logger_funct(bd_)
        return x_

class Csv_check():
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
        filename = 'data_{}.csv'.format(todaydate)
        #export file as CSV

        ecsv_ = "Exporting as CSV"
        Logger.logger_funct(ecsv_)

        x_.to_csv(filename, index = False)

if __name__ == '__main__':

    os.chdir('/Users/michelleyuu/Desktop/data245/project/')
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

    def prep_funct(data_, amt_, obo, nbo, obd, nbd, no_, nd_):

        #Str_checker.check_str(df, 'nameOrig')
        
        valid_func = Valid_checker.check_valid_type(data_)
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

        prep = prep_funct(x_, 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest','nameOrig', 'nameDest')

        if prep == True:
            x = Null_checker.check_null(x_)
            
            x = Convert_checker.type_convert(x)

            x = Balance_checker.balance_diff(x)

            Csv_check.export_csv(x)

            return x
        else:
            issue_ = "Data preparation function found an error"
            Logger.logger_funct(issue_)
            
    funcs(df)