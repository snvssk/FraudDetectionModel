import json
import pickle
import pandas as pd
from datetime import datetime
from datetime import date
import numpy as np

def getSelectedModels():
    df = pd.read_json('MetricOutput.json')
    df = pd.concat([df, df["confusionMatrix"].apply(pd.Series)], axis=1)
    df.drop(columns=['confusionMatrix'],inplace=True)

    date = df['timestamp'].max().replace(hour = 0, minute = 0, second = 0)

    df = df.loc[(df['timestamp'] >= date)]

    #computing F1 for each row
    df['F1'] = df.apply(lambda row:(int(row['truePositive']) / 
                                    (int(row['truePositive']) + 
                                     (1/2 * (int(row['falseNegative'] + row['falsePositive']))))) ,  axis=1)

    grouped_df = df.groupby("modelName")
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()

    min_recall = mean_df['F1'].min()
    mean_df = mean_df.loc[mean_df['F1'] > min_recall]

    return mean_df['modelName']

getSelectedModels()