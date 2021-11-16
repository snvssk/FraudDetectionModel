from google.cloud import bigquery
import json


client = bigquery.Client()


def writeDataToBigQuery(tableName, jsonData):
	client.insert_rows_json(tableName, jsonData)


data = '[{ "timestamp": "1636588823", "modelName": "test", "foldNumber": 1, "testDataSize": 10, "trainDataSize": 5, "accuracy": 98.3 }]'

data_json = json.loads(data)
print(data_json)

writeDataToBigQuery("ml_project.ml_metrics", data_json)
