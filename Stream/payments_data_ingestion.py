from google.cloud import pubsub_v1 
from time import sleep
import random
import datetime,time
import json,re,os
import string
import data_creator

#pub/sub Config
publisher = pubsub_v1.PublisherClient()
project = "fraud-detection-data245"

#topic
payment_topic = "payments_dev"



#Pub/Sub topics for Payment data publishing
payment_topic_path = publisher.topic_path(project, payment_topic)



#Generate 100 Transactions (Payments)
for beacons in range(100):
    payment_transaction = data_creator.create_payment_transaction()
    #Sending Transactions to pub/sub
    print(payment_transaction)
    publish_future = publisher.publish(payment_topic_path, bytes(json.dumps(payment_transaction),"utf-8"))
    print(publish_future.result())