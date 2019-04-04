from kafka import KafkaConsumer
from pymongo import MongoClient
from json import loads
import csv
import pandas as pd


consumer = KafkaConsumer(
    'guardian2',
     bootstrap_servers=['localhost:9092'],
     value_deserializer=lambda x: loads(x.decode('utf-8')))


i=0
with open('guardian.csv', mode='w') as guardian:
    csv_writer = csv.writer(guardian, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['text', 'category'])
    for message in consumer:

            p=message.value.split("||")

            csv_writer.writerow([p[1]+" "+p[2], p[0]])
            guardian.flush()
            print("Message received successfully. Message Number "+str(i))
            i=i+1
