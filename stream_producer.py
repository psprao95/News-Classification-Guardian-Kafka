# -*- coding: utf-8 -*-

from kafka import KafkaProducer
from time import sleep
import json, sys
from json import dumps
import requests
import time

def getData(url):
    jsonData = requests.get(url).json()
    data = []
    labels = {}
    index = 0
    print("Retrieved "+str(len(jsonData["response"]["results"]))+"articles")
    for i in range(len(jsonData["response"]["results"])):
        headline = jsonData["response"]['results'][i]['fields']['headline']
        bodyText = jsonData["response"]['results'][i]['fields']['bodyText']

        label = jsonData["response"]['results'][i]['sectionName']
        if label not in labels:
            labels[label] = index
            index += 1
        #data.append({'label':labels[label],'Descript':headline})
        toAdd=str(label)+'||'+headline+"||"+bodyText
        data.append(toAdd)
    return(data)

def publish_message(producer_instance, value):
    try:
        producer_instance.send('guardian2',  value=value)
        producer_instance.flush()

        print('Message published successfully.')
    except Exception as ex:
        print('Exception in publishing message')
        print(str(ex))

def connect_kafka_producer():
    producer = None
    try:
        #producer = KafkaProducer(value_serializer=lambda v:json.dumps(v).encode('utf-8'),bootstrap_servers=['localhost:9092'], api_version=(0, 10),linger_ms=10)
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                                 value_serializer=lambda x:
                                 dumps(x).encode('utf-8'))


    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return producer

if __name__== "__main__":

    if len(sys.argv) != 4:
        print ('Number of arguments is not correct')
        exit()

    key = sys.argv[1]
    fromDate = sys.argv[2]
    toDate = sys.argv[3]

    url = 'http://content.guardianapis.com/search?from-date='+ fromDate +'&to-date='+ toDate +'&order-by=newest&show-fields=all&page-size=200&%20num_per_section=10000&api-key='+key
    all_news=getData(url)
    if len(all_news)>0:
        prod=connect_kafka_producer();
        for story in all_news:
            #print(json.dumps(story))
            publish_message(prod,  story)
            time.sleep(1)
        if prod is not None:
                prod.close()
