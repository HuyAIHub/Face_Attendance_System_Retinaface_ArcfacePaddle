import time, threading
from kafka import KafkaConsumer
from cFunctions import GlobFunc
from cConst import Const

var             = Const()
topic_ppe       = var.TOPIC_PPE
kafka_broker    = var.KAFKA_BROKER
consumer        = KafkaConsumer(topic_ppe,bootstrap_servers=kafka_broker,auto_offset_reset ='latest')

class ReadMessageConsumer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.name = "Thread -- ReadMessageConsumer"
    def run(self):
        while True:
            print("reading...")
            GlobFunc.readMessage()
            time.sleep(2)
