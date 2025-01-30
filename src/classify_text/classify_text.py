import os
import warnings

import datetime
import shutil

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from google.cloud import aiplatform
from official.nlp import optimization

warnings.filterwarnings("ignore")
os.environ[ "TF_CPP_MIN_LOG_LEVEL" ] = "2"

tf.get_logger().setLevel("ERROR")
print("Num Gpus Available: ", len(tf.config.list_physical_devices("GPU")))

def main():
    print("START")

