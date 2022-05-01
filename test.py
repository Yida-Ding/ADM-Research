import os
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.autograph.set_verbosity(0)
from NetworkGenerator import Scenario
from VNSSolver import VNSSolver


config = {"DATASET": "ACF5",
          "SCENARIO": "ACF5-SC1",
          "SEED": 0,
          "EPISODE": 800,
          "TRAJLEN": 5,
          "ALPHA": 0.001,
          "GAMMA": 0.9,
          "FC1DIMS": 256,
          "FC2DIMS": 256
          }
    

model = keras.models.load_model('Results/%s/Policy_%s'%(config["SCENARIO"],config["EPISODE"]))

print(model)





