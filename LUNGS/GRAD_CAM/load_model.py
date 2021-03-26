import glob
import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam, Nadam
from tensorflow.keras.applications import MobileNetV2
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
#tf.random.set_seed(221)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

import sys
sys.path.insert(0, '../../')
from models import DRRMSAN_multiscale_attention_bayes_022

alpha_1 = 0.25
alpha_2 = 0.25
alpha_3 = 0.25
alpha_4 = 0.25
model = DRRMSAN_multiscale_attention_bayes_022(height=256, width=256, n_channels=3, alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_3 = alpha_3, alpha_4 = alpha_4)
from tensorflow.keras.utils import  plot_model as pm  #plotting the model structure
pm(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,dpi=60)

from tensorflow import keras
# model = load_model('modelW_drrmsan_lungs.h5')
model.load_weights('modelW_drrmsan_lungs.h5')

model.summary()
