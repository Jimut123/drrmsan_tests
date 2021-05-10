
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

from GRAD_CAM import get_img_array, make_gradcam_heatmap, get_jet_img

import sys
sys.path.insert(0, '../../')
from models import DRRMSAN_multiscale_attention_bayes_022_attn_3

alpha_1 = 0.25
alpha_2 = 0.25
alpha_3 = 0.25
alpha_4 = 0.25
model = DRRMSAN_multiscale_attention_bayes_022_attn_3(height=256, width=256, n_channels=3, alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_3 = alpha_3, alpha_4 = alpha_4)

model.summary()
# from tensorflow.keras.utils import  plot_model as pm  #plotting the model structure
# pm(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,dpi=60)

from tensorflow import keras
# model = load_model('modelW_drrmsan_lungs.h5')
model.load_weights('modelW_drrmsan_skinLesion.h5')

model.summary()

# img = cv2.imread('img_lungs.png',cv2.IMREAD_COLOR)
img = cv2.imread('X_img_2.bmp',cv2.IMREAD_COLOR)
# 1
plt.imsave('skin_lesion_img_attention.png', img[:,:,::-1])
img = cv2.resize(img,(256,256))

img_array = np.expand_dims(img, axis=0)
# plt.imshow(img)
# plt.show()
print(img_array.shape)
# Make model
# model = model_builder(weights="imagenet")

# Print what the top predicted class is
preds = model.predict(img_array)
yp = np.round(preds,0)
yp = yp[4]
# print("Predicted:", decode_predictions(preds, top=1)[0])
print("preds : ", preds)

                                      

#lclns = ['side_6','side_7','side_8','activation_92']#['activation_54','activation_73','activation_5','activation_111']
lclns = ['conv2d_40','conv2d_52','conv2d_62','conv2d_74','conv2d_8','conv2d_87']
last_conv_layer_name = "conv2d_91"

# activation_54 -- 
# activation_73 --
# activation_5 --
# activation_92  xx
# activation_111 --
# activation_10 xx


# Generate class activation heatmap
image_no = 1
for item in lclns:
    last_conv_layer_name = item
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = cv2.resize(heatmap,(256,256))
    
    # img, heatmap, heatmap_, superimposed_img = get_jet_img(img, heatmap)
    # a colormap and a normalization instance
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=heatmap.min(), vmax=heatmap.max())

    # map the normalized data to colors
    # image is now RGBA (512x512x4) 
    image = cmap(norm(heatmap))
    print("---------",heatmap.shape)
    img_save_name = "lungs_attention_{}.png".format(image_no)
    plt.matshow(image)
    plt.show()
    
    plt.imsave(img_save_name, image)
    image_no += 1
