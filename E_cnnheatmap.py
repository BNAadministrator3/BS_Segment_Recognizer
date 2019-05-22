from help_func.feature_transformation import read_wav_data,GetFrequencyFeatures
from help_func.utilities import focal_loss
import os
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import load_model
from keras import optimizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配
set_session(tf.Session(config=config))

#1. load image
three = []
results =[]
spath = os.path.join(os.getcwd(),'log&&materials','105660089020341_2018_11_13_08_20_07.wav')
three.append(spath)
spath = os.path.join(os.getcwd(),'log&&materials','20843728498872_2018_01_08_10_42_58.wav')
three.append(spath)
spath = os.path.join(os.getcwd(),'log&&materials','199481741554988_2018_10_20_00_04_25.wav')
three.append(spath)

# n1path = os.path.join(os.getcwd(),'log&&materials','noisy_sounds','199481741554988_2018_10_20_00_00_00.wav')
for spath in three:
    signal,fs = read_wav_data(spath)
    spec = GetFrequencyFeatures(signal,fs)
    img = spec.reshape(1,spec.shape[0], spec.shape[1], 1)
#2. load model
# mpath = os.path.join(os.getcwd(),'CNNevaluation','residual','resi=1','SPEC_Residual_8_epoch14.h5')
# mpath = os.path.join(os.getcwd(),'CNNdesign','spec+inception','10','SPEC_Inception_10_epoch4.h5')
    mpath = os.path.join(os.getcwd(),'CNNevaluation','regular','regi=0','spec_Regular_8_epoch12.h5')
    model = load_model(mpath,custom_objects={'FocalLoss': focal_loss,'focal_loss_fixed': focal_loss()})
    optimizer = optimizers.Adadelta()
    model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
#3. try to predictverbose=0
    print(np.argmax(model.predict(img)))
    print(model.predict(img,verbose=1))
# print(np.argmax(model.predict(img)[0]))
# print(model.layers[27])
# print(model.layers[28])
# print(model.layers[29])
    bowel_output = model.output[:,1]
# print(bowel_output)
    last_conv_layer = model.layers[28]
    grads = K.gradients(bowel_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(256):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
# heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    results.append(heatmap)

import pickle

files = os.path.join(os.getcwd(),'heatmaps.txt')
with open(files,'wb') as f:
    pickle.dump(results,f)

import cv2
# spec = np.rot90(spec)
nspec = np.uint8( 255 * ( spec - np.min(spec) ) / ( np.max(spec)-np.min(spec) ))
img = cv2.applyColorMap(nspec, cv2.COLORMAP_BONE)


# plt.axis("off")
# plt.imshow(nspec,cmap=plt.cm.gray)
# plt.tight_layout()
# plt.show()
# savepath = os.path.join(os.getcwd(), 'trial.jpg' )
# plt.savefig(savepath,bbox_inches='tight')


# img = cv2.imread(savepath)
heatmap = cv2.resize(heatmap, (spec.shape[1], spec.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imshow('whole',superimposed_img)
cv2.imwrite(os.path.join(os.getcwd(),'mask_cam.jpg'), superimposed_img)
