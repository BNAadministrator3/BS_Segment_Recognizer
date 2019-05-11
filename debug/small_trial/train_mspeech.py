#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import platform as plat
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# from Model0_spp import ModelSpeech
# from Model1_general import ModelSpeech
# from model_set.Model2_1smart import ModelSpeech
# from modification.SpeechModelClear import ModelSpeech
# from modification.AttentionModel import ModelSpeech
# from Model30_5folds import ModelSpeech
from Model44_trainSetMain import ModelSpeech
# from model_set.Model22_fl import ModelSpeech
# from Model41_bowel import ModelSpeech

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #only display error and warning; for 1: all info; for 3: only error.
# #进行配置，使用90%的GPU
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config))


datapath = ''
modelpath = 'model_set'


if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
	os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'E:\workspace\stdenv\individual_spp\dataset\segments'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	# datapath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/standard'
	datapath = '/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

ms = ModelSpeech(datapath)

#ms.LoadModel(modelpath + 'speech_model24_e_0_step_327500.model')
ms.TrainModel(datapath, epoch = 20, batch_size = 32, load_weights = False)

