#auxiliary package
import platform as plat
from tqdm import tqdm
import random
import os
import sys
import time

#algorithm package
import tensorflow as tf
import keras.backend as k
from keras.layers import *
from keras import optimizers
from keras.regularizers import l2
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential

#self-made package
from readdata_bowel import Testing
from readdata_bowel import DataSpeech
from readdata_bowel import FEATURE_TYPE, AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CLASS_NUM, PRIOR_ART
sys.path.append("../../help_func")
from evaluation import Compare2,plot_confusion_matrix
from utilities import focal_loss, ResiBlock, XcepBlock, ReguBlock


class ModelSpeech():  # 模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出四类：normal,wheezes,crackles,both
        '''
        self.datapath = datapath
        self.slash = ''
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜杠
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜杠
        else:
            print('*[Message] Unknown System\n')
            self.slash = '/'  # 正斜杠
        if (self.slash != self.datapath[-1]):  # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

        if PRIOR_ART == True:
            print('It is now concerning the related work!')
        print('This time the feature type is %s.'%FEATURE_TYPE.upper())
        self.modelname = ''
        self.model = self.CreateSimplifiedIntensifiedClassicModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1),classes=CLASS_NUM)
        # self.model = self.CreateCustomizedResNetModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1),classes=CLASS_NUM)
        # self.model = self.CreateCustomizedXceptionModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1),classes=CLASS_NUM)

        # self.model = self.CreateLSTMModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)
        # self.model = self.CreateLRModel(input_dim=672, classes=CLASS_NUM)

        # self.model = self.CreateResNetModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1),classes=CLASS_NUM)
        # self.model = self.CreateCustomizedInceptionModel2(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1),classes=CLASS_NUM)
        # self.model = self.CreateCustomizedInceptionModel3(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1),classes=CLASS_NUM)
        # self.model = self.CreateCustomizedInceptionModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)
        # self.model = self.CreateSimplifiedClassicModel2(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)

        # self.model = self.CreateClassicModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)
        self.metrics = {'type': 'eval', 'sensitivity': 0, 'specificity': 0, 'score': 0, 'accuracy': 0}

        if len(self.modelname) != 0:
            self.baseSavPath = []
            if AUDIO_FEATURE_LENGTH == 26:
                strname = 'mfcc_'+self.modelname
                self.baseSavPath.append(os.path.join(os.getcwd(), 'network&&weights', 'mfcc', self.modelname,strname))
                strname = 'mfcc_' + self.modelname + '_weights'
                self.baseSavPath.append(os.path.join(os.getcwd(), 'network&&weights', 'mfcc', self.modelname, strname))
            else:
                strname = 'spec_' + self.modelname
                self.baseSavPath.append(os.path.join(os.getcwd(), 'network&&weights', 'spectrogram', self.modelname,strname))
                strname = 'spec_' + self.modelname + '_weights'
                self.baseSavPath.append(os.path.join(os.getcwd(), 'network&&weights', 'spectrogram', self.modelname, strname))
            # print(self.savpath)
        else:
            assert(0)

    def CreateCustomizedResNetModel(self, input_shape, classes):
        X_input = Input(name='the_input', shape=input_shape)
        level_h1 = block(32)(X_input)
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = block(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = block(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层

        flayer = GlobalAveragePooling2D()(level_m3)
        fc = Dense(classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc)
        # self.interlayer = Model(inputs=X_input, outputs=flayer)
        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        # model.compile(optimizer=optimizer, loss='binary_crossentropy')# [focal_loss])
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Customized resnet model estabished.')
        self.modelname = 'residual'
        return model

    ###############################New inception module design##################################
    ############################################################################################
    def CreateCustomizedXceptionModel(self, input_shape, classes):
        X_input = Input(name='the_input', shape=input_shape)
        level_h1 = XcepBlock(32)(X_input)
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = XcepBlock(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = XcepBlock(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层

        flayer = GlobalAveragePooling2D()(level_m3)
        fc = Dense(classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc)
        # self.interlayer = Model(inputs=X_input, outputs=flayer)
        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        # model.compile(optimizer=optimizer, loss='binary_crossentropy')# [focal_loss])
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Customized Xception model estabished.')
        self.modelname = 'inception'
        return model

    def CreatePureXceptionModel(self,input_shape, classes):

        X_input = Input(name='the_input', shape=input_shape)

        # temp = BatchNormalization()(X_input)
        layer_h1 = SeparableConv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(X_input)  # 卷积层
        layer_h2 = SeparableConv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h1)  # 卷积层
        layer_p2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层

        layer_h3 = SeparableConv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p2)  # 卷积层
        layer_h4 = SeparableConv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h3)  # 卷积层
        layer_p4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h4)  # 池化层

        layer_h5 = SeparableConv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p4)  # 卷积层
        layer_h6 = SeparableConv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h5)  # 卷积层
        layer_p6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h6)  # 池化层

        # layer_h7 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_p6)  # 卷积层
        # layer_h8 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_h7)  # 卷积层
        # layer_p8 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        flayer = Flatten()(layer_p6)
        fc2 = Dense(classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Pure Xception cnn model estabished.')
        self.modelname = 'inception'

        self.intelayer = Model(inputs=X_input, outputs=fc2)

        return model

    ##################################cnn module design#########################################
    ############################################################################################

    def CreateSimplifiedIntensifiedClassicModel(self,input_shape, classes):

        X_input = Input(name='the_input', shape=input_shape)

        # temp = BatchNormalization()(X_input)
        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(X_input)  # 卷积层
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h1)  # 卷积层
        layer_p2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层

        layer_h3 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p2)  # 卷积层
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h3)  # 卷积层
        layer_p4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h4)  # 池化层

        layer_h5 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p4)  # 卷积层
        layer_h6 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h5)  # 卷积层
        layer_p6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h6)  # 池化层

        # layer_h7 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_p6)  # 卷积层
        # layer_h8 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=None)(layer_h7)  # 卷积层
        # layer_p8 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        flayer = Flatten()(layer_p6)
        fc2 = Dense(classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy')# [focal_loss])
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Simplified and intensified cnn model estabished.')
        self.modelname = 'cnn+dnn'

        self.intelayer = Model(inputs=X_input, outputs=fc2)

        return model

    def CreateSimplifiedClassicModel2(self,input_shape, classes):

        X_input = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(X_input)  # 卷积层
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h3)  # 卷积层
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h6)  # 卷积层
        layer_h8 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h7)  # 池化层

        layer_h9 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h8)  # 卷积层
        layer_h10 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h9)  # 池化层

        flayer = Flatten()(layer_h10)
        # fc1 = Dense(units=32, activation = "relu", use_bias = True, kernel_initializer = 'he_normal')(flayer)
        fc2 = Dense(classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        # model.compile(optimizer=optimizer, loss='binary_crossentropy')# [focal_loss])
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Further simplified cnn model estabished.')
        self.modelname = 'cnn+dnn'
        return model

    def CreateSimplifiedClassicModel(self,input_shape, classes):

        X_input = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(X_input)  # 卷积层
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h3)  # 卷积层
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h6)  # 卷积层
        layer_h8 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h7)  # 池化层

        layer_h9 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(layer_h8)  # 卷积层
        layer_h10 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h9)  # 池化层

        flayer = Flatten()(layer_h10)
        fc1 = Dense(units=32, activation = "relu", use_bias = True, kernel_initializer = 'he_normal')(flayer)
        fc2 = Dense(classes, use_bias=True, kernel_initializer='he_normal')(fc1)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        # model.compile(optimizer=optimizer, loss='binary_crossentropy')# [focal_loss])
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Simplified cnn model estabished.')
        return model

    def CreateClassicModel(self,input_shape, classes):

        X_input = Input(name='the_input', shape=input_shape)

        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(X_input)  # 卷积层
        # layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层

        # layer_h3 = Dropout(0.1)(layer_h3)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)  # 卷积层
        # layer_h4 = Dropout(0.2)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        # layer_h6 = Dropout(0.2)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h6)  # 卷积层
        # layer_h7 = Dropout(0.2)(layer_h7)
        ## layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h7)  # 池化层

        # layer_h9 = Dropout(0.3)(layer_h9)
        ## layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h9)  # 卷积层
        # layer_h10 = Dropout(0.4)(layer_h10)
        ## layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h10)  # 卷积层

        ## flayer = Flatten()(layer_h11)
        flayer = Flatten()(layer_h9)
        # flayer = Dropout(0.4)(flayer)
        fc1 = Dense(units=32, activation = "relu", use_bias = True, kernel_initializer = 'he_normal')(flayer)
        # fc1 = Dropout(0.5)(fc1)
        fc2 = Dense(classes, use_bias=True, kernel_initializer='he_normal')(fc1)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Classic cnn neural network established. ')
        self.modelname = 'cnn+dnn'
        return model
    ################################inception module design#####################################
    ############################################################################################

    def CreateCustomizedInceptionModel3(self, input_shape, classes): #0.42557M,perfect
        X_input = Input(name='the_input', shape=input_shape)
        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(X_input)  # 卷积层
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0001))(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=3, strides=None, padding="valid")(layer_h2)  # 池化层

        inception_3a = inception_Xseparable(input=layer_h3, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)
        inception_3b = inception_Xseparable(input=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)  # 224*25*480

        layer_h5 = MaxPooling2D(pool_size=3, strides=None, padding="valid")(inception_3b)  # 池化层
        transmute = core.Flatten()(layer_h5)
        linear = Dense(units=classes, activation='softmax', kernel_regularizer=l2(0.01))(transmute)
        last = linear

        model = Model(inputs=X_input, outputs=last)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('Customized inception neural network established. ')
        self.modelname='inception'
        return model

    def CreateCustomizedInceptionModel(self, input_shape, classes):
        X_input = Input(name='the_input', shape=input_shape)
        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(X_input)  # 卷积层
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=3, strides=None, padding="valid")(layer_h2)  # 池化层

        # inception_3a = inception_model(input=layer_h3, filters_1x1=32, filters_3x3_reduce=48, filters_3x3=64,filters_5x5_reduce=8, filters_5x5=16, filters_pool_proj=16)
        # inception_3b = inception_model(input=inception_3a, filters_1x1=85, filters_3x3_reduce=85, filters_3x3=128,filters_5x5_reduce=21, filters_5x5=64, filters_pool_proj=42)
        inception_3a = inception_model(input=layer_h3, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)
        inception_3b = inception_model(input=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)
        layer_h3 = MaxPooling2D(pool_size=3, strides=None, padding="valid")(inception_3b)  # 池化层

        transmute = core.Flatten()(layer_h3)
        linear = Dense(units=classes, activation='softmax', kernel_regularizer=l2(0.01))(transmute)
        last = linear

        model = Model(inputs=X_input, outputs=last)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        # model.compile(optimizer=optimizer,loss='binary_crossentropy')
        print('Customized inception neural network established. ')
        self.modelname = 'inception'
        return model

    #####################################else module design#####################################
    ############################################################################################

    def CreateLSTMModel(self, input_shape, classes):
        X_input = Input(name='the_input', shape=input_shape)
        y = Reshape((123, AUDIO_FEATURE_LENGTH), name='squeeze')(X_input)
        y = LSTM(256, return_sequences=False)(y)  # computation complexity
        y = Dense(classes, activation='softmax')(y)

        model = Model(inputs=X_input, outputs=y)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.modelname = 'lstm'
        print('The comparative lstm model is established.')
        return model

    def CreateLRModel(self, input_dim, classes):
        model = Sequential()
        # model.add(Dense(512, input_dim=input_dim, activation='sigmoid'))
        # model.add(Dense(256, input_dim=input_dim, activation='sigmoid'))
        # model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(classes, activation='softmax',input_dim=input_dim))

        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.modelname = 'mlp'
        print('The comparative logistic regression model is established.')
        return model

    def TrainModel(self, datapath, epoch=2, batch_size=32, load_weights=False, filename='model_set/speech_model25'):
        assert (batch_size % CLASS_NUM == 0)
        data = DataSpeech(datapath, 'train')
        num_data = sum(data.DataNum)  # 获取数据的数量

        os.system('pkill tensorboard')
        os.system('rm -rf ./checkpoints/files_summary/* ')
        train_writter = tf.summary.FileWriter(os.path.join(os.getcwd(), 'checkpoints', 'files_summary'))
        os.system('tensorboard --logdir=/home/zhaok14/example/PycharmProjects/setsail/individual_spp/checkpoints/files_summary/ &')
        print('\n')
        print(90 * '*')
        print(90 * '*')

        iterations_per_epoch = min(data.DataNum) // (batch_size // CLASS_NUM) + 1
        # iterations_per_epoch = 0
        print('trainer info:')
        print('training data size: %d' % num_data)
        print('increased epoches: ', epoch)
        print('minibatch size: %d' % batch_size)
        print('iterations per epoch: %d' % iterations_per_epoch)

        with k.get_session() as sess:
            train_writter.add_graph(sess.graph)
            if load_weights == True:
                try:
                    # modelpath = os.path.join(os.getcwd(), 'network&&weights', 'spectrogram', 'inception','spec_inception.h5')
                    self.model = load_model(modelpath,custom_objects={'focal_loss': focal_loss,'focal_loss_fixed': focal_loss()})
                    print('Successfully loading the model.')
                except:
                    print('Loading weights failed. Train from scratch.')
            sess.run(tf.global_variables_initializer())
            best_score = 0
            # epoch = 0
            duration = 0
            for i in range(0, epoch):
                iteration = 0
                yielddatas = data.data_genetator(batch_size,epoch)
                pbar = tqdm(yielddatas)
                for input, labels in pbar:
                    stime = time.time()
                    loss = self.model.train_on_batch(input[0],labels)
                    temp = self.model.predict_on_batch(input[0])
                    dtime = time.time() - stime
                    duration = duration + dtime
                    # okay = self.model.predict_on_batch(input[0])
                    # compare = self.interlayer.predict_on_batch(input[0])
                    train_summary = tf.Summary()
                    train_summary.value.add(tag='loss', simple_value=loss)
                    train_writter.add_summary(train_summary, iteration + i * iterations_per_epoch)
                    pr = 'epoch:%d/%d,iteration: %d/%d ,loss: %s' % (epoch, i, iterations_per_epoch, iteration, loss)
                    pbar.set_description(pr)
                    if iteration == iterations_per_epoch:
                        break
                    else:
                        iteration += 1
                pbar.close()
                if i % 1 == 0:
                    tmetrics = self.TestModel(sess=sess, datapath=datapath, str_dataset='train', data_count=1000, out_report=False, writer=train_writter, step=i)
                    metrics = self.TestModel(sess=sess, datapath=datapath, str_dataset='eval', data_count=-1, out_report=False, writer=train_writter, step=i)
                    if PRIOR_ART == True:
                        condition = tmetrics['score'] > metrics['score'] and i > 0 and tmetrics['sensitivity'] >= 90 and tmetrics['specificity'] >= 90
                    else:
                        condition = tmetrics['score'] > metrics['score'] and i > 0
                    if condition:
                        if metrics['score'] >= best_score:
                            self.metrics = metrics
                            self.metrics['epoch'] = i
                            best_score = metrics['score']
                            self.savpath = []
                            self.savpath.append((self.baseSavPath[0] + '_epoch' + str(i) + '.h5'))
                            self.savpath.append((self.baseSavPath[1] + '_epoch' + str(i) + '.h5'))
                            self.model.save(self.savpath[0])
                            self.model.save_weights(self.savpath[1])
            if 'epoch' in self.metrics.keys():
                print('The best metric after restriction took place in the epoch: ', self.metrics['epoch'])
                print('Sensitivity: {}; Specificity: {}; Score: {}; Accuracy: {}'.format(self.metrics['sensitivity'],self.metrics['specificity'],self.metrics['score'],self.metrics['accuracy']))
                self.TestGenerability(weightspath=self.savpath[1])
            else:
                print('The restricted best metric is not found. Done!')
                # path_test = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/network&&weights/spectrogram/mlp/spec_mlp_weights_epoch12.h5'
                # self.TestGenerability(weightspath=path_test)
            print('Training duration: {}s'.format(round(duration,2)))

    def TestModel(self, sess, writer, datapath='', str_dataset='eval', data_count=32, out_report=False, show_ratio=True, step=0):
        '''
        测试检验模型效果
        '''
        data = DataSpeech(datapath, str_dataset)
        # data.LoadDataList(str_dataset)
        num_data = sum(data.DataNum)  # 获取数据的数量
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data

        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            accuracy = 0
            sensitivity = 0
            specificity = 0
            score = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            txt_obj = []
            if (out_report == True):
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')  # 打开文件并读入

            start = time.time()
            cm_pre = []
            cm_lab = []
            map = {0: 'normal', 1: 'bowel sounds'}
            # data_count = 200
            for i in tqdm(range(data_count)):
                data_input, data_labels = data.GetData((ran_num + i) % num_data, mode='non-repetitive')  # 从随机数开始连续向后取一定数量数据

                predictions = []
                if PRIOR_ART == False:
                    if len(data_input) <= AUDIO_LENGTH:
                        data_in = np.zeros((1, AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
                        data_in[0, 0:len(data_input)] = data_input
                        data_pre = self.model.predict_on_batch(data_in)
                        predictions = np.argmax(data_pre[0], axis=0)
                    else:
                        assert(0)
                else:
                    data_pre = self.model.predict_on_batch(np.expand_dims(data_input,axis=0))
                    predictions = np.argmax(data_pre[0], axis=0)
                # print('predictions:',predictions)
                # print('data_pre:',np.argmax(data_pre[0], axis=0))
                # print ('data_label:',data_labels[0])

                cm_pre.append(map[predictions])
                cm_lab.append(map[data_labels[0]])

                tp, fp, tn, fn = Compare2(predictions, data_labels[0])  # 计算metrics
                overall_p += tp + fn
                overall_n += tn + fp
                overall_tp += tp
                overall_tn += tn

                txt = ''
                if (out_report == True):
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(data_pre) + '\n'
                    txt += '\n'
                    txt_obj.write(txt)

            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity, 2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity, 2)
            else:
                specificity = 'None'
            if sensitivity != 'None' and specificity != 'None':
                score = (sensitivity + specificity) / 2
                score = round(score, 2)
            else:
                score = 'None'
            accuracy = (overall_tp + overall_tn) / (overall_p + overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end - start, 2)
            # print('*[测试结果] 片段识别 ' + str_dataset + ' 敏感度：', sensitivity, '%, 特异度： ', specificity, '%, 得分： ', score, ', 准确度： ', accuracy, '%, 用时: ', dtime, 's.')
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(str_dataset,
                                                                                                sensitivity,
                                                                                                specificity, score,
                                                                                                accuracy, dtime)
            tqdm.write(strg)

            assert (len(cm_lab) == len(cm_pre))
            img_cm = plot_confusion_matrix(cm_lab, cm_pre, list(map.values()),tensor_name='MyFigure/cm', normalize=False)
            writer.add_summary(img_cm, global_step=step)
            summary = tf.Summary()
            summary.value.add(tag=str_dataset + '/sensitivity', simple_value=sensitivity)
            summary.value.add(tag=str_dataset + '/specificity', simple_value=specificity)
            summary.value.add(tag=str_dataset + '/score', simple_value=score)
            summary.value.add(tag=str_dataset + '/accuracy', simple_value=accuracy)
            writer.add_summary(summary, global_step=step)

            if (out_report == True):
                txt = '*[测试结果] 片段识别 ' + str_dataset + ' 敏感度：' + sensitivity + '%, 特异度： ' + specificity + '%, 得分： ' + score + ', 准确度： ' + accuracy + '%, 用时: ' + dtime + 's.'
                txt_obj.write(txt)
                txt_obj.close()

            metrics = {'data_set': str_dataset, 'sensitivity': sensitivity, 'specificity': specificity, 'score': score,
                       'accuracy': accuracy}
            return metrics

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

    def TestGenerability(self, weightspath, datasourceA=None, datasourceB=None):
        training = ['/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed/train/','train']
        validation = ['/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed/val/','validation']

        test_same = ['/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed/test/','same']
        # test_different = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-0419-different/','different']


        print(90 * '*')
        print('Firstly self-check the generability testing method:')
        self.__dataTesting__(training, validation, weightspath)
        print('')
        print('Then derive the generability testing results: ')
        self.__dataTesting__(test_same,  validation, weightspath)

    def __dataTesting__(self, dataSourceA, dataSourceB, weightspath):
        data = Testing(dataSourceA, dataSourceB)
        self.model.load_weights(weightspath)
        for choice in (dataSourceA[1], dataSourceB[1]):
            num_data = data.DataNum[choice]  # 获取数据的数量
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0

            start = time.time()
            # data_count = 200
            pbar = tqdm(range(num_data))
            for i in pbar:
                data_input, data_labels = data.GetData((ran_num + i) % num_data, dataType=choice)  # 从随机数开始连续向后取一定数量数据
                if PRIOR_ART == False:
                    if len(data_input) <= AUDIO_LENGTH:
                        data_in = np.zeros((1, AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
                        data_in[0, 0:len(data_input)] = data_input
                        data_pre = self.model.predict_on_batch(data_in)
                        predictions = np.argmax(data_pre[0], axis=0)
                    else:
                        assert (0)
                else:
                    data_pre = self.model.predict_on_batch(np.expand_dims(data_input, axis=0))
                    predictions = np.argmax(data_pre[0], axis=0)
                tp, fp, tn, fn = Compare2(predictions, data_labels[0])  # 计算metrics
                overall_p += tp + fn
                overall_n += tn + fp
                overall_tp += tp
                overall_tn += tn

            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity, 2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity, 2)
            else:
                specificity = 'None'
            if sensitivity != 'None' and specificity != 'None':
                score = (sensitivity + specificity) / 2
                score = round(score, 2)
            else:
                score = 'None'
            accuracy = (overall_tp + overall_tn) / (overall_p + overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end - start, 2)

            strg = '*[泛化性测试结果] 片段类型【{0}】 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(choice, sensitivity, specificity, score, accuracy, dtime)
            tqdm.write(strg)
            pbar.close()


if __name__ == '__main__':
    datapath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect'
    a = ModelSpeech(datapath)
    m1 = a.model
    # m1 = a.CreateClassicModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)
    print('cnn+dnn:',m1.count_params())
    print('cnn+dnn:', m1.summary())
    # print('inception:',a.model.count_params())

    # m2 = a.CreateInceptionModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)
    # print('inception:', m2.count_params())
    #
    # m3 = a.CreateLSTMModel(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), classes=CLASS_NUM)
    # print('LSTM:', m3.count_params())