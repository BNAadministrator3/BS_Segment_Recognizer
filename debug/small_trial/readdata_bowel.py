#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import platform as plat
import os
import keras
import sys
from os.path import dirname, join, abspath
from skimage.feature import hog

import numpy as np
sys.path.append('../../help_func')
from feature_transformation import GetFrequencyFeatures, read_wav_data, SimpleMfccFeatures
import random

PRIOR_ART = False#already choose hog features as the spec output!!!!
FEATURE_TYPE = 'mfcc'

AUDIO_LENGTH = 123  #size:200*197
AUDIO_FEATURE_LENGTH = 200 if FEATURE_TYPE in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM'] else 26
CLASS_NUM = 2

#For compatibility
MAX_AUDIO_LENGTH = AUDIO_LENGTH

class DataSpeech():

    def __init__(self, path, type, LoadToMem=False, MemWavCount=10000):
        '''
        参数：
            path：数据存放位置根目录
        '''
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        self.datapath = path  # 数据存放位置根目录
        self.type = type  # 数据类型，分为两种：训练集(train)、验证集(validation)

        self.slash = ''
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜杠
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜杠
        else:
            print('*[Warning] Unknown System\n')
            self.slash = '/'  # 正斜杠

        if (self.slash != self.datapath[-1]):  # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

        self.common_path = ''
        self.list_bowel1 = []
        self.list_non0 = []
        self.DataNum = ()  # 记录数据量
        self.LoadDataList()

        self.list_path = self.GenAll(self.type)

        self.class_num = CLASS_NUM

        self.feat_dimension = AUDIO_FEATURE_LENGTH
        self.frame_length = 400
        # self.max_time_step = self.GetMaxTimeStep()
        # self.min_time_step = self.GetMinTimeStep()

        pass

    def LoadDataList(self):
        '''
        加载用于计算的数据列表
        参数：
            type：选取的数据集类型
                train 训练集
                eval 验证集
        '''
        # 设定选取哪一项作为要使用的数据集

        if (self.type == 'train'):
            self.common_path = self.datapath + 'train' + self.slash
        elif (self.type == 'eval'):
            self.common_path = self.datapath + 'val' + self.slash
        else:
            print('*[Error] Index reading error!\n')
            assert (0)
        self.list_bowel1 = os.listdir(self.common_path + 'bowels')
        self.list_non0 = os.listdir(self.common_path + 'non')
        self.DataNum = (len(self.list_bowel1), len(self.list_non0)) #primary map

    def GenAll(self, type):

        s = []
        link = ('bowels','non')
        for i in link:
            list_name_folder = os.listdir(self.common_path + i)
            tag = 1 if i == 'bowels' else 0
            for j in list_name_folder:
                str = self.common_path + i + self.slash + j
                s.append((str,tag))
        random.shuffle(s)

        return s

    def listShuffle(self,terminus):
        temp = self.list_bowel1[0:terminus]
        random.shuffle(temp)
        self.list_bowel1[0:terminus] = temp
        temp = self.list_non0[0:terminus]
        random.shuffle(temp)
        self.list_non0[0:terminus] = temp


    def shifting(self,image,bias=39):
        bias=int(image.shape[0] *0.2)
        translation = random.randint(0,(bias-1)//3)*3
        case = random.randint(1,3)
        if case != 2:#up blank
            image[0:translation]=0
        if case != 1:
            image[-1-translation:-1] = 0
        return image

    def GetData(self, n_start, n_amount=32, featureType = 'spectrogram',  mode='balanced'):
        '''
        读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
        参数：
            n_start：从编号为n_start数据开始选取数据
            n_amount：选取的数据数量，默认为1，即一次一个wav文件
        返回：
            四个音频四个label，
        '''
        # 随机从四个文件夹中拿一条数据，判断是否大于1s，否就重拿
        assert(n_amount%CLASS_NUM==0)
        category = (self.list_bowel1, self.list_non0)
        link = ('bowels', 'non')
        label = (1, 0)
        # extract = {'spectrogram':GetFrequencyFeatures,'mfcc':mfccFeatures}
        path = ''
        if FEATURE_TYPE in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM']:
            if mode == 'balanced':
                data = []
                labels = []
                for genre in range(CLASS_NUM):
                    for file in range(n_amount//CLASS_NUM):
                        filename = category[genre][(n_start + file)%self.DataNum[genre]]
                        # filename = category[genre][(n_start + file) % min(self.DataNum)]
                        path = self.common_path + link[genre] + self.slash + filename
                        wavsignal, fs = read_wav_data(path)
                        # data_input = SimpleMfccFeatures(wavsignal, fs)
                        data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length,shift=160)
                        data_input = self.shifting(data_input)
                        if PRIOR_ART == True:
                            data_input = hog(data_input, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=False)
                        # data_input = MelSpectrogram(wavsignal, fs,frame_length = self.frame_length, shift=160,filternum = 26)
                        else:

                            data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                        data.append(data_input)
                        data_label = np.array([label[genre]])
                        labels.append(data_label)
                return data, labels
            if mode == 'non-repetitive':
                path = self.list_path[n_start][0]
                data_label = np.array([self.list_path[n_start][1]])
                wavsignal, fs = read_wav_data(path)
                data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length,shift=160)
                # data_input = MelSpectrogram(wavsignal, fs, frame_length=self.frame_length, shift=160, filternum=26)
                # data_input = SimpleMfccFeatures(wavsignal, fs)
                if PRIOR_ART == True:
                    data_input = hog(data_input, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),visualise=False)
                else:
                    data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                return data_input,  data_label
        elif FEATURE_TYPE in ['mfcc','MFCC','Mfcc']:
            if mode == 'balanced':
                data = []
                labels = []
                for genre in range(CLASS_NUM):
                    for file in range(n_amount//CLASS_NUM):
                        filename = category[genre][(n_start + file)%self.DataNum[genre]]
                        # filename = category[genre][(n_start + file) % min(self.DataNum)]
                        path = self.common_path + link[genre] + self.slash + filename
                        wavsignal, fs = read_wav_data(path)
                        data_input = SimpleMfccFeatures(wavsignal, fs)
                        data_label = np.array([label[genre]])
                        # if data_label[0] == 0:
                        # if PRIOR_ART == False:
                        data_input = self.shifting(data_input)
                        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                        data.append(data_input)
                        labels.append(data_label)
                return data, labels
            if mode == 'non-repetitive':
                path = self.list_path[n_start][0]
                data_label = np.array([self.list_path[n_start][1]])
                wavsignal, fs = read_wav_data(path)
                data_input = SimpleMfccFeatures(wavsignal, fs)
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                return data_input,  data_label
        else:
            print('Unknown feature type.')
            assert(0)


    def data_genetator(self, batch_size=32, epochs=0, audio_length=AUDIO_LENGTH):
        '''
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        '''

        assert(batch_size%CLASS_NUM==0)
        iterations_per_epoch = min(self.DataNum)//(batch_size//CLASS_NUM)+1
        self.listShuffle(min(self.DataNum))
        while True:
            ran_num = random.randint(0, iterations_per_epoch-1)  # 获取一个随机数
            origin = int(ran_num * batch_size // CLASS_NUM)
            bias = origin + epochs*min(self.DataNum)
            X,y = self.GetData(n_start=origin, n_amount=batch_size )
            X = np.array(X)
            y = np.array(y)
            yield [X, keras.utils.to_categorical(y, num_classes=self.class_num)], keras.utils.to_categorical(y, num_classes=self.class_num)  # 功能只是转成独热编码
        pass

class Testing():
    def __init__(self, pathSame, pathDistinct):
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        self.pathSame = pathSame[0]
        self.pathSameLabel = pathSame[1]
        self.pathDistinct = pathDistinct[0]
        self.pathDistinctLabel = pathDistinct[1]

        self.slash = ''
        if (system_type == 'Windows'):
            self.slash = '\\'  # 反斜杠
        elif (system_type == 'Linux'):
            self.slash = '/'  # 正斜杠
        else:
            print('*[Warning] Unknown System\n')
            self.slash = '/'  # 正斜杠

        if (self.slash != self.pathSame[-1]):  # 在目录路径末尾增加斜杠
            self.pathSame = self.pathSame + self.slash
        if (self.slash != self.pathDistinct[-1]):  # 在目录路径末尾增加斜杠
            self.pathDistinct = self.pathDistinct + self.slash

        self.LoadDataList()
        self.class_num = CLASS_NUM
        self.feat_dimension = AUDIO_FEATURE_LENGTH
        self.frame_length = 400

    def LoadDataList(self):
        self.listSame = []
        self.listDistinct = []
        link = ('bowels', 'non')
        for i in link:
            tag = 1 if i == 'bowels' else 0
            list_name_folder = os.listdir(self.pathSame + i)
            for j in list_name_folder:
                str = self.pathSame + i + self.slash + j
                self.listSame.append((str, tag))
            list_name_folder = os.listdir(self.pathDistinct + i)
            for j in list_name_folder:
                str = self.pathDistinct + i + self.slash + j
                self.listDistinct.append((str, tag))
        random.shuffle(self.listSame)
        random.shuffle(self.listDistinct)
        self.DataNum_Same = len(self.listSame)
        self.DataNum_Distinct = len(self.listDistinct)
        self.DataNum = {self.pathSameLabel:self.DataNum_Same,self.pathDistinctLabel:self.DataNum_Distinct}

    def GetData(self, n_start, n_amount=32, dataType = 'same'):
        assert(n_amount%CLASS_NUM==0)
        if dataType == self.pathSameLabel:
            path = self.listSame[n_start][0]
            data_label = np.array([self.listSame[n_start][1]])
        elif dataType == self.pathDistinctLabel:
            path = self.listDistinct[n_start][0]
            data_label = np.array([self.listDistinct[n_start][1]])
        wavsignal, fs = read_wav_data(path)
        if FEATURE_TYPE in ['spec', 'Spec', 'SPEC', 'Spectrogram', 'SPECTROGRAM']:
            data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length, shift=160)
            if PRIOR_ART == True:
               data_input = hog(data_input, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),visualise=False)
               return data_input, data_label
        elif FEATURE_TYPE in ['mfcc', 'MFCC', 'Mfcc']:
            data_input = SimpleMfccFeatures(wavsignal, fs)
        else:
            print('Unknown feature type.')
            assert (0)


        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        return data_input, data_label



if (__name__ == '__main__'):
    path = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/unbalanced'
    l = DataSpeech(path, 'train')
    l.LoadDataList()
    l.listShuffle(2)
    # print('max time step:', l.max_time_step)
    # print('max time step:', l.GetMaxTimeStep())
    # print('min time step:', l.min_time_step)
    # print('min time step:', l.GetMinTimeStep())
    # print('data size:', l.DataNum)
    print(l.GetData(random.randint(0,8000), mode='non-repetitive')[0].shape)
    aa = l.data_genetator(batch_size=32,)
    for i in aa:
        a, b = i
        print(a, b)
    pass
