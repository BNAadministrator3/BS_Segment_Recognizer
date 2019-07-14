#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import platform as plat
import os
import random
import shutil
import time
import keras
from keras.layers import *
import json
from distutils.dir_util import copy_tree


from help_func.feature_transformation import GetFrequencyFeatures, read_wav_data, SimpleMfccFeatures

AUDIO_LENGTH = 123  #size:200*197
CLASS_NUM = 2
SUBJECT_NUM = 20


# fold number; stochastic or specify
# input_folder = os.path.join(os.getcwd(),'dataset','scattered')
# output_folder = os.path.join(os.getcwd(),'dataset','constructed')
class trainvalFormation():
    def __init__(self,input_folder,output_folder,fold_number,aggregate_mode='random'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        if fold_number in {2,5,10}:
            self.fold_number = int(fold_number)
        else:
            print('can not aggregate data to {} folds. '.format(fold_number))
            assert(0)
        if aggregate_mode in {'random','specified'}:
            self.mode = aggregate_mode
        else:
            print("unresolved paramters:'aggregate_mode'.")
            assert(0)
        self.lookup_table = {}
        if self.mode == 'random':
            self.__innerconstrcution__()

#specify demo:
    # {'fold_0':('746031','762758','813278','910996'),
    # 'fold_1':('955552','1006643','1013057','1026977'),
    # 'fold_2':('1027757p','1028760p','1029382p','1034801'),
    # 'fold_3':('1035711','1036456','1037609','1037785',),
    # 'fold_4':('1037870','1038197','1039379','1040221')}

    def specifybywriting(self,lookup_table):
        if self.mode != 'specified':
            print('You should not call this method.')
            assert(0)
        elif isinstance(lookup_table,dict):
            if len(lookup_table)!=self.fold_number:
                print('lookup table only support manual assignment.')
                assert(0)
            else:
                self.lookup_table = lookup_table
        else:
            print('lookup table is not the dictionary type.')
            assert(0)

    def specifybyloading(self,path=None):
        if self.mode != 'specified':
            print('You should not call this method.')
            assert(0)
        else:
            self.__load2dict__(path)
            if len(self.lookup_table) != self.fold_number:
                print('lookup table is not correctly loaded.')
                assert (0)

    def __innerconstrcution__(self):
        folders = os.listdir(self.input_folder)
        #place a self-check
        if len(folders) != SUBJECT_NUM:
            print('error occurred when cheking subject numbers.')
            assert(0)
        else:
            random.shuffle(folders)
            for i in range(self.fold_number):
                strs = 'fold_'+str(i)
                values = tuple(folders[i*len(folders)//self.fold_number:(i+1)*len(folders)//self.fold_number])
                self.lookup_table.update({strs:values})
        self.__dumplookuptable__()

    def __dumplookuptable__(self):
        strs = 'lookuptable_'+str(self.fold_number)+'folds.json'
        path = os.path.join( os.getcwd(),strs)
        with open(path, 'w+') as fp:
            json.dump(self.lookup_table,fp)

    def __load2dict__(self,path=None):
        if path is None:
            strs = 'lookuptable_' + str(self.fold_number) + 'folds.json'
            path = os.path.join( os.getcwd(),strs)
        with open(path, 'r') as fp:
            if not self.lookup_table:
                pass
            else:
                print('the lookup table of the object is not empty! Possible error occurred.')
            self.lookup_table = json.load(fp)


    def __form_train_val_test__(self):
        #1. clear the output_folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        #2. reconstruct the out_put folder
        try:
            os.makedirs(self.output_folder)
            os.makedirs(os.path.join(self.output_folder, 'train'))
            os.makedirs(os.path.join(self.output_folder, 'train','bowels'))
            os.makedirs(os.path.join(self.output_folder, 'train', 'non'))
            os.makedirs(os.path.join(self.output_folder, 'val'))
            os.makedirs(os.path.join(self.output_folder, 'val','bowels'))
            os.makedirs(os.path.join(self.output_folder, 'val','non'))
            os.makedirs(os.path.join(self.output_folder, 'test'))
            os.makedirs(os.path.join(self.output_folder, 'test','bowels'))
            os.makedirs(os.path.join(self.output_folder, 'test','non'))
        except:
            print('folder creation fails.')
            assert(0)

    def subjects_transfer(self,subject_container):
        if os.path.exists(subject_container):
            shutil.rmtree(subject_container)
        os.makedirs(subject_container)
        for i in range(self.fold_number):
            strs = 'fold_' + str(i)
            folders = self.lookup_table[strs]
            for name in folders:
                src_folder = os.path.join(self.input_folder, name) #copy all content under src to dst
                dst_folder = os.path.join(subject_container,name)
                if os.path.exists(dst_folder):
                    shutil.rmtree(dst_folder)
                os.makedirs(dst_folder)
                assert(os.path.exists(src_folder))
                # copy_tree(src_folder,dst_folder)
                self.asic_copytree(src_folder,dst_folder)

    def asic_copytree(self,src_folder,dst_folder):
        dst_bowel = os.path.join( dst_folder,'bowels' )
        os.makedirs(dst_bowel)
        dst_non = os.path.join( dst_folder,'non' )
        os.makedirs(dst_non)
        src_bowel = os.path.join(src_folder, 'bowels')
        src_non = os.path.join(src_folder, 'non')
        for files in os.listdir(src_bowel):
            path = os.path.join(src_bowel, files)
            if os.path.isfile(path):
                shutil.copy(path, dst_bowel)
        for files in os.listdir(src_non):
            path = os.path.join(src_non, files)
            if os.path.isfile(path):
                shutil.copy(path, dst_non)
        len_src_bowel = len(os.listdir(src_bowel))
        len_src_non = len(os.listdir(src_non))
        len_dst_bowel = len(os.listdir(dst_bowel))
        len_dst_non = len(os.listdir(dst_non))
        assert(len_src_bowel == len_dst_bowel)
        assert(len_src_non == len_dst_non)

    def files_transfer(self):
        self.__form_train_val_test__()
        for i in range(self.fold_number):
            sent = 'train'
            if i == self.fold_number-1:
                sent = 'test'
            dst_bowel = os.path.join(self.output_folder, sent, 'bowels')
            dst_non = os.path.join(self.output_folder, sent, 'non')
            strs = 'fold_' + str(i)
            folders = self.lookup_table[strs]
            for name in folders:
                src_bowel = os.path.join(self.input_folder,name,'bowels')
                src_non = os.path.join(self.input_folder, name, 'non')
                for files in os.listdir(src_bowel):
                    path = os.path.join(src_bowel,files)
                    if os.path.isfile(path):
                        shutil.copy(path,dst_bowel)
                for files in os.listdir(src_non):
                    path = os.path.join(src_non,files)
                    if os.path.isfile(path):
                        shutil.copy(path,dst_non)

        files_bowel = os.listdir(os.path.join(self.output_folder,'train','bowels'))
        random.shuffle(files_bowel)
        for files in files_bowel[0:len(files_bowel)//10]:
            path = os.path.join(self.output_folder,'train','bowels',files)
            if os.path.isfile(path):
                shutil.move(path, os.path.join(self.output_folder,'val','bowels')) #error!should be cut into
        files_non = os.listdir(os.path.join(self.output_folder, 'train', 'non'))
        random.shuffle(files_non)
        for files in files_non[0:len(files_non) // 10]:
            path = os.path.join(self.output_folder, 'train', 'non', files)
            if os.path.isfile(path):
                shutil.move(path, os.path.join(self.output_folder, 'val', 'non')) #error!should be cut into

    def loop_files_transfer(self,needle=0):
        if needle in set(range(self.fold_number)):
            pass
        else:
            print('uncorrected setting!')
            assert(0)

        self.__form_train_val_test__()
        for i in range(self.fold_number):
            sent = 'train'
            if i == needle:
                sent = 'test'
            dst_bowel = os.path.join(self.output_folder, sent, 'bowels')
            dst_non = os.path.join(self.output_folder, sent, 'non')
            strs = 'fold_' + str(i)
            folders = self.lookup_table[strs]
            for name in folders:
                src_bowel = os.path.join(self.input_folder,name,'bowels')
                src_non = os.path.join(self.input_folder, name, 'non')
                for files in os.listdir(src_bowel):
                    path = os.path.join(src_bowel,files)
                    if os.path.isfile(path):
                        shutil.copy(path,dst_bowel)
                for files in os.listdir(src_non):
                    path = os.path.join(src_non,files)
                    if os.path.isfile(path):
                        shutil.copy(path,dst_non)

        files_bowel = os.listdir(os.path.join(self.output_folder,'train','bowels'))
        random.shuffle(files_bowel)
        for files in files_bowel[0:len(files_bowel)//10]:
            path = os.path.join(self.output_folder,'train','bowels',files)
            if os.path.isfile(path):
                shutil.move(path, os.path.join(self.output_folder,'val','bowels')) #error!should be cut into
        files_non = os.listdir(os.path.join(self.output_folder, 'train', 'non'))
        random.shuffle(files_non)
        for files in files_non[0:len(files_non) // 10]:
            path = os.path.join(self.output_folder, 'train', 'non', files)
            if os.path.isfile(path):
                shutil.move(path, os.path.join(self.output_folder, 'val', 'non')) #error!should be cut into


    def files_check(self):
        train_bowel = os.path.join(self.output_folder, 'train', 'bowels')
        train_non = os.path.join(self.output_folder, 'train', 'non')
        val_bowel = os.path.join(self.output_folder, 'val', 'bowels')
        val_non = os.path.join(self.output_folder, 'val', 'non')
        test_bowel = os.path.join(self.output_folder, 'test', 'bowels')
        test_non = os.path.join(self.output_folder, 'test', 'non')

        train_number = len(os.listdir(train_bowel))+len(os.listdir(train_non))
        val_number = len(os.listdir(val_bowel))+len(os.listdir(val_non))
        test_number = len(os.listdir(test_bowel))+len(os.listdir(test_non))

        print('training set size:{}; val set size:{}; test set size:{}'.format(train_number,val_number,test_number))

def clrdir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            clrdir(c_path)
        else:
            os.remove(c_path)

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        clrdir(path)
        print(path + ' 目录已存在,已清空里面内容')
        return False

def stringCheck(feature_type, module_type, layer_numbers):
    if feature_type in ('spec', 'Spec', 'SPEC', 'Spectrogram', 'SPECTROGRAM'):
        new_feature_type = 'spec'.upper()
    elif feature_type in ('mfcc', 'Mfcc', 'MFCC', 'MFC', 'mfc', 'Mfc'):
        new_feature_type = 'mfcc'.upper()
    else:
        print('*[ERROR]Unknown feature type.')
        assert (0)

    if module_type in ('regular', 'residual', 'inception'):
        new_module_type = module_type.capitalize()
    else:
        print('*[ERROR]Out of set: module_type.')
        assert (0)

    if layer_numbers in (2, 4, 6, 8, 10):
        new_layer_counts = layer_numbers
    else:
        print('*[ERROR]Out of set: layer_counts.')
        assert (0)
    return new_feature_type,new_module_type,new_layer_counts

class pathpoper():
    def __init__(self):
        base = os.path.split(os.path.realpath(__file__))[0]
        self.root = os.path.join(base, 'CNNdesign')
        child_mreg = 'mfcc+regular'
        child_sreg = 'spec+regular'
        child_mres = 'mfcc+residual'
        child_sres = 'spec+residual'
        child_mi = 'mfcc+inception'
        child_si = 'spec+inception'
        subfolders = (child_mreg, child_sreg, child_mres, child_sres, child_mi, child_si)
        for subfolder in subfolders:
            for layers in ('2', '4', '6', '8', '10'):
                mkdir(os.path.join(self.root, subfolder, layers))

    def popup(self,feature_type,module_type,layers):
        return os.path.join(self.root,feature_type.lower()+'+'+module_type.lower(),str(layers))

class DataSpeech():
    def __init__(self, path, feature_type, type):
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

        self.feature_type = feature_type
        self.feat_dimension = 200 if self.feature_type in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM'] else 26

        self.common_path = ''
        self.list_bowel1 = []
        self.list_non0 = []
        self.DataNum = ()  # 记录数据量
        self.LoadDataList()

        self.list_path = self.GenAll(self.type)

        self.class_num = CLASS_NUM
        self.frame_length = 400
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

    def GetData(self, n_start, n_amount=32,  mode='balanced'):
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
        if self.feature_type in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM']:
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
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                return data_input,  data_label
        elif self.feature_type in ['mfcc','MFCC','Mfcc']:
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
    def __init__(self, feature_type, pathSame, pathDistinct):
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
        self.feature_type = feature_type
        self.feat_dimension = 200 if self.feature_type in ['spec','Spec','SPEC','Spectrogram','SPECTROGRAM'] else 26
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
        if self.feature_type in ['spec', 'Spec', 'SPEC', 'Spectrogram', 'SPECTROGRAM']:
            data_input = GetFrequencyFeatures(wavsignal, fs, self.feat_dimension, self.frame_length, shift=160)
        elif self.feature_type in ['mfcc', 'MFCC', 'Mfcc']:
            data_input = SimpleMfccFeatures(wavsignal, fs)
        else:
            print('Unknown feature type.')
            assert (0)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        return data_input, data_label





if __name__ =='__main__':
    print(os.getcwd())
    input = os.path.join(os.getcwd(),'dataset','scattered','standard1')
    output = os.path.join(os.getcwd(),'dataset','constructed')
    looper = trainvalFormation(input,output,5,'specified')
    looper.specifybyloading()
    print('The specified result is: ',looper.lookup_table)
    i=1
    dur = time.time()
    looper.loop_files_transfer(i)
    dur = round(time.time() - dur,2)
    print('formation time:{}s'.format(dur))
    print('{}th check:'.format(i))
    looper.files_check()