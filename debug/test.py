# ensemble = [80.19,81.79,74.28,78.38,73.5]
# lstm = [59.13,59.76,60.89,59.75,51.19]
# lstmno = [58.88, 58.14, 60.2, 59.19, 51.31]
# fc = [58,56.88,60.14,59.62,52.56]
# print(np.mean(fc))
# print(np.std(fc))


import os
import time
import platform as plat
import random
from tqdm import tqdm
import json
from collections import Counter

import tensorflow as tf
import keras
from keras.layers import *
from keras import optimizers
from keras.models import Model
import keras.backend as k


import shutil
from help_func.evaluation import Compare2,plot_confusion_matrix
from help_func.utilities import focal_loss, ReguBlock, ResiBlock, XcepBlock
from help_func.feature_transformation import GetFrequencyFeatures, read_wav_data, SimpleMfccFeatures

AUDIO_LENGTH = 123  #size:200*197
CLASS_NUM = 2
SUBJECT_NUM = 20

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
        path = ''
        data_label = ''
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

class foldermanager():
    def __init__(self,fold_number,parentdir):
        if not (fold_number in (2,5,10)):
            print('uunconsistent parameters.')
            assert(0)
        else:
            self.fold_number = fold_number
        self.parentdir = parentdir
        if os.path.exists(self.parentdir):
            shutil.rmtree(self.parentdir)
        os.makedirs(self.parentdir)


    def creation_desginfolder(self,approach,keytree):
        assert(approach.upper() in ('CNN','LSTM','FC'))
        if isinstance(keytree,list) or isinstance(keytree,tuple):
            basepaths = []
            designdir = os.path.join(self.parentdir, approach.upper() + 'design')
            basepaths.append(designdir)
            for stage in keytree:
                assert(isinstance(stage,tuple))
                temppaths = []
                for basepath in basepaths:
                    for element in stage:
                        temppaths.append(os.path.join(basepath,str(element)))
                        os.makedirs(os.path.join(basepath,str(element)))
                basepaths = temppaths
        else:
            assert(0)

    # This method does not give more detailed classification about evaluation folder.
    def creation_evaluationfolder(self,approach, addlevel = None):
        assert (approach.upper() in ('CNN', 'LSTM', 'FC'))
        if not addlevel:
            evaldir = os.path.join(self.parentdir, approach.upper() + 'eval')
        else:
            assert( addlevel in ('mfcc','spec') )
            evaldir = os.path.join(self.parentdir, approach.upper() + 'eval',addlevel)
        os.makedirs(evaldir)
        for i in range(self.fold_number):
            strs = 'i={}'.format(str(i))
            deep = os.path.join(evaldir, strs)
            os.makedirs(deep)

    def trnasferfiles(self,src_dir,dst_dir):
        if os.path.exists(src_dir) and os.path.exists(dst_dir):
            src_files = os.listdir(src_dir)
            assert(len(src_files)!=0)
            for file_name in src_files:
                full_file_name = os.path.join(src_dir, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dst_dir)


    def deletefolder(self,approach,phase):
        assert (approach.upper() in ('CNN', 'LSTM', 'FC'))
        assert( phase.lower() in ('design','evaluation') )
        folder = os.path.join(self.parentdir, approach.upper() + phase.lower())
        if os.path.exists(folder):
            shutil.rmtree(folder)
        else:
            assert(0)

class folderoperation():
    def __init__(self,fold_number,parentdir=None):
        if not (fold_number in (2,5,10)):
            print('uunconsistent parameters.')
            assert(0)
        else:
            self.fold_number = fold_number
        self.parentdir = parentdir
        if not self.parentdir:
            self.parentdir = os.getcwd()
        self.base = os.path.join(self.parentdir, 'CNNevaluation')

    def creation(self):
        regular = os.path.join(self.base,'regular')
        residual = os.path.join(self.base,'residual')
        inceptional = os.path.join(self.base,'inception')
        for i in range(self.fold_number):
            strs = 'i={}'.format(i)
            regi = os.path.join(regular,'reg'+strs)
            os.makedirs(regi)
            resi = os.path.join(residual, 'res' + strs)
            os.makedirs(resi)
            inci = os.path.join(inceptional, 'inc' + strs)
            os.makedirs(inci)

    def deletion(self):
        if os.path.exists(self.base):
            shutil.rmtree(self.base)

class Network():
    def __init__(self):
        print("Let's begin!")
        pass #it seems all the network can be described using a single function

    def CNNForest(self,feature_type, module_type, layer_counts):
        feature_length = 200 if feature_type == 'SPEC' else 26
        input_shape = (AUDIO_LENGTH, feature_length, 1)
        dictmap = {'Regular':ReguBlock, 'Residual':ResiBlock, 'Inception':XcepBlock}
        Module = dictmap[module_type]
        X_input = Input(name='the_input', shape=input_shape)
        level_h1 = Module(32)(X_input)
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = Module(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = Module(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层
        level_h4 = Module(256)(level_m3)
        level_m4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h4)  # 池化层
        level_s5 = Module(512)(level_m4)
        if feature_type == 'SPEC':
            level_s5 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_s5)  # 池化层
        layers = [level_m1,level_m2,level_m3,level_m4,level_s5]
        output = layers[layer_counts//2-1]
        flayer = GlobalAveragePooling2D()(output)
        fc2 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax', name='Activation0')(fc2)

        model = Model(inputs=X_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        print('{} cnn model with the {} feature and {} layers are estabished.'.format(module_type, feature_type, str(layer_counts)))
        modelname = feature_type+'_'+module_type+'_'+str(layer_counts)
        return model,modelname

    def LSTMbush(self, unit_counts, feature_type='MFCC'):
        assert(unit_counts in (64,128,256))
        feature_length = 200 if feature_type == 'SPEC' else 26
        input_shape = (AUDIO_LENGTH, feature_length, 1)
        X_input = Input(name='the_input', shape=input_shape)
        x = Reshape((AUDIO_LENGTH, feature_length), name='squeeze')(X_input)
        y = LSTM(unit_counts, return_sequences=False)(x)  # computation complexity
        y_pred = Dense(CLASS_NUM, activation='softmax')(y)
        lstmmodel = Model(inputs=X_input, outputs=y_pred)
        lstmmodelname = feature_type.lower() + '_lstm_'+str(unit_counts)
        print('The lstm model with {} featue and {} states is established.'.format(feature_type.lower(), str(unit_counts)))
        self.ModelTrainingSetting(lstmmodel)
        return lstmmodel, lstmmodelname

    def FCbush(self,unit_counts, feature_type='MFCC'):
        assert(unit_counts in (128, 256, 500, 1000))
        feature_length = 200 if feature_type == 'SPEC' else 26
        input_shape = (AUDIO_LENGTH, feature_length, 1)
        X_input = Input(name='the_input', shape=input_shape)
        x = Flatten(name='squeeze')(X_input)
        y1 = Dense(1000, kernel_regularizer=regularizers.l2(0.0005), activation='relu')(x)  # computation complexity
        y2 = Dense(1000, kernel_regularizer=regularizers.l2(0.0005), activation='relu')(y1)
        y3 = Dense(unit_counts, kernel_regularizer=regularizers.l2(0.0005), activation='relu')(y2)
        y_pred = Dense(CLASS_NUM, activation='softmax')(y3)
        fcmodel = Model(inputs=X_input, outputs=y_pred)
        fcmodelname = feature_type.lower() + '_fc_' + str(unit_counts)
        print('The fc model with {} featue and [1000 1000 {}] layers is established.'.format(feature_type.lower(),str(unit_counts)))
        self.ModelTrainingSetting(fcmodel)
        return fcmodel, fcmodelname

    def ModelTrainingSetting(self, model):
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])

    def ModelWeigthsLoading(self, model, weightsdir):
        WeightsFile = [i for i in os.listdir(weightsdir) if 'weights' in i]
        assert(len(WeightsFile)==1)
        model.load_weights(os.path.join(weightsdir,WeightsFile[0]))
        for layer in model.layers:
            layer.trainable = False

class operation():
    def __init__(self,model,modelname,basePath):
        self.model = model
        self.clearPath = basePath
        self.basePath = os.path.join(self.clearPath,modelname)
        self.baseSavPath = []
        self.baseSavPath.append(self.basePath)
        self.baseSavPath.append(self.basePath+'_weights')
        self.modelname = modelname

    def train(self,datapath,feature_type,batch_size=32,epoch=20):
        assert (batch_size % CLASS_NUM == 0)
        data = DataSpeech(datapath,feature_type, 'train')
        num_data = sum(data.DataNum)  # 获取数据的数�?
        os.system('pkill tensorboard')
        os.system('rm -rf ./checkpoints/files_summary/* ')
        train_writter = tf.summary.FileWriter(os.path.join(os.getcwd(), 'checkpoints', 'files_summary'))
        os.system('tensorboard --logdir=/home/zhaok14/example/PycharmProjects/setsail/individual_spp/checkpoints/files_summary/ --port=6006 &')
        print('\n')
        print(90 * '*')
        print(40 * ' ',self.modelname)
        print(90 * '*')

        iterations_per_epoch = min(data.DataNum) // (batch_size // CLASS_NUM) + 1
        # iterations_per_epoch = 30
        print('trainer info:')
        print('training data size: %d' % num_data)
        print('increased epoches: ', epoch)
        print('minibatch size: %d' % batch_size)
        print('iterations per epoch: %d' % iterations_per_epoch)

        sess = k.get_session()
        train_writter.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        best_score = 0
        # epoch = 2
        duration = 0
        for i in range(0, epoch):
            iteration = 0
            yielddatas = data.data_genetator(batch_size, epoch)
            pbar = tqdm(yielddatas)
            for input, labels in pbar:
                stime = time.time()
                loss = self.model.train_on_batch(input[0], labels)
                dtime = time.time() - stime
                duration = duration + dtime
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

            self.TestModel(sess=sess, feature_type = feature_type, datapath=datapath, str_dataset='train', data_count=1000, writer=train_writter, step=i)
            metrics = self.TestModel(sess=sess, feature_type = feature_type, datapath=datapath, str_dataset='eval', data_count=-1, writer=train_writter, step=i)
            if i > 0:
                if metrics['score'] >= best_score:
                    self.metrics = metrics
                    self.metrics['epoch'] = i
                    best_score = metrics['score']
                    clrdir(self.clearPath)
                    self.savpath = []
                    self.savpath.append((self.baseSavPath[0] + '_epoch' + str(i) + '.h5'))
                    self.savpath.append((self.baseSavPath[1] + '_epoch' + str(i) + '.h5'))
                    self.model.save(self.savpath[0])
                    self.model.save_weights(self.savpath[1])
        if 'epoch' in self.metrics.keys():
            print('The best metric (without restriction) took place in the epoch: ', self.metrics['epoch'])
            print('Sensitivity: {}; Specificity: {}; Score: {}; Accuracy: {}'.format(self.metrics['sensitivity'],self.metrics['specificity'],self.metrics['score'],self.metrics['accuracy']))
            # self.TestGenerability(feature_type = feature_type, weightspath=self.savpath[1])
        else:
            print('The best metric (without restriction) is not found. Done!')
        print('Training duration: {}s'.format(round(duration, 2)))
        return self.metrics['accuracy']

    def TestModel(self, sess, writer, feature_type, datapath='', str_dataset='eval', data_count=32, show_ratio=True, step=0):
        '''
        测试检验模型效果
        '''
        data = DataSpeech(datapath, feature_type, str_dataset)
        num_data = sum(data.DataNum)  # 获取数据的数量
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data
        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            start = time.time()
            cm_pre = []
            cm_lab = []
            map = {0: 'normal', 1: 'bowel sounds'}
            # data_count = 200
            for i in tqdm(range(data_count)):
                data_input, data_labels = data.GetData((ran_num + i) % num_data, mode='non-repetitive')  # 从随机数开始连续向后取一定数量数据
                data_pre = self.model.predict_on_batch(np.expand_dims(data_input, axis=0))
                predictions = np.argmax(data_pre[0], axis=0)
                cm_pre.append(map[predictions])
                cm_lab.append(map[data_labels[0]])
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
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(str_dataset,sensitivity,specificity, score,accuracy, dtime)
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

            metrics = {'data_set': str_dataset, 'sensitivity': sensitivity, 'specificity': specificity, 'score': score,'accuracy': accuracy}
            return metrics

        except StopIteration:
            print('*[Error] Model Test Error. please check data format.')

    def __dataTesting__(self, feature_type, dataSourceA, dataSourceB, weightspath):
        data = Testing(feature_type, dataSourceA, dataSourceB)
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
import sys
class logger(object):
    def __init__(self,filename=None):
        if filename is None:
            filename = os.path.join(os.getcwd(), 'Default.log')
        self.terminal = sys.stdout
        self.log = open(filename,'w')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

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

def evaluation(dataSourceA, dataSourceB,model,modelname,featureType='spec',votingchecking=False):
    data = Testing(featureType, dataSourceA, dataSourceB)
    strg = 'The tested model is {}.'.format(modelname)
    tqdm.write(strg)
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
            data_pre = model.predict_on_batch(np.expand_dims(data_input, axis=0))
            if votingchecking == False:
                predictions = np.argmax(data_pre[0], axis=0)
            else:
                predictions = []
                for prob in data_pre[0]:
                    if prob[1] > prob[0]:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                predictions = Counter(predictions).most_common(1)[0][0]
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
        if choice == dataSourceB[1]:
            return accuracy

if __name__ == '__main__':
    from keras.backend.tensorflow_backend import set_session
    import gc
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配
    set_session(tf.Session(config=config))
    temp_path = os.path.join(os.getcwd(),'files_json')
    jsonfiles = []
    for i in (5,10,15):
        strs = 'lookuptable_5folds_'+str(i)+'.json'
        jsonfiles.append(os.path.join(temp_path,strs))
    #print(jsonfiles)
    temp_path = os.path.join(os.getcwd(),'files_h5')
    h5folders = []
    for i in (5,10,15):
        strs = 'subjects_'+str(i)
        h5folders.append(os.path.join(temp_path,strs))
    # print(h5folders)
    file_trans_input = os.path.join(os.getcwd(),'dataset','scattered','standard1' )
    file_trans_output = os.path.join(os.getcwd(),'dataset','difsizes')
    valpath = os.path.join(file_trans_output, 'val')
    testpath = os.path.join(file_trans_output, 'test')
    report1 = [valpath, 'validation']
    report2 = [testpath, 'test']
    sys.stdout = logger(filename=os.path.join(os.getcwd(),'log&&materials','Bigloop_multipledesignevalutionresults.log'))
    # This loop is for different sizess of datasets
    topCon = {'CNN':[],'LSTM':[],'FC':[]}
    wholest = time.time()
    for jsonfile, h5folder in zip(jsonfiles,h5folders):
        datalooper = trainvalFormation(file_trans_input,file_trans_output,5,'specified')
        datalooper.specifybyloading(path=jsonfile)
        print()
        print()
        print()
        print('#'*80)
        print('The followed data formation jsonfile:{}'.format( jsonfile.split('/')[-1] ))
        print('#'*80)
        print()
        nn = Network()
        commonfolger = foldermanager(5, h5folder)
#################################################################################################################################
        #phase 1 cnn affairs
        print('^' * 80)
        print('Firstly comes CNN..')
        print('^' * 80)
        #step1 1st fold split is always used for design
        print('This time the corrected stft features are applied....')
        print()
        print(40 * '-' + 'DESIGNSTEP:ROUND_1' + 40 * '-')
        print()
        #step2 denote the model, weight files and stock up the dataset
        datalooper.loop_files_transfer(1)
        hierachy = (('spec+regular','spec+residual','spec+inception'),(2,4,6,8,10))
        commonfolger.creation_desginfolder('cnn', hierachy)
        #step3 generate and train different networks to locate the highest one
        st = time.time()
        cnn_designprofile = []
        for module_type in ('Regular', 'Residual', 'Inception'):
            for layer_counts in (2, 4, 6, 8, 10):
                print()
                print(40 * '-' + 'CNNHANDLEFLAG' + 40 * '-')
                print()
                model, name = nn.CNNForest('SPEC', module_type, layer_counts)
                path = os.path.join( h5folder,'CNNdesign','spec+'+module_type.lower(), str(layer_counts) )
                controller = operation(model, name, path)
                accu = controller.train(file_trans_output, 'spec')
                cnn_designprofile.append((module_type, layer_counts, accu))
                gc.collect()
        en = time.time() - st
        hour = en // 3600
        minute = (en - (hour * 3600)) // 60
        seconds = en - (hour * 3600) - (minute * 60)
        print(40 * '-' + 'CNNDESIGNRESULTS' + 40 * '-')
        print('Overall design time for our CNN model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))
        #step4 make a decision
        cnn_accu = [t[2] for t in cnn_designprofile]
        cnn_final = cnn_designprofile[ cnn_accu.index(max(cnn_accu)) ]#flag
        print('The optimal CNN model is with {} module and {} layers'.format( cnn_final[0].lower(), str(cnn_final[1]) ))
        print('The highest training accuracy: {}%'.format(cnn_final[2]))
        print()
        print('Now we start evaluating our CNN model..')
        #step5 transfer file into the specified evaluation folder
        commonfolger.creation_evaluationfolder('cnn')
        src_path = os.path.join(h5folder,'CNNdesign','spec+'+cnn_final[0].lower(), str(cnn_final[1]))
        dst_path = os.path.join(h5folder,'CNNeval','i=1')
        commonfolger.trnasferfiles(src_path,dst_path)

        #step6 launch 5-fold evaluation
        cnn_accu = []
        st = time.time()
        print()
        print(40 * '-' + 'EVALSTEP:ROUND_1' + 40 * '-')
        print()
        # datalooper.loop_files_transfer(1)
        print('this checking we do not need training..')
        model, name = nn.CNNForest('SPEC', cnn_final[0], cnn_final[1])#its a excellent to fix cnn_final since its too vital to be changes any little.
        nn.ModelWeigthsLoading(model,dst_path)
        accu = evaluation(report1, report2, model, name)
        cnn_accu.append(accu)
        #step5.5 it is changable for the other folds
        for i in (0, 2, 3, 4):
            strg = 'EVALSTEP:ROUND_{}'.format(str(i))
            print()
            print(40 * '-' + strg + 40 * '-')
            print()
            print('this checking we do need training each model..')
            datalooper.loop_files_transfer(i)
            model, name = nn.CNNForest('SPEC', cnn_final[0], cnn_final[1])
            path = os.path.join( h5folder,'CNNeval', 'i={}'.format(str(i)))
            controller = operation(model, name, path)
            controller.train(file_trans_output, 'spec')
            nn.ModelWeigthsLoading(model,path)
            accu = evaluation(report1, report2, model, name)
            cnn_accu.append(accu)
            gc.collect()

        mean = np.mean(cnn_accu)
        stddev = np.std(cnn_accu)
        topCon['CNN'].append((cnn_final,(mean,stddev)))
        en = time.time() - st
        hour = en // 3600
        minute = (en - (hour * 3600)) // 60
        seconds = en - (hour * 3600) - (minute * 60)
        print(40 * '-' + 'CNNEVALUATIONRESULTS' + 40 * '-')
        print('Overall evaluation time for our CNN model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))
        print('Evaluation accuracy of our CNN model: {} +/- {}'.format(round(mean,2),round(stddev,2)))
        print('Appendix regarding desgin resulinfo: ')
        print('The optimal CNN model is with {} module and {} layers'.format(cnn_final[0].lower(), str(cnn_final[1])))
        print('The highest training accuracy: {}%'.format(cnn_final[2]))
        del cnn_accu, cnn_final, cnn_designprofile, mean, stddev


#################################################################################################################################
        #phase 2 lstm affairs
        print('^' * 80)
        print('Secondly comes LSTM..')
        print('^' * 80)
        #step1 1st fold split is always used for design
        print('This time the mfcc features are applied....')
        print()
        print(40 * '-' + 'DESIGNSTEP:ROUND_1' + 40 * '-')
        print()
        #step2 denote the model, weight files and stock up the dataset
        datalooper.loop_files_transfer(1)
        hierachy = ((64,128,256),)
        commonfolger.creation_desginfolder('lstm', hierachy)
        #step3 generate and train different networks to locate the highest one
        st = time.time()
        lstm_designprofile = []
        for units in (64,128,256):
            print()
            print(40 * '-' + 'LSTMHANDLEFLAG' + 40 * '-')
            print()
            model, name = nn.LSTMbush(units)
            path = os.path.join( h5folder,'LSTMdesign', str(units) )
            controller = operation(model, name, path)
            accu = controller.train(file_trans_output, 'mfcc')
            lstm_designprofile.append((units, accu))
            gc.collect()
        en = time.time() - st
        hour = en // 3600
        minute = (en - (hour * 3600)) // 60
        seconds = en - (hour * 3600) - (minute * 60)
        print(40 * '-' + 'LSTMDESIGNRESULTS' + 40 * '-')
        print('Overall design time for the LSTM model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))
        #step4 make a decision
        lstm_accu = [t[1] for t in lstm_designprofile]
        lstm_final = lstm_designprofile[ lstm_accu.index(max(lstm_accu)) ]#flag
        print('The optimal LSTM model is with {} units'.format( str(lstm_final[0]) ))
        print('The highest training accuracy: {}%'.format(lstm_final[1]))
        print()
        print('Now we start evaluating the LSTM model..')
        #step5 transfer file into the specified evaluation folder
        commonfolger.creation_evaluationfolder('lstm')
        src_path = os.path.join(h5folder,'LSTMdesign',str(lstm_final[0]))
        dst_path = os.path.join(h5folder,'LSTMeval','i=1')
        commonfolger.trnasferfiles(src_path,dst_path)

        #step6 launch 5-fold evaluation
        lstm_accu = []
        st = time.time()
        print()
        print(40 * '-' + 'EVALSTEP:ROUND_1' + 40 * '-')
        print()
        # datalooper.loop_files_transfer(1)
        print('this checking we do not need training..')
        model, name = nn.LSTMbush(lstm_final[0])
        nn.ModelWeigthsLoading(model,dst_path)
        accu = evaluation(report1, report2, model, name, featureType='mfcc')
        lstm_accu.append(accu)
        #step5.5 it is changable for the other folds
        for i in (0, 2, 3, 4):
            strg = 'EVALSTEP:ROUND_{}'.format(str(i))
            print()
            print(40 * '-' + strg + 40 * '-')
            print()
            print('this checking we do need training each model..')
            datalooper.loop_files_transfer(i)
            model, name = nn.LSTMbush(lstm_final[0])
            path = os.path.join( h5folder,'LSTMeval', 'i={}'.format(str(i)))
            controller = operation(model, name, path)
            controller.train(file_trans_output, 'mfcc')
            nn.ModelWeigthsLoading(model,path)
            accu = evaluation(report1, report2, model, name, featureType='mfcc')
            lstm_accu.append(accu)
            gc.collect()

        mean = np.mean(lstm_accu)
        stddev = np.std(lstm_accu)
        topCon['LSTM'].append((lstm_final,(mean,stddev)))
        en = time.time() - st
        hour = en // 3600
        minute = (en - (hour * 3600)) // 60
        seconds = en - (hour * 3600) - (minute * 60)
        print(40 * '-' + 'LSTMEVALUATIONRESULTS' + 40 * '-')
        print('Overall evaluation time for the LSTM model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))
        print('Evaluation accuracy of the LSTM model: {} +/- {}'.format(round(mean,2), round(stddev,2)))
        print('Appendix regarding desgin resulinfo: ')
        print('The optimal LSTM model is with {} units'.format(str(lstm_final[0])))
        print('The highest training accuracy: {}%'.format(lstm_final[1]))
        del lstm_accu, lstm_final, lstm_designprofile, mean, stddev

#################################################################################################################################
        # phase 3 fc affairs
        print('^' * 80)
        print('Thirdly comes FC..')
        print('^' * 80)
        # step1 1st fold split is always used for design
        print('This time the mfcc features are applied....')
        print()
        print(40 * '-' + 'DESIGNSTEP:ROUND_1' + 40 * '-')
        print()
        # step2 denote the model, weight files and stock up the dataset
        datalooper.loop_files_transfer(1)
        hierachy = ((128, 256, 500, 1000),)
        commonfolger.creation_desginfolder('fc', hierachy)
        # step3 generate and train different networks to locate the highest one
        st = time.time()
        fc_designprofile = []
        for units in (128, 256, 500, 1000):
            print()
            print(40 * '-' + 'FCHANDLEFLAG' + 40 * '-')
            print()
            model, name = nn.FCbush(units)
            path = os.path.join(h5folder, 'FCdesign', str(units))
            controller = operation(model, name, path)
            accu = controller.train(file_trans_output, 'mfcc')
            fc_designprofile.append((units, accu))
            gc.collect()
        en = time.time() - st
        hour = en // 3600
        minute = (en - (hour * 3600)) // 60
        seconds = en - (hour * 3600) - (minute * 60)
        print(40 * '-' + 'FCDESIGNRESULTS' + 40 * '-')
        print('Overall design time for the FC model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour),int(minute),seconds))
        # step4 make a decision
        fc_accu = [t[1] for t in fc_designprofile]
        fc_final = fc_designprofile[fc_accu.index(max(fc_accu))]  # flag
        print('The optimal FC model is with {} units'.format(str(fc_final[0])))
        print('The highest training accuracy: {}%'.format(fc_final[1]))
        print()
        print('Now we start evaluating the fc model..')
        # step5 transfer file into the specified evaluation folder
        commonfolger.creation_evaluationfolder('fc')
        src_path = os.path.join(h5folder, 'FCdesign', str(fc_final[0]))
        dst_path = os.path.join(h5folder, 'FCeval', 'i=1')
        commonfolger.trnasferfiles(src_path, dst_path)

        # step6 launch 5-fold evaluation
        fc_accu = []
        st = time.time()
        print()
        print(40 * '-' + 'EVALSTEP:ROUND_1' + 40 * '-')
        print()
        # datalooper.loop_files_transfer(1)
        print('this checking we do not need training..')
        model, name = nn.FCbush(fc_final[0])
        nn.ModelWeigthsLoading(model, dst_path)
        accu = evaluation(report1, report2, model, name, featureType='mfcc')
        fc_accu.append(accu)
        # step5.5 it is changable for the other folds
        for i in (0, 2, 3, 4):
            strg = 'EVALSTEP:ROUND_{}'.format(str(i))
            print()
            print(40 * '-' + strg + 40 * '-')
            print()
            print('this checking we do need training each model..')
            datalooper.loop_files_transfer(i)
            model, name = nn.FCbush(fc_final[0])
            path = os.path.join(h5folder, 'FCeval', 'i={}'.format(str(i)))
            controller = operation(model, name, path)
            controller.train(file_trans_output, 'mfcc')
            nn.ModelWeigthsLoading(model, path)
            accu = evaluation(report1, report2, model, name, featureType='mfcc')
            fc_accu.append(accu)
            gc.collect()

        mean = np.mean(fc_accu)
        stddev = np.std(fc_accu)
        topCon['FC'].append((fc_final,(mean, stddev)))
        en = time.time() - st
        hour = en // 3600
        minute = (en - (hour * 3600)) // 60
        seconds = en - (hour * 3600) - (minute * 60)
        print(40 * '-' + 'FCEVALUATIONRESULTS' + 40 * '-')
        print('Overall evaluation time for the FC model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour),int(minute),seconds))
        print('Evaluation accuracy of the FC model: {} +/- {}'.format(round(mean,2), round(stddev,2)))
        print('Appendix regarding desgin resulinfo: ')
        print('The optimal FC model is with {} units'.format(str(fc_final[0])))
        print('The highest training accuracy: {}%'.format(fc_final[1]))
        del fc_accu, commonfolger, fc_final, fc_designprofile, mean, stddev

    en = time.time() - wholest
    hour = en // 3600
    minute = (en - (hour * 3600)) // 60
    seconds = en - (hour * 3600) - (minute * 60)
    print(40 * '-' + 'ALLINALLRESULTS' + 40 * '-')
    print('Overall time for designing and evaluating these three models: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))
    print('Curve [dict]:')
    print( "\n".join("{}\t{}".format(k, v) for k, v in topCon.items()) )
