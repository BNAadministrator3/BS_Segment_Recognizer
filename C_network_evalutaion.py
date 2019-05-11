from B_network_pick import operation, logger
from A_form_trainval import trainvalFormation, Testing
from A_form_trainval import AUDIO_LENGTH, CLASS_NUM
from help_func.utilities import focal_loss, ReguBlock, ResiBlock, XcepBlock
from help_func.evaluation import Compare2

from keras import optimizers
import tensorflow as tf
from keras.models import Model
from keras.layers import *
import random
from tqdm import tqdm

import os
import shutil
import time
import sys

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


class oneInputNetwork(): #this class is highly specific
    def __init__(self):
        self.featuretype = 'spec'
        input_shape = (AUDIO_LENGTH, 200, 1)
        self.model_input = Input(shape=input_shape)
        print('to mark that the feature type is %s.' % self.featuretype.upper())
        self.__CreateRegularCNNModel__()
        self.__CreateResidualCNNModel__()
        self.__CreateInceptionModel__()

    def __CreateRegularCNNModel__(self):
        level_h1 = ReguBlock(32)(self.model_input)  # 卷积层
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = ReguBlock(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = ReguBlock(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层
        level_h4 = ReguBlock(256)(level_m3)
        level_m4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h4)  # 池化层
        flayer = GlobalAveragePooling2D()(level_m4)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc1)

        self.regularCNNmodel = Model(inputs=self.model_input, outputs=y_pred)
        print('Regular cnn model with spec feature and 8 layers are estabished.')
        self.regularCNNmodelname = 'spec_Regular_8'
        return self.regularCNNmodel, self.regularCNNmodelname

    def __CreateResidualCNNModel__(self):
        level_h1 = ResiBlock(32)(self.model_input)  # 卷积层
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = ResiBlock(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = ResiBlock(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层
        level_h4 = ResiBlock(256)(level_m3)
        level_m4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h4)  # 池化层
        flayer = GlobalAveragePooling2D()(level_m4)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc1)

        self.residualCNNmodel = Model(inputs=self.model_input, outputs=y_pred)
        print('Residual cnn model with spec feature and 8 layers are estabished.')
        self.residualCNNmodelname = 'spec_Residual_8'
        return self.residualCNNmodel, self.residualCNNmodelname

    def __CreateInceptionModel__(self):
        level_h1 = XcepBlock(32)(self.model_input)  # 卷积层
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = XcepBlock(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = XcepBlock(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层
        level_h4 = XcepBlock(256)(level_m3)
        level_m4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h4)  # 池化层
        flayer = GlobalAveragePooling2D()(level_m4)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc1)

        self.inceptionCNNmodel = Model(inputs=self.model_input, outputs=y_pred)
        print('Inception cnn model with spec feature and 8 layers are estabished.')
        self.inceptionCNNmodelname = 'spec_Inception_8'
        return self.inceptionCNNmodel, self.inceptionCNNmodelname

    def CNNmodelTrainingSetting(self,model):
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])

    def ensembleForward(self,regudir,residir,incedir):
        RegMWeightsFile = [i for i in os.listdir(regudir) if 'weights' in i]
        ResMWeightsFile = [i for i in os.listdir(residir) if 'weights' in i]
        XceMWeightsFile = [i for i in os.listdir(incedir) if 'weights' in i]

        if len(RegMWeightsFile)==1 and len(ResMWeightsFile)==1 and len(XceMWeightsFile)==1:
            self.regularCNNmodel.load_weights(os.path.join(regudir,RegMWeightsFile[0]))
            self.residualCNNmodel.load_weights(os.path.join(residir, ResMWeightsFile[0]))
            self.inceptionCNNmodel.load_weights(os.path.join(incedir,XceMWeightsFile[0]))
        else:
            print('multiple or no weights files detected in the specified directory ')
            assert(0)
        self.forwardPart = [self.regularCNNmodel, self.residualCNNmodel, self.inceptionCNNmodel]
        for subModel in self.forwardPart:
            for layer in subModel.layers:
                layer.trainable = False

    def average(self):
        outputs = [model.outputs[0] for model in self.forwardPart]
        y = Average()(outputs)
        self.averageEnsemblemodel = Model(self.model_input, y, name='ensemble_Average')
        print('Averaged-based ensemble model established.')
        return self.averageEnsemblemodel

def evaluation(dataSourceA, dataSourceB,model,modelname,featureType='spec'):
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
    from keras.backend.tensorflow_backend import set_session
    import gc
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配
    set_session(tf.Session(config=config))

    file_trans_input = os.path.join(os.getcwd(), 'dataset', 'scattered', 'standard1')
    file_trans_output = os.path.join(os.getcwd(), 'dataset', 'constructed')
    looper = trainvalFormation(file_trans_input, file_trans_output, 5, 'specified')
    looper.specifybyloading()

    basedatapath = '/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed'
    valdatapath = os.path.join(os.getcwd(),'dataset','constructed','val')
    testdatapath = os.path.join(os.getcwd(),'dataset','constructed','test')
    report1 = [valdatapath,'validation']
    report2 = [testdatapath, 'test']

    ev = time.time()
    # 1. initialize the dataset
    sys.stdout = logger()
    print()
    print(40 * '-' + 'NEWCHECKING:ROUND_1' + 40 * '-')
    print()
    looper.loop_files_transfer(1)
    nn = oneInputNetwork()
    print('this checking we do not need training..' )
    # 2. create individual and ensemble model
    regudir = os.path.join(os.getcwd(),'CNNevaluation','regular','regi=1')
    residir = os.path.join(os.getcwd(), 'CNNevaluation', 'residual', 'resi=1')
    incedir = os.path.join(os.getcwd(),'CNNevaluation','inception','inci=1')
    nn.ensembleForward(regudir=regudir,residir=residir,incedir=incedir)
    nn.average()
    # 3. evaluation
    models = [nn.regularCNNmodel,nn.residualCNNmodel,nn.inceptionCNNmodel,nn.averageEnsemblemodel]
    names  = [nn.regularCNNmodelname,nn.residualCNNmodelname,nn.inceptionCNNmodelname,'average model']
    i = 0
    for (model,name) in zip(models,names):
        print('NO.{}:'.format(str(i)))
        evaluation(report1,report2,model,name)
        i = i + 1
    # 4. for other rounds of evaluation, we need to train the models.
    for i in (0,2,3,4):
        strg = 'NEWCHECKING:ROUND_{}'.format(str(i))
        print()
        print(40 * '-' + strg + 40 * '-')
        print()
        # 4.1 generate different data
        dur = time.time()
        looper.loop_files_transfer(i)
        dur = round(time.time() - dur, 2)
        print('file transformation finished. time:{}s'.format(dur))
        # 4.2 build individual networks
        nn = oneInputNetwork()
        print('this checking we do need training individual models..')
        models = [nn.regularCNNmodel, nn.residualCNNmodel, nn.inceptionCNNmodel]
        names = [nn.regularCNNmodelname, nn.residualCNNmodelname, nn.inceptionCNNmodelname]
        paths = []
        paths.append(os.path.join( os.getcwd(),'CNNevaluation','regular','regi=' + str(i) ))
        paths.append(os.path.join(os.getcwd(), 'CNNevaluation', 'residual', 'resi=' + str(i)))
        paths.append(os.path.join(os.getcwd(), 'CNNevaluation', 'inception', 'inci=' + str(i)))
        # 4.3 train individual networks
        for (model,name,path) in zip(models,names,paths):
            nn.CNNmodelTrainingSetting(model)
            controller = operation(model, name, path)
            controller.train(basedatapath, 'spec')
            gc.collect()
        # 4.4 rebuild the individual networks and load the weights
        print(90 * '=')
        print('')
        print('!Having finish all the training ,let us now rebuild individual networks and load the corresponding weights!')
        newnn = oneInputNetwork()
        newnn.ensembleForward(regudir=paths[0],residir=paths[1],incedir=paths[2])
        newnn.average()
        models = [newnn.regularCNNmodel,newnn.residualCNNmodel,newnn.inceptionCNNmodel, newnn.averageEnsemblemodel]
        names = [newnn.regularCNNmodelname,newnn.residualCNNmodelname,newnn.inceptionCNNmodelname,'average model']
        cnt = 0
        for (model, name) in zip(models, names):
            print('NO.{}:'.format(str(cnt)))
            evaluation(report1, report2, model, name)
            cnt = cnt + 1
        gc.collect()
    en = time.time() - ev
    hour = en // 3600
    minute = (en - (hour * 3600)) // 60
    seconds = en - (hour * 3600) - (minute * 60)
    print('Overall evaluation time: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))