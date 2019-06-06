import os
import shutil
import time
import sys

from A_form_trainval import AUDIO_LENGTH, CLASS_NUM
from A_form_trainval import trainvalFormation
from B_network_pick import operation, logger
from C1_network_evalutaion import evaluation
from help_func.utilities import focal_loss
#neural network package
from keras import optimizers
import tensorflow as tf
from keras.models import Model
from keras.layers import *



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
        self.base = os.path.join(self.parentdir, 'DNNevaluation','stft')

    def creation(self):
        for i in range(self.fold_number):
            strs = 'i={}'.format(i)
            regi = os.path.join(self.base,strs)
            os.makedirs(regi)

    def deletion(self):
        if os.path.exists(self.base):
            shutil.rmtree(self.base)

# a=folderoperation(5)
# a.creation()

class comparativeNetwork():
    def __init__(self):
        self.featuretype = 'spec'
        input_shape = (AUDIO_LENGTH, 200, 1)
        self.model_input = Input(shape=input_shape)
        print('to mark that the feature type is genuinely %s.' % self.featuretype.upper())

    def CreateDnnModel(self):
        x = Flatten(name='squeeze')(self.model_input)
        y1 = Dense(1000,kernel_regularizer=regularizers.l2(0.5),activation='relu')(x)  # computation complexity
        y1 = Dropout(0.9)(y1)
        y2 = Dense(1000,kernel_regularizer=regularizers.l2(0.5),activation='relu')(y1)
        y2 = Dropout(0.9)(y2)
        y3 = Dense(1000,kernel_regularizer=regularizers.l2(0.5),activation='relu')(y2)
        y3 = Dropout(0.9)(y3)
        y_pred = Dense(CLASS_NUM, activation='softmax')(y3)
        self.dnnmodel = Model(inputs=self.model_input, outputs=y_pred)
        self.dnnmodelname = self.featuretype+'_dnn_1000'
        print('The dnn model with {} featue and [1000 1000 1000] layers is established.'.format(self.featuretype))
        # print(self.dnnmodel.summary())
        return self.dnnmodel,self.dnnmodelname

    def ModelTrainingSetting(self,model):
        optimizer = optimizers.Adadelta()
        #model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        model.compile(optimizer=optimizer,loss='binary_crossentropy')

    def ModelweightsLoading(self,model,dnndir):
        WeightsFile = [i for i in os.listdir(dnndir) if 'weights' in i]
        if len(WeightsFile)==1:
            model.load_weights(os.path.join(dnndir,WeightsFile[0]))
        else:
            print('multiple or no weights files detected in the specified directory ')
            assert(0)

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
    sys.stdout = logger(filename=os.path.join(os.getcwd(),'log&&materials','spec_fcresults.log' ))
    # 2. for every single rounds of evaluation, we need to train the models.
    for i in (0,1,2,3,4):
        strg = 'NEWCHECKING:ROUND_{}'.format(str(i))
        print()
        print(40 * '-' + strg + 40 * '-')
        print()
        # 2.1 generate different data
        dur = time.time()
        looper.loop_files_transfer(i)
        dur = round(time.time() - dur, 2)
        print('file transformation finished. time:{}s'.format(dur))
        # 2.2 build individual networks
        nn = comparativeNetwork()
        nn.CreateDnnModel()
        nn.ModelTrainingSetting(nn.dnnmodel)
        print('this checking we do need training the fc model..')
        path = os.path.join( os.getcwd(),'DNNevaluation','stft','i=' + str(i) )
        # 4.3 train individual networks
        controller = operation(nn.dnnmodel, nn.dnnmodelname, path)
        controller.train(basedatapath, 'spec')
        gc.collect()
        # 4.4 rebuild the individual networks and load the weights
        print(90 * '=')
        print('')
        print('!Having finish the training ,let us now rebuild the fc network and load the corresponding weights!')
        newnn = comparativeNetwork()
        newnn.CreateDnnModel()
        newnn.ModelweightsLoading(newnn.dnnmodel,path)
        print('for round_{}:'.format(str(i)))
        evaluation(report1, report2, newnn.dnnmodel, newnn.dnnmodelname,featureType='spec')
        gc.collect()
    en = time.time() - ev
    hour = en // 3600
    minute = (en - (hour * 3600)) // 60
    seconds = en - (hour * 3600) - (minute * 60)
    print('Overall evaluation time: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))
