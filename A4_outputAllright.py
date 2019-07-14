from help_func.feature_transformation import read_wav_data,GetFrequencyFeatures
from help_func.utilities import focal_loss
from work_0to1.A_form_trainval import trainvalFormation
import os
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras import optimizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配
set_session(tf.Session(config=config))

class modelusing():
    def __init__(self,modelh5=None):
        if modelh5 != False:
            if modelh5 == None:
                mpath = os.path.join(os.getcwd(), 'CNNevaluation', 'regular', 'regi=1', 'SPEC_Regular_8_epoch13.h5')
            else:
                mpath = modelh5
            self.model = load_model(mpath, custom_objects={'FocalLoss': focal_loss, 'focal_loss_fixed': focal_loss()})
            optimizer = optimizers.Adadelta()
            self.model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])

    def loadbyFolder(self,foldername):
        filename = [os.path.join(foldername,file) for file in  os.listdir(foldername) if 'weights' not in file]
        if len(filename) == 1:
            print('Start building model..')
            self.model = load_model(filename[0], custom_objects={'FocalLoss': focal_loss, 'focal_loss_fixed': focal_loss()})
            optimizer = optimizers.Adadelta()
            self.model.compile(optimizer=optimizer, loss=[focal_loss(alpha=0.25, gamma=2)])
        else:
            print('Zero or multiple files are detected!')
            assert(0)

    def prediction(self, image, probab=False, verbose=False):
        if probab == False:
            # print( np.argmax( self.model.predict(image,verbose=verbose) ) )
            return np.argmax( self.model.predict(image,verbose=verbose) )
        else:
            # print( self.model.predict(image, verbose=verbose) )
            return self.model.predict(image, verbose=verbose)[0]



if __name__ == '__main__':
    file_trans_input = os.path.join(os.getcwd(), 'dataset', 'scattered', 'standard1')
    file_trans_output = os.path.join(os.getcwd(), 'dataset', 'constructed2')
    jsonfile = os.path.join(os.getcwd(),'work_0to1','lookuptable_5folds.json')
    datalooper = trainvalFormation(file_trans_input, file_trans_output, 5, 'specified')
    datalooper.specifybyloading(path=jsonfile)

    boweldir = os.path.join(os.getcwd(),'dataset','constructed2','test','bowels')
    nondir = os.path.join(os.getcwd(),'dataset','constructed2','test','non')

    Bwrong = {}
    Nwrong = {}
    for i in (0,1,2,3,4):
        datalooper.loop_files_transfer(i)
        bpaths = [os.path.join(boweldir,i) for i in os.listdir(boweldir)]
        npaths = [os.path.join(nondir,i) for i in os.listdir(nondir)]

        model = modelusing(modelh5=False)
        model.loadbyFolder(os.path.join( os.getcwd(),'CNNevaluation', 'regular','regi='+str(i) ))

        bwrong = []
        for bpath in bpaths:
            signal,fs = read_wav_data(bpath)
            spec = GetFrequencyFeatures(signal,fs)
            img = spec.reshape(1,spec.shape[0], spec.shape[1], 1)
            if model.prediction(img) == True:
                bwrong.append( (os.path.split(bpath)[1], model.prediction(img,probab=True)) )

        nwrong = []
        for npath in npaths:
            signal, fs = read_wav_data(npath)
            spec = GetFrequencyFeatures(signal, fs)
            img = spec.reshape(1, spec.shape[0], spec.shape[1], 1)
            if model.prediction(img) == False:
                nwrong.append( (os.path.split(npath)[1], model.prediction(img, probab=True)) )

        Bwrong[str(i)] = bwrong
        Nwrong[str(i)] = nwrong

        del bwrong, nwrong

    import pickle

    files = os.path.join(os.getcwd(), 'Bright.txt')
    with open(files, 'wb') as f:
        pickle.dump(Bwrong, f)
    files = os.path.join(os.getcwd(), 'Nright.txt')
    with open(files, 'wb') as f:
        pickle.dump(Nwrong, f)




