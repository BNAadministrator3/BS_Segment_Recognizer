#auxiliary package
from tqdm import tqdm
import random
import os
import time

#algorithm package
from keras.layers import *
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.models import Model
from keras import optimizers
from keras.initializers import Constant

#self-made package
from general_func.file_wav import GetFrequencyFeatures, read_wav_data
from debug.mfcc_trial import SimpleMfccFeatures
from release.readdata_bowel import Testing
from general_func.gen_func import Comapare2
from release.readdata_bowel import CLASS_NUM,FEATURE_TYPE
from help_func.utilities_keras import block,XcepBlock
from release.Model44_trainSetMain import ModelSpeech
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

AUDIO_STEPS = 123
SPEC_FEATURE_LENGTH = 200
MFCC_FEATURE_LENGTH = 26


class unifiedEntranceModelForward():
    def __init__(self,feature_type):
        self.classes = CLASS_NUM
        audio_legnth = AUDIO_STEPS
        self.commonPath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/network&&weights'
        if feature_type in ['spec', 'Spec', 'SPEC', 'Spectrogram', 'SPECTROGRAM']:
            self.commonPath = os.path.join(self.commonPath, 'spectrogram')
            audio_feature_legnth = SPEC_FEATURE_LENGTH
        else:
            self.commonPath = os.path.join(self.commonPath, 'mfcc')
            audio_feature_legnth = MFCC_FEATURE_LENGTH
        input_shape = (audio_legnth, audio_feature_legnth, 1)
        self.model_input = Input(shape=input_shape)
        print('This time the feature type is %s.' % feature_type.upper())
        self.ensembleForward()

    def __CreateSimplifiedIntensifiedClassicModel__(self):
        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(self.model_input)  # 卷积层
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h1)  # 卷积层
        layer_p2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层

        layer_h3 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p2)  # 卷积层
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h3)  # 卷积层
        layer_p4 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h4)  # 池化层

        layer_h5 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_p4)  # 卷积层
        layer_h6 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(layer_h5)  # 卷积层
        layer_p6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h6)  # 池化层

        flayer = Flatten()(layer_p6)
        fc2 = Dense(self.classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc2)

        model = Model(inputs=self.model_input, outputs=y_pred)
        submodel = Model(inputs=self.model_input,outputs=flayer)
        print('Simplified and intensified cnn model estabished.')
        modelname = 'cnn+dnn'
        return model, modelname, submodel

    def __CreateCustomizedResNetModel__(self):
        level_h1 = block(32)(self.model_input)
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = block(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = block(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层

        flayer = GlobalAveragePooling2D()(level_m3)
        fc = Dense(self.classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc)
        model = Model(inputs=self.model_input, outputs=y_pred)
        submodel = Model(inputs=self.model_input, outputs=flayer)
        print('Customized resnet model estabished.')
        modelname = 'residual'
        return model, modelname, submodel

    def __CreateCustomizedXceptionModel__(self):
        level_h1 = XcepBlock(32)(self.model_input)
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = XcepBlock(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        level_h3 = XcepBlock(128)(level_m2)
        level_m3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h3)  # 池化层

        flayer = GlobalAveragePooling2D()(level_m3)
        fc = Dense(self.classes, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc)
        model = Model(inputs=self.model_input, outputs=y_pred)
        submodel = Model(inputs=self.model_input, outputs=flayer)
        print('Customized Xception model estabished.')
        modelname = 'inception'
        return model,modelname,submodel

    def ensembleForward(self):
        RegularModel, RegM, SubRegularModel = self.__CreateSimplifiedIntensifiedClassicModel__()
        ResidualModel, ResM, SubResidualModel = self.__CreateCustomizedResNetModel__()
        XceptionModel, XceM, SubXceptionModel = self.__CreateCustomizedXceptionModel__()

        RegMdirectory = os.path.join(self.commonPath, RegM)
        RegMWeightsFile = [i for i in os.listdir(RegMdirectory) if 'weights' in i]
        ResMdirectory = os.path.join(self.commonPath, ResM)
        ResMWeightsFile = [i for i in os.listdir(ResMdirectory) if 'weights' in i]
        XceMdirectory = os.path.join(self.commonPath, XceM)
        XceMWeightsFile = [i for i in os.listdir(XceMdirectory) if 'weights' in i]

        if len(RegMWeightsFile)==1 and len(ResMWeightsFile)==1 and len(XceMWeightsFile)==1:
            RegularModel.load_weights(os.path.join(RegMdirectory,RegMWeightsFile[0]))
            # SubRegularModel.load_weights(os.path.join(RegMdirectory,RegMWeightsFile[0]),by_name=True)
            ResidualModel.load_weights(os.path.join(ResMdirectory, ResMWeightsFile[0]))
            # SubResidualModel.load_weights(os.path.join(ResMdirectory, ResMWeightsFile[0]),by_name=True)
            XceptionModel.load_weights(os.path.join(XceMdirectory,XceMWeightsFile[0]))
            # SubXceptionModel.load_weights(os.path.join(XceMdirectory,XceMWeightsFile[0]),by_name=True)
        else:
            print('multiple weights files detected in the specified directory {}'.format(self.commonPath))
            assert(0)
        self.forwardPart = [RegularModel, ResidualModel, XceptionModel]
        self.featureEnsemble = [SubRegularModel,SubResidualModel,SubXceptionModel] #submodel and model are the same thing.
        for subModel in self.forwardPart:
            for layer in subModel.layers:
                layer.trainable = False

numpyA = np.array([[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0]])
numpyB = np.array([[0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1]])
def multA(x):
    A = K.variable(numpyA)
    return K.dot(x,A)
def multB(x):
    B = K.variable(numpyB)
    return K.dot(x,B)
def multAB_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors and 3-model ensemble
    shape[-1] = 3
    return tuple(shape)

class unifiedEntranceEnsembleMethods():
    def __init__(self, modelinput, modelForwardPart):
        self.modelinput = modelinput
        self.modelForwardPart = modelForwardPart
    def average(self):
        outputs = [model.outputs[0] for model in self.modelForwardPart]
        y = Average()(outputs)
        model = Model(self.modelinput, y, name='ensemble_Average')
        print('Averaged-based ensemble model established.')
        return model
    def voting(self):
        pass
    def logisticReg(self):
        outputs = [model.outputs[0] for model in self.modelForwardPart]
        merge = concatenate(outputs)
        sift1 = Lambda(multA,output_shape=multAB_output_shape)(merge)
        sift2 = Lambda(multB, output_shape=multAB_output_shape)(merge)
        y1 = Dense(1,activation=None,use_bias=None,kernel_initializer=Constant(0.333))(sift1)
        y2 = Dense(1, activation=None, use_bias=None, kernel_initializer=Constant(0.333))(sift2)
        y = concatenate([y1,y2])
        model = Model(self.modelinput, y, name='ensemble_LogisticReg')
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adadelta())
        print('logisticReg-based ensemble model established.')
        return model
    def newLogisticReg(self,featureEnsemble):
        outputs = [subfeature.outputs[0] for subfeature in featureEnsemble]
        merge = concatenate(outputs)
        y = Dense(CLASS_NUM, activation='softmax',kernel_initializer='he_normal')(merge)
        model = Model(self.modelinput, y, name='ensemble_NewLogisticReg')
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adadelta())
        print('newLogisticReg-based ensemble model established.')
        return model

class Training(ModelSpeech): #note before running this class, please close the prior art. hybrid training is NOT available!!!!
    def __init__(self, model, feature_type):
        self.model = model
        assert(FEATURE_TYPE.upper() == feature_type.upper())
        self.baseSavPath = []
        strtemp = os.path.join(os.getcwd(), 'network&&weights', 'ensemble', feature_type+'_LogisticReg')
        self.baseSavPath.append(strtemp)
        strtemp = strtemp + '_weights'
        self.baseSavPath.append(strtemp)
        self.metrics = {'type': 'eval', 'sensitivity': 0, 'specificity': 0, 'score': 0, 'accuracy': 0}

class dataTestbench():
    def __init__(self,model,feature_type):
        self.model = model
        self.featureType = feature_type
        self.voting = False

    def dataTesting(self, dataSourceA, dataSourceB):
        data = newTesting(dataSourceA, dataSourceB)
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
                stuff = data.GetData((ran_num + i) % num_data, dataType=choice, feature_type=self.featureType)  # 从随机数开始连续向后取一定数量数据
                if self.featureType in ('both','Both','BOTH'):
                    dataSpec,dataMfcc,data_labels = stuff
                    data_input = [dataSpec,dataMfcc]
                else:
                    data_input,data_labels = stuff
                    data_input = data_input[np.newaxis,:]
                data_pre = self.model.predict_on_batch(data_input)
                if self.voting == False:
                    predictions = np.argmax(data_pre[0], axis=0)
                else:
                    predictions = sum([np.argmax(element[0], axis=0) for element in data_pre])
                    predictions = 1 if predictions>=2 else 0
                tp, fp, tn, fn = Comapare2(predictions, data_labels[0])  # 计算metrics
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

class newTesting(Testing):
    def __init__(self, pathSame, pathDistinct):
        super().__init__(pathSame, pathDistinct)
    def GetData(self, n_start, n_amount=32, dataType = 'same', feature_type = 'spec'):
        assert(n_amount%CLASS_NUM==0)
        if dataType == self.pathSameLabel:
            path = self.listSame[n_start][0]
            data_label = np.array([self.listSame[n_start][1]])
        elif dataType == self.pathDistinctLabel:
            path = self.listDistinct[n_start][0]
            data_label = np.array([self.listDistinct[n_start][1]])
        wavsignal, fs = read_wav_data(path)
        if feature_type in ('spec', 'Spec', 'SPEC', 'Spectrogram', 'SPECTROGRAM'):
            data_input = GetFrequencyFeatures(wavsignal, fs, SPEC_FEATURE_LENGTH, 400, shift=160)
            data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
            return data_input, data_label
        elif feature_type in ('mfcc', 'MFCC', 'Mfcc'):
            data_input = SimpleMfccFeatures(wavsignal, fs)
            data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
            return data_input, data_label
        elif feature_type in ('both','Both','BOTH'):
            data_inputSpec = GetFrequencyFeatures(wavsignal, fs, SPEC_FEATURE_LENGTH, 400, shift=160)
            data_inputMfcc = SimpleMfccFeatures(wavsignal, fs)
            data_inputSpec = data_inputSpec[np.newaxis,:,:,np.newaxis]
            data_inputMfcc = data_inputMfcc[np.newaxis,:,:,np.newaxis]
            return data_inputSpec, data_inputMfcc, data_label
        else:
            print('Unknown feature type.')
            assert (0)

ENSEMBLE_TYPE = 'SPEC_NEWLOGISTICREG' #'SPEC_AVERAGE'
test_same = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-same','same']
test_different = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-0419-different/','different']
if ENSEMBLE_TYPE == 'SPEC_AVERAGE':
    fd = unifiedEntranceModelForward(feature_type='spec')
    ensemble = unifiedEntranceEnsembleMethods(fd.model_input,fd.forwardPart)
    ensembleModel = ensemble.average()
    testing = dataTestbench(ensembleModel,feature_type='spec')
    print(90 * '*')
    print('Check the ensemble model:')
    testing.dataTesting(test_same,test_different)
# print('ensemble model summary:', ensembleModel.summary())
if ENSEMBLE_TYPE == 'ALL_AVERAGE':
    fdSpec = unifiedEntranceModelForward(feature_type='spec')
    fdMfcc = unifiedEntranceModelForward(feature_type='mfcc')
    ensemble = unifiedEntranceEnsembleMethods([fdSpec.model_input, fdMfcc.model_input], fdSpec.forwardPart+fdMfcc.forwardPart)
    ensembleModel = ensemble.average()
    # print('ensemble model summary:', ensembleModel.summary())
    testing = dataTestbench(ensembleModel, feature_type='both')
    print(90 * '*')
    print('Check the ensemble model:')
    training = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/train/', 'train']
    validation = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/validation/','validation']
    testing.dataTesting(training, validation)
if ENSEMBLE_TYPE == 'SPEC_VOTING':
    fd = unifiedEntranceModelForward(feature_type='spec')
    outputs = [model.outputs[0] for model in fd.forwardPart]
    ensembleModel = Model(fd.model_input, outputs, name='ensemble_Voting')
    testing = dataTestbench(ensembleModel, feature_type='spec')
    testing.voting = True
    print(90 * '*')
    print('Check the ensemble model:')
    testing.dataTesting(test_same, test_different)
if ENSEMBLE_TYPE == 'SPEC_LOGISTICREG':
    fd = unifiedEntranceModelForward(feature_type='spec')
    ensemble = unifiedEntranceEnsembleMethods(fd.model_input, fd.forwardPart)
    ensembleModel = ensemble.logisticReg()
    fitModel = Training(ensembleModel,feature_type='spec')
    datapath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect'
    fitModel.TrainModel(datapath, epoch=20, batch_size=32, load_weights=False)
if ENSEMBLE_TYPE == 'SPEC_NEWLOGISTICREG':
    fd = unifiedEntranceModelForward(feature_type='spec')
    ensemble = unifiedEntranceEnsembleMethods(fd.model_input, fd.forwardPart)
    ensembleModel = ensemble.newLogisticReg(fd.featureEnsemble)
    ensembleModel.summary()
    fitModel = Training(ensembleModel, feature_type='spec')
    datapath = '/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect'
    fitModel.TrainModel(datapath, epoch=20, batch_size=32, load_weights=False)