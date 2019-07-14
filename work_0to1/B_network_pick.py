from tqdm import tqdm
import os
import time
import random

import tensorflow as tf
import keras.backend as k
from keras import optimizers
from keras.layers import *
from keras.models import Model

from help_func.evaluation import Compare2,plot_confusion_matrix
from help_func.utilities import focal_loss, ReguBlock, ResiBlock, XcepBlock

from work_0to1.A_form_trainval import AUDIO_LENGTH, CLASS_NUM
from work_0to1.A_form_trainval import DataSpeech, Testing, clrdir, pathpoper,stringCheck, trainvalFormation

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

    def TestGenerability(self, feature_type, weightspath, datasourceA=None, datasourceB=None):
        training = ['/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed/train/', 'train']
        validation = ['/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed/val/','validation']
        test_same = ['/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/constructed/test/', 'same']
        # test_different = ['/home/zhaok14/example/PycharmProjects/setsail/individual_spp/bowelsounds/perfect/test-0419-different/','different']

        print(90 * '*')
        print('Firstly self-check the generability testing method:')
        self.__dataTesting__(feature_type, training, validation, weightspath)
        print('')
        print('Then derive the generability testing results: ')
        self.__dataTesting__(feature_type, test_same, validation, weightspath)

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
if __name__ == '__main__':
    from keras.backend.tensorflow_backend import set_session
    import gc
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #only display error and warning; for 1: all info; for 3: only error.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True   #不全部占满显�? 按需分配
    set_session(tf.Session(config=config))
    file_trans_input = os.path.join(os.getcwd(), 'dataset', 'scattered', 'standard1')
    file_trans_output = os.path.join(os.getcwd(), 'dataset', 'constructed')
    looper = trainvalFormation(file_trans_input, file_trans_output, 5, 'specified')
    looper.specifybyloading()
    looper.loop_files_transfer(1)
    SUMMARY = True
    if SUMMARY:
        nn = Network()
        model, name = nn.CNNForest('spec', 'Regular', 8)
        model.summary()
    else:
        sys.stdout = logger(filename=os.path.join(os.getcwd(),'Newfeature.log'))
        folders = pathpoper()
        nn = Network()
        print('This time corrected features are applied....')
        st = time.time()
        for feature_type in ('spec','mfcc'):
            for module_type in ('regular','residual','inception'):
                for layer_counts in (2,4,6,8,10):
                    print()
                    print(40 * '-'+'HANDLEFLAG'+40 * '-')
                    print()
                    feature, module, layer = stringCheck(feature_type,module_type,layer_counts)
                    path = folders.popup(feature,module,layer)
                    model,name = nn.CNNForest(feature,module,layer)
                    controller = operation(model,name,path)
                    controller.train(file_trans_output,feature)
                    gc.collect()
        en = time.time()-st
        hour = en // 3600
        minute = ( en - (hour * 3600) ) //60
        seconds = en - (hour * 3600) - (minute * 60)
        print('Overall design time: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en,int(hour),int(minute),seconds))