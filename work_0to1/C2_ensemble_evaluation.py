from work_0to1.B_network_pick import logger
from work_0to1.A_form_trainval import trainvalFormation
from C1_network_evalutaion import oneInputNetwork, evaluation

import tensorflow as tf
from keras.models import Model
from keras.layers import *

import os
import time
import sys

class flexible_average(oneInputNetwork): #this class is highly specific
    def __init__(self):
        super(flexible_average,self).__init__()

    def CNNmodelTrainingSetting(self,model):
        assert(0)

    def average(self, regudir=False, residir=False, incedir=False):
        assert( isinstance(regudir,bool) and isinstance(residir,bool) and isinstance(incedir,bool) ) #flag_3
        if not regudir and not residir and not incedir:
            assert(0)
        else:
            blist = [regudir, residir, incedir]
            self.Mslice = [ self.forwardPart[i] for i in (0,1,2) if blist[i]==True ]
            if len(self.Mslice) == 1:
                print('[*ERROR]Its not an ensemble model.')
                assert(0)
            else:
                order = ['reg', 'res', 'inc']
                suborder = [order[i] for i in (0,1,2) if blist[i]==True]
                name = 'A'.join(suborder)+'_aveEnsemble' #flag_2
                cnt = len(self.Mslice)
                outputs = [model.outputs[0] for model in self.Mslice]
                y = Average()(outputs)
                averageEnsemblemodel = Model(self.model_input, y, name=name)
                print("{}-averaged-based ensemble model called '{}' are established.".format( str(cnt),name ))
                return averageEnsemblemodel,name

    def voting(self):
        outputs = [model.outputs[0] for model in self.forwardPart]
        votingEnsemblemodel = Model(self.model_input, outputs, name='ensemble_Voting')
        print('Voting-based ensemble model established.')
        return votingEnsemblemodel, 'votEnsemble'



if __name__ == '__main__':
    from keras.backend.tensorflow_backend import set_session

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配
    set_session(tf.Session(config=config))

    #define the file transfer object
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
    sys.stdout = logger(filename=os.path.join(os.getcwd(),'log&&materials','Newfeature_multipleensemble_evaluationresults.log'))
    print('This time the corrected features are applied....')
    for i in (0, 1, 2, 3, 4):
        print()
        print(40 * '-' + 'NEWCHECKING:ROUND_{}'.format(str(i)) + 40 * '-')
        print()
        looper.loop_files_transfer(i)
        nn = flexible_average()
        # 2. create individual and ensemble model
        regudir = os.path.join(os.getcwd(),'CNNevaluation','regular','regi='+ str(i))
        residir = os.path.join(os.getcwd(), 'CNNevaluation', 'residual','resi='+ str(i))
        incedir = os.path.join(os.getcwd(),'CNNevaluation','inception','inci='+ str(i))
        nn.ensembleForward(regudir=regudir, residir=residir, incedir=incedir)
        regAresModel, regAresName = nn.average(regudir=True,residir=True)
        regAincModel, regAincName = nn.average(regudir=True,incedir=True)
        resAincModel, resAincName = nn.average(residir=True,incedir=True)
        wholeModel, wholeName = nn.average(regudir=True,residir=True,incedir=True)
        voteModel, voteName = nn.voting()
        # 3. evaluation
        models = [regAresModel,regAincModel,resAincModel,wholeModel,voteModel]
        names  = [regAresName,regAincName,resAincName,wholeName,voteName]
        j = 0
        for (model,name) in zip(models,names):
            print('NO.{}:'.format(str(j)))
            if name == 'votEnsemble':
                evaluation(report1, report2, model, name, votingchecking=True)
            else:
                evaluation(report1,report2,model,name)
            j = j + 1
    en = time.time() - ev
    hour = en // 3600
    minute = (en - (hour * 3600)) // 60
    seconds = en - (hour * 3600) - (minute * 60)
    print('Overall evaluation time for these five ensemble models: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))