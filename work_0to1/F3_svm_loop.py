import os
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import tensorflow as tf

from work_0to1.A_form_trainval import trainvalFormation
from work_0to1.B_network_pick import logger
from work_0to1.D3_svm_evaluation import skstyleDataset,perf_measure

if __name__ == '__main__':
    from keras.backend.tensorflow_backend import set_session
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    file_trans_output = os.path.join(os.getcwd(),'dataset','difsizesSVM')
    valpath = os.path.join(file_trans_output, 'val')
    testpath = os.path.join(file_trans_output, 'test')
    report1 = [valpath, 'validation']
    report2 = [testpath, 'test']
    sys.stdout = logger(filename=os.path.join(os.getcwd(),'log&&materials','Supploopsvm_multipledesignevalutionresults.log'))
    # This loop is for different sizess of datasets
    topCon = {'SVM':[]}
    wholest = time.time()
    for jsonfile, h5folder, flag in zip(jsonfiles,h5folders,(5,10,15,20)):
        datalooper = trainvalFormation(file_trans_input, None, 5, 'specified')
        # jsonfile = os.path.join(os.getcwd(),'files_json', 'lookuptable_5folds_10.json')
        datalooper.specifybyloading(path=jsonfile)
        dur = time.time()
        datalooper.subjects_transfer(file_trans_output)
        features, targets = skstyleDataset(file_trans_output,flag)
        print('dataset formation time: {}s.'.format(round(time.time() - dur, 2)))
        print()
        print()
        print()
        print('#'*80)
        print('The followed data formation jsonfile:{}'.format( jsonfile.split('/')[-1] ))
        print('#'*80)
        print()
        #1. prepare the data
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        i = 0
        searchedparas = {}
        individual = []
        for train_index, test_index in skf.split(features, targets):
            strg = 'NEWCHECKING:ROUND_{}'.format(str(i))
            print()
            print(40 * '-' + strg + 40 * '-')
            print()
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            #1. for round 1, its designing stage
            if i == 0:
                print('This round we DO need to grid search the model parameters..')
                tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
                clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy')
                st=time.time()
                clf.fit(X_train, y_train)
                print('Search and refitting time (round_{}): '.format(str(i)),'s.'.format(round(time.time()-st,2)))
                print("Best parameters set found on development set:")
                print(clf.best_params_)
                searchedparas = clf.best_params_
                print("Grid scores on development set:")
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
            else:
                print('This round we do NOT need to perfrom grid search..')
                assert(searchedparas)
                if searchedparas['kernel'] == 'rbf':
                    clf = svm.SVC(kernel='rbf', C=searchedparas['C'],gamma=searchedparas['gamma'])
                else:
                    clf = svm.SVC(kernel='linear', C=searchedparas['C'])
                st = time.time()
                clf.fit(X_train, y_train)
                print('Training time (round_{}): '.format(str(i)), 's.'.format(round(time.time() - st, 2)))

            print('^' * 80)
            print('Now start checking..')
            dur = time.time()
            train_pred = clf.predict(X_train)
            dtime = round(time.time() - dur, 2)
            TP, FP, TN, FN = perf_measure(y_train, train_pred)
            recall = round(TP / (TP + FN) * 100, 2)
            specificity = round(TN / (TN + FP) * 100, 2)
            accuracy = round((TP + TN) / (TP + FP + TN + FN) * 100, 2)
            print('片段类型【train】 敏感度：{0} %, 特异度： {1} %, 准确度： {2} %, 推理用时: {3}s.'.format(recall, specificity, accuracy, dtime))

            dur = time.time()
            test_pred = clf.predict(X_test)
            dtime = round(time.time() - dur, 2)
            TP, FP, TN, FN = perf_measure(y_test, test_pred)
            recall = round(TP / (TP + FN) * 100, 2)
            specificity = round(TN / (TN + FP) * 100, 2)
            accuracy = round((TP + TN) / (TP + FP + TN + FN) * 100, 2)
            print('片段类型【test】 敏感度：{0} %, 特异度： {1} %, 准确度： {2} %, 推理用时: {3}s.'.format(recall, specificity, accuracy, dtime))
            individual.append(accuracy)

            i = i+1

        mean = np.mean(individual)
        stddev = np.std(individual)
        print('For json file {}, the evaluation results are: {} +/- {} %.'.format( jsonfile.split('/')[-1], round(mean,2), round(stddev,2) ) )
        topCon['SVM'].append( (searchedparas, mean, stddev) )

    en = time.time() - wholest
    hour = en // 3600
    minute = (en - (hour * 3600)) // 60
    seconds = en - (hour * 3600) - (minute * 60)
    print()
    print()
    print(40 * '-' + 'ALLINALLRESULTS' + 40 * '-')
    print('Overall time for designing and evaluating these three models: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour), int(minute), seconds))
    print('Curve [dict]:')
    print("\n".join("{}\t{}".format(k, v) for k, v in topCon.items()))

