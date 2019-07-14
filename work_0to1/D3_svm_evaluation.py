import time
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
from sklearn import svm

from help_func.feature_transformation import read_wav_data, SimpleMfccFeatures, GetFrequencyFeatures
from work_0to1.B_network_pick import logger

def skstyleDataset(input_folder,check_num, stft_spec = False):

    assert(isinstance(check_num,int))
    feature = []
    target = []
    folders = os.listdir(input_folder)
    # place a self-check
    if len(folders) != check_num:
        print('error occurred when cheking subject numbers.')
        assert (0)
    else:
        for folder in folders:
            subf_bowel = os.path.join(input_folder,folder,'bowels')
            files = os.listdir(subf_bowel)
            if len(files) in (200,198):
                for file in files:
                    path = os.path.join(subf_bowel,file)
                    wavsignal, fs = read_wav_data(path)
                    if stft_spec == True:
                        data_input = GetFrequencyFeatures(wavsignal, fs, 200, 400, shift=160)
                    else:
                        data_input = SimpleMfccFeatures(wavsignal, fs)
                    data_input = np.reshape(data_input,[-1])
                    feature.append(data_input)
                    target.append(1)

            subf_non = os.path.join(input_folder, folder, 'non')
            files = os.listdir(subf_non)
            if len(files) in (200, 198):
                for file in files:
                    path = os.path.join(subf_non, file)
                    wavsignal, fs = read_wav_data(path)
                    if stft_spec == True:
                        data_input = GetFrequencyFeatures(wavsignal, fs, 200, 400, shift=160)
                    else:
                        data_input = SimpleMfccFeatures(wavsignal, fs)
                    data_input = np.reshape(data_input, [-1])
                    feature.append(data_input)
                    target.append(0)
            else:
                print('wrong wavfile number.')
                assert(0)

        return np.array(feature),np.array(target)
    
def perf_measure(y_truth, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_truth[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_truth[i]!=y_hat[i]:
           FP += 1
        if y_truth[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_truth[i]!=y_hat[i]:
           FN += 1

    return (TP, FP, TN, FN)

if __name__ == '__main__':
    datadir = '/home/zhaok14/example/PycharmProjects/setsail/5foldCNNdesign/dataset/scattered/standard1'
    dur = time.time()
    features, targets = skstyleDataset(datadir,20)
    print('dataset formation time: {}s.'.format(round(time.time()-dur,2)))
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    i = 0
    test_recall = []
    test_reject = []
    test_accur = []
    sys.stdout = logger(filename=os.path.join(os.getcwd(),'log&&materials','spec_svmresults.log' ))
    print('In this running the svm is all-the-same with the stft feature...')
    for train_index, test_index in skf.split(features, targets):
        strg = 'NEWCHECKING:ROUND_{}'.format(str(i))
        print()
        print(40 * '-' + strg + 40 * '-')
        print()
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        # print('y_train length: ',len(y_train))
        # print('ones counts:',y_train.tolist().count(1))
        clf = svm.SVC(kernel='rbf', C=1)
        print('Now fitting begins..')
        dur = time.time()
        clf.fit(X_train,y_train)
        print('The training time: {}s'.format( round( time.time()-dur,2 ) ))
        dur = time.time()
        train_pred = clf.predict(X_train)
        dtime = round(time.time() - dur,2)
        TP, FP, TN, FN = perf_measure(y_train, train_pred)
        recall = round(TP / (TP+FN) * 100, 2)
        specificity = round(TN / (TN+FP) * 100, 2)
        accuracy = round((TP+TN) / (TP+FP+TN+FN) *100, 2)
        print('片段类型【train】 敏感度：{0} %, 特异度： {1} %, 准确度： {2} %, 推理用时: {3}s.'.format(recall, specificity, accuracy, dtime))

        dur = time.time()
        test_pred = clf.predict(X_test)
        dtime = round(time.time() - dur, 2)
        TP, FP, TN, FN = perf_measure(y_test, test_pred)
        recall = round(TP / (TP + FN) * 100, 2)
        specificity = round(TN / (TN + FP) * 100, 2)
        accuracy = round((TP + TN) / (TP + FP + TN + FN) * 100, 2)
        print('片段类型【test】 敏感度：{0} %, 特异度： {1} %, 准确度： {2} %, 推理用时: {3}s.'.format(recall, specificity, accuracy, dtime))

        test_recall.append(recall)
        test_reject.append(specificity)
        test_accur.append(accuracy)

        i = i + 1

    for words,name in zip((test_recall,test_reject,test_accur),('recall','reject','accuracy')):
        mean = np.mean(words)
        std = np.std(words)
        print('5 fold cross validation results:')
        print('{}: {}±{}'.format( name,round(mean,4),round(std,4) ) )
        # print('official accuracy is: {}'.format(accuracy_score(y_train, train_pred)))

    # scores = cross_val_score(clf, features, targets, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

