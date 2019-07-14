import os
import sys
import time

import tensorflow as tf
from keras.layers import *

from work_0to1.A_form_trainval import trainvalFormation
from work_0to1.B_network_pick import operation, logger, Network
from work_0to1.C1_network_evaluation import evaluation, foldermanager

ALLOW_CNN = True
ALLOW_LSTM = False
ALLOW_FCN = False

if __name__ == '__main__':
    from keras.backend.tensorflow_backend import set_session
    import gc

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配
    set_session(tf.Session(config=config))
    temp_path = os.path.join(os.getcwd(), 'files_json')
    jsonfiles = []
    for i in (5, 10, 15, 20):
        strs = 'lookuptable_5folds_' + str(i) + '.json'
        jsonfiles.append(os.path.join(temp_path, strs))
    # print(jsonfiles)
    temp_path = os.path.join(os.getcwd(), 'files_h5')
    h5folders = []
    for i in (5, 10, 15, 20):
        strs = 'subjects_' + str(i)
        h5folders.append(os.path.join(temp_path, strs))
    # print(h5folders)
    file_trans_input = os.path.join(os.getcwd(), 'dataset', 'scattered', 'standard_aligned')
    file_trans_output = os.path.join(os.getcwd(), 'dataset', 'difsizes')
    valpath = os.path.join(file_trans_output, 'val')
    testpath = os.path.join(file_trans_output, 'test')
    report1 = [valpath, 'validation']
    report2 = [testpath, 'test']
    sys.stdout = logger(
        filename=os.path.join(os.getcwd(), 'log&&materials', 'std150_designevalutionresults.log'))
    # This loop is for different sizess of datasets
    topCon = {'CNN': [], 'LSTM': [], 'FC': []}
    wholest = time.time()
    for jsonfile, h5folder, flag in zip(jsonfiles, h5folders, (5, 10, 15, 20)):
        if flag != 20:
            print('We just skip the {}-scale dataset..'.format(str(flag)))
            continue
        datalooper = trainvalFormation(file_trans_input, file_trans_output, 5, 'specified')
        datalooper.specifybyloading(path=jsonfile)
        print()
        print()
        print()
        print('#' * 80)
        print('The followed data formation jsonfile:{}'.format(jsonfile.split('/')[-1]))
        print('#' * 80)
        print()
        nn = Network()
        commonfolger = foldermanager(5, h5folder)
        #################################################################################################################################
        # ALLOW_CNN = False
        if ALLOW_CNN:
            # phase 1 cnn affairs
            print('^' * 80)
            print('Firstly comes CNN..')
            print('^' * 80)
            # step1 1st fold split is always used for design
            print('This time the corrected stft features are applied....')
            print()
            print(40 * '-' + 'DESIGNSTEP:ROUND_1' + 40 * '-')
            print()
            # step2 denote the model, weight files and stock up the dataset
            datalooper.loop_files_transfer(1)
            hierachy = (('spec+regular', 'spec+residual', 'spec+inception'), (2, 4, 6, 8, 10))
            commonfolger.creation_desginfolder('cnn', hierachy)
            # step3 generate and train different networks to locate the highest one
            st = time.time()
            cnn_designprofile = []
            for module_type in ('Regular', 'Residual', 'Inception'):
                for layer_counts in (2, 4, 6, 8, 10):
                    print()
                    print(40 * '-' + 'CNNHANDLEFLAG' + 40 * '-')
                    print()
                    model, name = nn.CNNForest('SPEC', module_type, layer_counts)
                    path = os.path.join(h5folder, 'CNNdesign', 'spec+' + module_type.lower(), str(layer_counts))
                    controller = operation(model, name, path)
                    accu = controller.train(file_trans_output, 'spec')
                    cnn_designprofile.append((module_type, layer_counts, accu))
                    gc.collect()
            en = time.time() - st
            hour = en // 3600
            minute = (en - (hour * 3600)) // 60
            seconds = en - (hour * 3600) - (minute * 60)
            print(40 * '-' + 'CNNDESIGNRESULTS' + 40 * '-')
            print(
                'Overall design time for our CNN model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(hour),
                                                                                                         int(minute),
                                                                                                         seconds))
            # step4 make a decision
            cnn_accu = [t[2] for t in cnn_designprofile]
            cnn_final = cnn_designprofile[cnn_accu.index(max(cnn_accu))]  # flag
            print(
                'The optimal CNN model is with {} module and {} layers'.format(cnn_final[0].lower(), str(cnn_final[1])))
            print('The highest training accuracy: {}%'.format(cnn_final[2]))
            print()
            print('Now we start evaluating our CNN model..')
            # step5 transfer file into the specified evaluation folder
            commonfolger.creation_evaluationfolder('cnn')
            src_path = os.path.join(h5folder, 'CNNdesign', 'spec+' + cnn_final[0].lower(), str(cnn_final[1]))
            dst_path = os.path.join(h5folder, 'CNNeval', 'i=1')
            commonfolger.trnasferfiles(src_path, dst_path)

            # step6 launch 5-fold evaluation
            cnn_accu = []
            st = time.time()
            print()
            print(40 * '-' + 'EVALSTEP:ROUND_1' + 40 * '-')
            print()
            # datalooper.loop_files_transfer(1)
            print('this checking we do not need training..')
            model, name = nn.CNNForest('SPEC', cnn_final[0], cnn_final[
                1])  # its a excellent to fix cnn_final since its too vital to be changes any little.
            nn.ModelWeigthsLoading(model, dst_path)
            accu = evaluation(report1, report2, model, name)
            cnn_accu.append(accu)
            # step5.5 it is changable for the other folds
            for i in (0, 2, 3, 4):
                strg = 'EVALSTEP:ROUND_{}'.format(str(i))
                print()
                print(40 * '-' + strg + 40 * '-')
                print()
                print('this checking we do need training each model..')
                datalooper.loop_files_transfer(i)
                model, name = nn.CNNForest('SPEC', cnn_final[0], cnn_final[1])
                path = os.path.join(h5folder, 'CNNeval', 'i={}'.format(str(i)))
                controller = operation(model, name, path)
                controller.train(file_trans_output, 'spec')
                nn.ModelWeigthsLoading(model, path)
                accu = evaluation(report1, report2, model, name)
                cnn_accu.append(accu)
                gc.collect()

            mean = np.mean(cnn_accu)
            stddev = np.std(cnn_accu)
            topCon['CNN'].append((cnn_final, (mean, stddev)))
            en = time.time() - st
            hour = en // 3600
            minute = (en - (hour * 3600)) // 60
            seconds = en - (hour * 3600) - (minute * 60)
            print(40 * '-' + 'CNNEVALUATIONRESULTS' + 40 * '-')
            print('Overall evaluation time for our CNN model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(
                hour), int(minute), seconds))
            print('Evaluation accuracy of our CNN model: {} +/- {}'.format(round(mean, 2), round(stddev, 2)))
            print('Appendix regarding desgin resulinfo: ')
            print(
                'The optimal CNN model is with {} module and {} layers'.format(cnn_final[0].lower(), str(cnn_final[1])))
            print('The highest training accuracy: {}%'.format(cnn_final[2]))
            del cnn_accu, cnn_final, cnn_designprofile, mean, stddev

        #################################################################################################################################
        if ALLOW_LSTM:
            # phase 2 lstm affairs
            print('^' * 80)
            print('Secondly comes LSTM..')
            print('^' * 80)
            # step1 1st fold split is always used for design

            print()
            print(40 * '-' + 'DESIGNSTEP:ROUND_1' + 40 * '-')
            print()
            # step2 denote the model, weight files and stock up the dataset
            datalooper.loop_files_transfer(1)
            hierachy = (('mfcc', 'spec'), (64, 128, 256))
            commonfolger.creation_desginfolder('lstm', hierachy)
            # step3 generate and train different networks to locate the highest one
            for fname in ('mfcc', 'spec'):
                st = time.time()
                lstm_designprofile = []
                print('This time the {} features are applied..'.format(fname))
                for units in (64, 128, 256):
                    print()
                    print(40 * '-' + 'LSTMHANDLEFLAG' + 40 * '-')
                    print()
                    model, name = nn.LSTMbush(units, feature_type=fname.upper())
                    path = os.path.join(h5folder, 'LSTMdesign', fname, str(units))
                    controller = operation(model, name, path)
                    accu = controller.train(file_trans_output, fname)
                    lstm_designprofile.append((fname + '_' + str(units), accu))
                    gc.collect()
                en = time.time() - st
                hour = en // 3600
                minute = (en - (hour * 3600)) // 60
                seconds = en - (hour * 3600) - (minute * 60)
                print(40 * '-' + 'LSTMDESIGNRESULTS' + 40 * '-')
                print('Overall design time for the LSTM model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en, int(
                    hour), int(minute), seconds))
                # step4 make a decision
                lstm_accu = [t[1] for t in lstm_designprofile]
                lstm_final = lstm_designprofile[lstm_accu.index(max(lstm_accu))]  # flag
                print('The optimal LSTM model for {} feature is with {} info'.format(fname, str(lstm_final[0])))
                print('The highest training accuracy: {}%'.format(lstm_final[1]))
                print()
                print('Now we start evaluating the LSTM model..')
                # step5 transfer file into the specified evaluation folder
                commonfolger.creation_evaluationfolder('lstm', addlevel=fname)
                src_path = os.path.join(h5folder, 'LSTMdesign', lstm_final[0].split('_')[0],
                                        lstm_final[0].split('_')[1])
                dst_path = os.path.join(h5folder, 'LSTMeval', fname, 'i=1')
                commonfolger.trnasferfiles(src_path, dst_path)

                # step6 launch 5-fold evaluation
                lstm_accu = []
                st = time.time()
                print()
                print(40 * '-' + 'EVALSTEP:ROUND_1' + 40 * '-')
                print()
                # datalooper.loop_files_transfer(1)
                print('this checking we do NOT need training..')
                model, name = nn.LSTMbush(int(lstm_final[0].split('_')[1]), feature_type=fname.upper())
                nn.ModelWeigthsLoading(model, dst_path)
                accu = evaluation(report1, report2, model, name, featureType=fname)
                lstm_accu.append(accu)
                # step5.5 it is changable for the other folds
                for i in (0, 2, 3, 4):
                    strg = 'EVALSTEP:ROUND_{}'.format(str(i))
                    print()
                    print(40 * '-' + strg + 40 * '-')
                    print()
                    print('this checking we DO need training each model..')
                    datalooper.loop_files_transfer(i)
                    model, name = nn.LSTMbush(int(lstm_final[0].split('_')[1]), feature_type=fname.upper())
                    path = os.path.join(h5folder, 'LSTMeval', fname, 'i={}'.format(str(i)))
                    controller = operation(model, name, path)
                    controller.train(file_trans_output, fname)
                    nn.ModelWeigthsLoading(model, path)
                    accu = evaluation(report1, report2, model, name, featureType=fname)
                    lstm_accu.append(accu)
                    gc.collect()

                mean = np.mean(lstm_accu)
                stddev = np.std(lstm_accu)
                topCon['LSTM'].append((lstm_final, (mean, stddev)))
                en = time.time() - st
                hour = en // 3600
                minute = (en - (hour * 3600)) // 60
                seconds = en - (hour * 3600) - (minute * 60)
                print(40 * '-' + 'LSTMEVALUATIONRESULTS' + 40 * '-')
                print('Overall evaluation time for the LSTM model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en,
                                                                                                                    int(
                                                                                                                        hour),
                                                                                                                    int(
                                                                                                                        minute),
                                                                                                                    seconds))
                print('Evaluation accuracy of the LSTM model: {} +/- {}'.format(round(mean, 2), round(stddev, 2)))
                print('Appendix regarding desgin resulinfo: ')
                print('The optimal LSTM model for {} feature is with {} info'.format(fname, str(lstm_final[0])))
                print('The highest training accuracy: {}%'.format(lstm_final[1]))
                del lstm_accu, lstm_final, lstm_designprofile, mean, stddev

        #################################################################################################################################
        if ALLOW_FCN:
            # phase 3 fc affairs
            print('^' * 80)
            print('Thirdly comes FC..')
            print('^' * 80)
            # step1 1st fold split is always used for design
            print()
            print(40 * '-' + 'DESIGNSTEP:ROUND_1' + 40 * '-')
            print()
            # step2 denote the model, weight files and stock up the dataset
            datalooper.loop_files_transfer(1)
            hierachy = (('mfcc', 'spec'), (128, 256, 500, 1000))
            commonfolger.creation_desginfolder('fc', hierachy)
            # step3 generate and train different networks to locate the highest one
            for fname in ('mfcc', 'spec'):
                print('This time the {} features are applied..'.format(fname))
                st = time.time()
                fc_designprofile = []
                for units in (128, 256, 500, 1000):
                    print()
                    print(40 * '-' + 'FCHANDLEFLAG' + 40 * '-')
                    print()
                    model, name = nn.FCbush(units, feature_type=fname.upper())
                    path = os.path.join(h5folder, 'FCdesign', fname, str(units))
                    controller = operation(model, name, path)
                    accu = controller.train(file_trans_output, fname)
                    fc_designprofile.append((fname + '_' + str(units), accu))
                    gc.collect()
                en = time.time() - st
                hour = en // 3600
                minute = (en - (hour * 3600)) // 60
                seconds = en - (hour * 3600) - (minute * 60)
                print(40 * '-' + 'FCDESIGNRESULTS' + 40 * '-')
                print('Overall design time for the FC model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en,
                                                                                                              int(hour),
                                                                                                              int(
                                                                                                                  minute),
                                                                                                              seconds))
                # step4 make a decision
                fc_accu = [t[1] for t in fc_designprofile]
                fc_final = fc_designprofile[fc_accu.index(max(fc_accu))]  # flag
                print('The optimal FC model for {} feature is with {} info'.format(fname, str(fc_final[0])))
                print('The highest training accuracy: {}%'.format(fc_final[1]))
                print()
                print('Now we start evaluating the fc model..')
                # step5 transfer file into the specified evaluation folder
                commonfolger.creation_evaluationfolder('fc', addlevel=fname)
                src_path = os.path.join(h5folder, 'FCdesign', fc_final[0].split('_')[0], fc_final[0].split('_')[1])
                dst_path = os.path.join(h5folder, 'FCeval', fname, 'i=1')
                commonfolger.trnasferfiles(src_path, dst_path)

                # step6 launch 5-fold evaluation
                fc_accu = []
                st = time.time()
                print()
                print(40 * '-' + 'EVALSTEP:ROUND_1' + 40 * '-')
                print()
                # datalooper.loop_files_transfer(1)
                print('this checking we do NOT need training..')
                model, name = nn.FCbush(int(fc_final[0].split('_')[1]), feature_type=fname.upper())
                nn.ModelWeigthsLoading(model, dst_path)
                accu = evaluation(report1, report2, model, name, featureType=fname)
                fc_accu.append(accu)
                # step5.5 it is changable for the other folds
                for i in (0, 2, 3, 4):
                    strg = 'EVALSTEP:ROUND_{}'.format(str(i))
                    print()
                    print(40 * '-' + strg + 40 * '-')
                    print()
                    print('this checking we DO need training each model..')
                    datalooper.loop_files_transfer(i)
                    model, name = nn.FCbush(int(fc_final[0].split('_')[1]), feature_type=fname.upper())
                    path = os.path.join(h5folder, 'FCeval', fname, 'i={}'.format(str(i)))
                    controller = operation(model, name, path)
                    controller.train(file_trans_output, fname)
                    nn.ModelWeigthsLoading(model, path)
                    accu = evaluation(report1, report2, model, name, featureType=fname)
                    fc_accu.append(accu)
                    gc.collect()

                mean = np.mean(fc_accu)
                stddev = np.std(fc_accu)
                topCon['FC'].append((fc_final, (mean, stddev)))
                en = time.time() - st
                hour = en // 3600
                minute = (en - (hour * 3600)) // 60
                seconds = en - (hour * 3600) - (minute * 60)
                print(40 * '-' + 'FCEVALUATIONRESULTS' + 40 * '-')
                print('Overall evaluation time for the FC model: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(en,
                                                                                                                  int(
                                                                                                                      hour),
                                                                                                                  int(
                                                                                                                      minute),
                                                                                                                  seconds))
                print('Evaluation accuracy of the FC model: {} +/- {}'.format(round(mean, 2), round(stddev, 2)))
                print('Appendix regarding desgin resulinfo: ')
                print('The optimal FC model for {} feature is with {} info'.format(fname, str(fc_final[0])))
                print('The highest training accuracy: {}%'.format(fc_final[1]))
                del fc_accu, fc_final, fc_designprofile, mean, stddev

    en = time.time() - wholest
    hour = en // 3600
    minute = (en - (hour * 3600)) // 60
    seconds = en - (hour * 3600) - (minute * 60)
    print(40 * '-' + 'ALLINALLRESULTS' + 40 * '-')
    print(
        'Overall time for designing and evaluating these three models: {}s, i.e., {} hour(s), {} minute(s), {}s'.format(
            en, int(hour), int(minute), seconds))
    print('Curve [dict]:')
    print("\n".join("{}\t{}".format(k, v) for k, v in topCon.items()))
