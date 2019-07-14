from A_export2snoop import modelusing
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配

from help_func.feature_transformation import read_wav_data,GetFrequencyFeatures
data_dir = os.path.join(os.getcwd(),'dataset','supplementary','1035711_coarse')
data_path = [ os.path.join(data_dir,i) for i in os.listdir(data_dir) ]
results = []
model = modelusing()
for path in data_path:
    signal, fs = read_wav_data(path)
    spec = GetFrequencyFeatures(signal, fs)
    img = spec.reshape(1, spec.shape[0], spec.shape[1], 1)
    if model.prediction(img) == True and model.prediction(img,probab=True)[1]>0.9:
        results.append( (os.path.split(path)[1], model.prediction(img,probab=True)) )
print(results)
# import pickle
# files = os.path.join(os.getcwd(), '1035711intense.txt')
# with open(files, 'wb') as f:
#    pickle.dump(results, f)