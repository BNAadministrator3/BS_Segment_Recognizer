import wave
import numpy as np
from scipy.fftpack import fft
from python_speech_features import mfcc

def read_wav_data(filename):
	'''
	读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
	'''
	wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
	num_frame = wav.getnframes() # 获取帧数
	num_channel=wav.getnchannels() # 获取声道数
	framerate=wav.getframerate() # 获取帧速率
	num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
	str_data = wav.readframes(num_frame) # 读取全部的帧
	wav.close() # 关闭流
	wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
	wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
	wave_data = wave_data.T # 将矩阵转置
	#wave_data = wave_data
	return wave_data, framerate

def GetFrequencyFeatures(wavsignal, fs, feature_dimension = 200,frame_length = 400, shift=160):

	length = frame_length
	nfft = int(feature_dimension*2)
	#1. forming the time-window
	x = np.linspace(0, length - 1, length, dtype=np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (length - 1))  # 汉明窗

	wav_arr = np.array(wavsignal)
	wav_length = wav_arr.shape[1]

	range0_end = (wav_length - length) // shift + 1
	data_input = np.zeros((range0_end, nfft//2), dtype=np.float)  # 用于存放最终的频率特征数据
	data_line = np.zeros((1, nfft), dtype=np.float)

	for i in range(0, range0_end):
		p_start = i * shift
		p_end = p_start + length
		data_line = wav_arr[0, p_start:p_end]
		data_line = data_line * w  # 加窗

		data_line = np.abs(fft(data_line,n=nfft)) / wav_length
		data_input[i] = data_line[0:nfft//2]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

	# print(data_input.shape)
	data_input = np.log(data_input)
	data_input = (data_input - np.min(data_input)) / np.ptp(data_input)
	# data_input = (data_input - data_input.mean()) / data_input.std()
	return data_input

def SimpleMfccFeatures(wave_data, samplerate, featurelength=26):
    temp = mfcc(wave_data[0], samplerate=samplerate, winlen=0.1, winstep=0.04, numcep=featurelength, appendEnergy=False)
    temp = temp[0:123, :]
    # return stats.zscore(temp)
    b = (temp - np.min(temp)) / np.ptp(temp)
    return b
