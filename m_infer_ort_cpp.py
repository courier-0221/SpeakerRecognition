     
import os
import glob
import numpy as np
from tqdm import tqdm
import shutil
import json
from io import BufferedReader
from loguru import logger
import onnxruntime as ort
import ctypes
import pdb

'''model'''
model_path = r"/home/dear/code/VoiceprintRecognition-Pytorch/models/Res2Net_pt_onnx/inference.onnx"

'''audio'''
audio_path1 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_1.wav"
audio_path2 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_2.wav"
threshold = 0.6

'''lib'''
lib_path = "./script/kaldi_test/cpp_fbank_so/libcompute_features.so"


class KaldiFbankFeat:
    def __init__(self, cpp_lib_path):
        self.fbank_lib = ctypes.CDLL(cpp_lib_path)
        # 配置接口参数和返回值类型
        self.fbank_lib.compute_fbank.argtypes = [
            ctypes.c_char_p,  # wav_path
            ctypes.c_int,     # num_bins
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # output
            ctypes.POINTER(ctypes.c_int),  # rows
            ctypes.POINTER(ctypes.c_int)   # cols
        ]
        self.fbank_lib.compute_fbank.restype = ctypes.c_int

        self.fbank_lib.free_fbank_output.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.fbank_lib.free_fbank_output.restype = None

    def compute_fbank(self, wav_path, num_bins=80):
        wav_path_c = ctypes.c_char_p(wav_path.encode('utf-8'))

        # 创建指针变量
        output_ptr = ctypes.POINTER(ctypes.c_float)()
        rows = ctypes.c_int()
        cols = ctypes.c_int()

        # 调用 C++ 接口
        result = self.fbank_lib.compute_fbank(wav_path_c, num_bins, ctypes.byref(output_ptr), ctypes.byref(rows), ctypes.byref(cols))
        if result != 0:
            raise RuntimeError("Fbank computation failed. Check C++ logs for more details.")

        # 转换为 numpy 数组
        rows = rows.value
        cols = cols.value
        output = np.ctypeslib.as_array(output_ptr, shape=(rows, cols))

        # 复制数据到新的 numpy 数组并释放内存
        output_copy = np.copy(output)
        self.fbank_lib.free_fbank_output(output_ptr)

        return output_copy


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    logger.info(f"get cuda {i} error, will return cpu")
    return torch.device('cpu')


class Inference:
    def __init__(self, device, model_path):
        self.device = device
        self.session = self._load_model(model_path)
        self._audio_featurizer = KaldiFbankFeat(lib_path)

    def _load_model(self, model_path):
        session = ort.InferenceSession(model_path)
        logger.info(f'path={model_path}, loaded model...')
        return session
    
    def predict(self, audio_data, sample_rate=16000):
        """预测一个音频的特征

        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 声纹特征向量
        """

        print(f"audio_data: {audio_data}")
        audio_feature = self._audio_featurizer.compute_fbank(audio_data)
        # 添加一个维度
        audio_feature = np.expand_dims(audio_feature, axis=0)
        print(f"====debug.audio_feature.shape: {audio_feature.shape}")
        # for i in range(audio_feature.shape[1]):
        #     print(f"[{i}] {np.array_str(audio_feature[0, i], precision=5, suppress_small=True)}" )
        
        input_names = [input.name for input in self.session.get_inputs()]
        output_names = [output.name for output in self.session.get_outputs()]
        print("Inputs:", input_names)
        print("Outputs:", output_names)

        inputs = {input_names[0]: audio_feature}
        outputs = self.session.run(output_names, inputs)
        
        # print(f"====debug.outputs type: {type(outputs)}")
        # print(f"====debug.outputs[0] type: {type(outputs[0])}")
        # print(f"====debug.outputs[0].shape: {outputs[0].shape}")
        # print(f"====debug.outputs[0].reshape(-1).shape: {outputs[0].reshape(-1).shape}")
        # print(f"====debug.outputs[0]: {outputs[0]}")
        
        feature = outputs[0].reshape(-1)
        return feature

    def contrast(self, audio_data1, audio_data2):
        """声纹对比

        param audio_data1: 需要对比的音频1，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整的字节文件
        param audio_data2: 需要对比的音频2，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整的字节文件

        return: 两个音频的相似度
        """
        feature1 = self.predict(audio_data1)
        feature2 = self.predict(audio_data2)
        # 对角余弦值
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return dist


def main():
    # 获取识别器
    infer = Inference('cpu', model_path)
    dist = infer.contrast(audio_path1, audio_path2)
    if dist > threshold:
        print(f"{audio_path1} 和 {audio_path2} 为同一个人，相似度为：{dist}")
    else:
        print(f"{audio_path1} 和 {audio_path2} 不是同一个人，相似度为：{dist}")

if __name__ == "__main__":
    main()
