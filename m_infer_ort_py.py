     
import torch
import os
import glob
import numpy as np
from tqdm import tqdm
import shutil
import json
from yeaudio.audio import AudioSegment
from io import BufferedReader
from mvector.data_utils.featurizer import AudioFeaturizer
from loguru import logger
import onnxruntime as ort
import pdb

'''model'''
model_path = r"/home/dear/code/VoiceprintRecognition-Pytorch/models/Res2Net_pt_onnx/inference.onnx"

'''audio'''
audio_path1 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_1.wav"
audio_path2 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_2.wav"
threshold = 0.6

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    logger.info(f"get cuda {i} error, will return cpu")
    return torch.device('cpu')

class Inference:
    def __init__(self, device, model_path):
        self.device = device
        self.session = self._load_model(model_path)
        self._audio_featurizer = AudioFeaturizer(feature_method='Fbank',
                                                 use_hf_model=False,
                                                 method_args={'sample_frequency':16000, 'num_mel_bins':80})
    
    def _load_audio(self, audio_data, sample_rate=16000):
        """加载音频
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        elif isinstance(audio_data, AudioSegment):
            audio_segment = audio_data
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        assert audio_segment.duration >= 0.3, \
            f'音频太短，最小应该为{0.3}s，当前音频为{audio_segment.duration}s'
        # 重采样
        if audio_segment.sample_rate != 16000:
            audio_segment.resample(16000)
        # decibel normalization
        # audio_segment.normalize(-20)
        return audio_segment

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

        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        input_data = torch.tensor(input_data.samples, dtype=torch.float32).unsqueeze(0)
        # print(f"====debug.input_data.shape: {input_data.shape}")
        # print(f"====debug.input_data: {input_data}")

        audio_feature = self._audio_featurizer(input_data).to(self.device).numpy()
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
