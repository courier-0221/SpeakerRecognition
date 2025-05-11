import torchaudio.compliance.kaldi as Kaldi
import torchaudio

audio_path1 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_1.wav"
audio_path2 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_2.wav"



def main():
    waveform, sample_rate = torchaudio.load(audio_path1)

    # 提取 fbank 特征，设置 dither 为 0.0
    fbank_features = Kaldi.fbank(
        waveform,
        num_mel_bins=80,      # 设置 Mel 滤波器数量
        frame_length=25,      # 帧长 (ms)
        frame_shift=10,       # 帧移 (ms)
        sample_frequency=sample_rate,  # 音频采样率
    )

    # 打印特征形状
    print(fbank_features.shape)
    for i in range(fbank_features.shape[0]):
                print(f"[{i}] {fbank_features[i].tolist()}" )

if __name__ == "__main__":
    main()