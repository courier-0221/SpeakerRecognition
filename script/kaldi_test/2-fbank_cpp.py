import ctypes
import numpy as np

'''
需要安装interl的mkl包
conda list | grep mkl
blas                      1.0                         mkl    defaults
mkl                       2020.2                      256    defaults
mkl-service               2.3.0            py39he8ac12f_0    defaults
mkl_fft                   1.3.0            py39h54f3939_0    defaults
mkl_random                1.0.2            py39h63df603_0    defaults
'''

# 加载动态库
lib_path = "./cpp_fbank_so/libcompute_features.so"
fbank_lib = ctypes.CDLL(lib_path)

# 配置接口参数和返回值类型
fbank_lib.compute_fbank.argtypes = [
    ctypes.c_char_p,  # wav_path
    ctypes.c_int,     # num_bins
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # output
    ctypes.POINTER(ctypes.c_int),  # rows
    ctypes.POINTER(ctypes.c_int)   # cols
]
fbank_lib.compute_fbank.restype = ctypes.c_int

fbank_lib.free_fbank_output.argtypes = [ctypes.POINTER(ctypes.c_float)]
fbank_lib.free_fbank_output.restype = None

# 函数包装
def compute_fbank(wav_path, num_bins=80):
    wav_path_c = ctypes.c_char_p(wav_path.encode('utf-8'))

    # 创建指针变量
    output_ptr = ctypes.POINTER(ctypes.c_float)()
    rows = ctypes.c_int()
    cols = ctypes.c_int()

    # 调用 C++ 接口
    result = fbank_lib.compute_fbank(wav_path_c, num_bins, ctypes.byref(output_ptr), ctypes.byref(rows), ctypes.byref(cols))
    if result != 0:
        raise RuntimeError("Fbank computation failed. Check C++ logs for more details.")

    # 转换为 numpy 数组
    rows = rows.value
    cols = cols.value
    output = np.ctypeslib.as_array(output_ptr, shape=(rows, cols))

    # 复制数据到新的 numpy 数组并释放内存
    output_copy = np.copy(output)
    fbank_lib.free_fbank_output(output_ptr)

    return output_copy

# 测试
if __name__ == "__main__":
    audio_path1 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_1.wav"
    audio_path2 = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/a_2.wav"
    features = compute_fbank(audio_path1, num_bins=80)
    print(f"Features shape: {features.shape}")
    # print(features)
    # for i in range(features.shape[0]):
    #     print(f"[{i}] {features[i].tolist()}" )

