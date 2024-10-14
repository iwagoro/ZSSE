import sys

sys.path.append("ZS-N2N/src")
from data.dataset import SpeechDataset
from utils.stfts import stft, istft

dataset = SpeechDataset(
    noise_file="",
    clean_file="/Users/rockwell/Documents/python/ZS-N2N/data/source/clean/p226_001.wav",
    noise_type="white",
    snr_level=0,
)

# イテレータを生成して、データを取得
iterator = iter(dataset)
data = next(iterator)  # イテレータから最初の要素を取得

stft = stft(data[0], n_fft=1022, hop_length=256)
