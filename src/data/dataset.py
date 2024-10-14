import torch
import torchaudio
from torch.utils.data import Dataset


class SpeechDataset(Dataset):

    def __init__(self, noise_file, clean_file, noise_type, snr_level):
        super().__init__()
        self.noise_file_path = noise_file
        self.clean_file_path = clean_file
        self.snr_level = snr_level

        self.noise_type = noise_type
        self.max_len = 65280

        # データセットの長さを1に設定
        self.length = 1

    def __len__(self):
        return self.length

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def __getitem__(self, index):
        # load to tensors and normalization
        clean = self.load_sample(self.clean_file_path)
        noise = None
        if self.noise_type != "white":
            noise = self.load_sample(self.noise_file_path)

        # padding/cutting
        clean = self._prepare_sample(clean)
        noisy = None
        if self.noise_type == "white":
            noisy = self.add_white_noise(clean, self.snr_level)
        else:
            noisy = self.add_noise(clean, noise, self.snr_level)

        return noisy, clean

    def _prepare_sample(self, waveform):
        # Assume waveform is of shape (channels, time)
        channels, current_len = waveform.shape

        # Initialize output tensor with zeros
        output = torch.zeros((channels, self.max_len), dtype=torch.float32, device=waveform.device)

        # Copy the necessary part of the data
        output[:, -min(current_len, self.max_len) :] = waveform[:, : min(current_len, self.max_len)]

        return output

    def add_white_noise(self, clean_waveform, snr):
        """指定したSNRでクリーン音声にホワイトノイズを追加します。"""
        # Calculate power of the clean signal
        clean_power = torch.mean(clean_waveform**2)

        # Calculate the noise power required to achieve the desired SNR
        noise_power = clean_power / (10 ** (snr / 10))

        # Generate white noise with the calculated power
        white_noise = torch.randn_like(clean_waveform) * torch.sqrt(noise_power)

        # Add white noise to the clean signal
        noisy_waveform = clean_waveform + white_noise

        return self._prepare_sample(noisy_waveform)

    def add_noise(self, clean_waveform, noise_waveform, snr):
        """指定したSNRでクリーン音声に雑音音声を追加します。"""
        # クリーン音声の長さに合わせてノイズ音声を調整
        clean_len = clean_waveform.shape[1]
        noise_len = noise_waveform.shape[1]

        if noise_len < clean_len:
            # ノイズがクリーン音声より短い場合、繰り返してノイズを長くする
            repeat_factor = (clean_len // noise_len) + 1
            noise_waveform = noise_waveform.repeat(1, repeat_factor)[:, :clean_len]
        else:
            # ノイズがクリーン音声より長い場合、切り取る
            noise_waveform = noise_waveform[:, :clean_len]

        # クリーン音声とノイズ音声のパワーを計算
        clean_power = torch.mean(clean_waveform**2)
        noise_power = torch.mean(noise_waveform**2)

        # SNRを達成するためのスケーリング係数を計算
        scaling_factor = torch.sqrt(clean_power / (noise_power * 10 ** (snr / 10)))

        # ノイズをスケーリングしてクリーン信号に加える
        scaled_noise = noise_waveform * scaling_factor
        noisy_waveform = clean_waveform + scaled_noise

        return self._prepare_sample(noisy_waveform)
