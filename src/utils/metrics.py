import torch
import torchaudio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class Metrics:
    def __init__(self, n_fft, hop_length, sampling_rate, clean, pred):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.resampling_rate = 16000

        self.clean = clean
        if self.clean.is_cuda:
            self.clean = self.clean.detach().cpu()

        self.pred = pred
        if self.pred.is_cuda:
            self.pred = self.pred.detach().cpu()

        self.resampled_clean = torchaudio.transforms.Resample(self.sampling_rate, 16000)(clean)
        self.resampled_pred = torchaudio.transforms.Resample(self.sampling_rate, 16000)(pred)

    def pesq(self, mode="wb"):
        """
        PESQメトリクスの計算
        :param mode: PESQのタイプ ('nb' または 'wb')
        :return: PESQスコア
        """
        pesq_fn = PerceptualEvaluationSpeechQuality(self.resampling_rate, mode=mode)
        score = pesq_fn(self.resampled_pred, self.resampled_clean)
        return score

    def snr(self):
        """
        SNRメトリクスの計算
        :return: SNRスコア
        """
        snr_fn = SignalNoiseRatio()
        score = snr_fn(self.resampled_pred, self.resampled_clean)
        return score

    def stoi(self):
        """
        STOIメトリクスの計算
        :return: STOIスコア
        """
        stoi_fn = ShortTimeObjectiveIntelligibility(self.sampling_rate, False)
        score = stoi_fn(self.resampled_pred, self.resampled_clean)
        return score

    def si_sdr(self):
        """
        SI-SDRメトリクスの計算
        :return: SI-SDRスコア
        """
        si_sdr_fn = ScaleInvariantSignalDistortionRatio()
        score = si_sdr_fn(self.resampled_pred, self.resampled_clean)
        return score
