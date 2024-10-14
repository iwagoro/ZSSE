import sys

sys.path.append("ZS-N2N/src")
import torch
import torch.nn as nn
import lightning as l
from utils.subsample import subsample
from utils.metrics import Metrics
from utils.stfts import stft, istft
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

# from .Encoder import Encoder
# from .Decoder import Decoder


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, last_layer=False):
        super(Decoder, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.last_layer = last_layer

    def forward(self, x):
        x = self.conv_transpose(x)
        if not self.last_layer:
            x = self.bn(x)
            x = self.relu(x)
        return x


class ZSN2N(l.LightningModule):
    def __init__(self, config):
        super(ZSN2N, self).__init__()
        self.save_hyperparameters()  # ハイパーパラメータを保存

        # パラメータ設定の読み込み
        self.n_fft = config.data.n_fft
        self.hop_length = config.data.hop_length
        self.sampling_rate = config.data.sample_rate

        self.input_channels = config.model.input_channels
        self.embed_dim = config.model.embed_dim

        self.loss = config.training.loss.type
        self.optimizer_type = config.training.optimizer.type
        self.optimizer_params = config.training.optimizer.params
        self.scheduler_type = None
        self.scheduler_params = None
        if config.training.get("scheduler") is not None:
            self.scheduler_type = config.training.scheduler.type
            self.scheduler_params = config.training.scheduler.params

            # モデルの定義
        #         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #         self.conv1 = nn.Conv1d(self.input_channels, self.embed_dim, kernel_size=3, padding=1)
        #         self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)
        #         self.conv2_2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=5, padding=2)
        #         self.conv3 = nn.Conv1d(self.embed_dim, self.input_channels, kernel_size=3, padding=1)

        # def forward(self, x):
        #     x = self.act(self.conv1(x))
        #     x = self.act(self.conv2(x))
        #     x = self.act(self.conv2_2(x))
        #     x = self.conv3(x)
        #     return x

        # def training_step(self, batch, batch_idx):
        #     x, _ = batch

        #     g1, g2 = subsample(x)
        #     pred1 = g1 - self.forward(g1)
        #     pred2 = g2 - self.forward(g2)

        #     loss_res = 0.5 * (torch.nn.MSELoss()(g1, pred2) + torch.nn.MSELoss()(g2, pred1))

        #     denoised = x - self.forward(x)
        #     dg1, dg2 = subsample(denoised)

        #     loss_cons = 0.5 * (torch.nn.MSELoss()(pred1, dg1) + torch.nn.MSELoss()(pred2, dg2))

        #     loss = loss_res + loss_cons

        #     # メトリクスのログ
        #     self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        #     return loss

        #     # downsampling/encoding
        #     self.downsample0 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=1, out_channels=45)
        #     self.downsample1 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=45, out_channels=90)
        #     self.downsample2 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        #     self.downsample3 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        #     self.downsample4 = Encoder(filter_size=(3, 3), stride_size=(2, 1), in_channels=90, out_channels=90)

        #     # upsampling/decoding
        #     self.upsample0 = Decoder(filter_size=(3, 3), stride_size=(2, 1), in_channels=90, out_channels=90)
        #     self.upsample1 = Decoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        #     self.upsample2 = Decoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        #     self.upsample3 = Decoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=180, out_channels=45)
        #     self.upsample4 = Decoder(
        #         filter_size=(3, 3),
        #         stride_size=(2, 2),
        #         in_channels=90,
        #         output_padding=(1, 1),
        #         out_channels=1,
        #         last_layer=True,
        #     )

        # def forward(self, x):
        #     if isinstance(x, list):
        #         x = torch.stack(x)
        #     # downsampling/encoding
        #     d0 = self.downsample0(x)
        #     # print("d0 : " + str(d0.shape))
        #     d1 = self.downsample1(d0)
        #     # print("d1 : " + str(d1.shape))
        #     d2 = self.downsample2(d1)
        #     # print("d2 : " + str(d2.shape))
        #     d3 = self.downsample3(d2)
        #     # print("d3 : " + str(d3.shape))
        #     d4 = self.downsample4(d3)
        #     # print("d4 : " + str(d4.shape))

        #     # upsampling/decoding
        #     u0 = self.upsample0(d4)
        #     # print("u0 : " + str(u0.shape))
        #     # skip-connection
        #     c0 = torch.cat((u0, d3), dim=1)
        #     # print("c0 : " + str(c0.shape))

        #     u1 = self.upsample1(c0)
        #     # print("u1 : " + str(u1.shape))
        #     c1 = torch.cat((u1, d2), dim=1)
        #     # print("c1 : " + str(c1.shape))

        #     u2 = self.upsample2(c1)
        #     # print("u2 : " + str(u2.shape))
        #     c2 = torch.cat((u2, d1), dim=1)
        #     # print("c2 : " + str(c2.shape))

        #     u3 = self.upsample3(c2)
        #     # print("u3 : " + str(u3.shape))
        #     c3 = torch.cat((u3, d0), dim=1)
        #     # print("c3 : " + str(c3.shape))

        #     u4 = self.upsample4(c3)
        #     # print("u4 : " + str(u4.shape))

        #     # u4 - the mask
        #     output = u4 * x
        #     # print("output : " + str(output.shape))

        #     return output

        # def training_step(self, batch, batch_idx):
        #     x, _ = batch

        #     g1, g2 = subsample(x)
        #     g1 = stft(g1, n_fft=self.n_fft, hop_length=self.hop_length)
        #     g2 = stft(g2, n_fft=self.n_fft, hop_length=self.hop_length)
        #     pred1 = g1 - self.forward(g1)
        #     pred2 = g2 - self.forward(g2)

        #     loss_res = 0.5 * (torch.nn.MSELoss()(g1, pred2) + torch.nn.MSELoss()(g2, pred1))

        #     x_stft = stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        #     denoised = x_stft - self.forward(x_stft)
        #     denoised = istft(denoised, n_fft=self.n_fft, hop_length=self.hop_length)
        #     dg1, dg2 = subsample(denoised)

        #     dg1 = stft(dg1, n_fft=self.n_fft, hop_length=self.hop_length)
        #     dg2 = stft(dg2, n_fft=self.n_fft, hop_length=self.hop_length)

        #     loss_cons = 0.5 * (torch.nn.MSELoss()(pred1, dg1) + torch.nn.MSELoss()(pred2, dg2))

        #     loss = loss_res + loss_cons

        #     # メトリクスのログ
        #     self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        #     return loss

        self.downsample0 = Encoder(in_channels=1, out_channels=16, kernel_size=15, stride=2)
        self.downsample1 = Encoder(in_channels=16, out_channels=32, kernel_size=15, stride=2)
        self.downsample2 = Encoder(in_channels=32, out_channels=64, kernel_size=15, stride=2)
        self.downsample3 = Encoder(in_channels=64, out_channels=128, kernel_size=15, stride=2)
        self.downsample4 = Encoder(in_channels=128, out_channels=256, kernel_size=15, stride=2)
        # Decoding (Upsampling)
        self.upsample0 = Decoder(in_channels=256, out_channels=128, kernel_size=15, stride=2, output_padding=1)
        self.upsample1 = Decoder(in_channels=256, out_channels=64, kernel_size=15, stride=2, output_padding=1)
        self.upsample2 = Decoder(in_channels=128, out_channels=32, kernel_size=15, stride=2, output_padding=1)
        self.upsample3 = Decoder(in_channels=64, out_channels=16, kernel_size=15, stride=2, output_padding=1)
        self.upsample4 = Decoder(
            in_channels=32, out_channels=1, kernel_size=15, stride=2, output_padding=1, last_layer=True
        )

    def forward(self, x):
        # Encoding path
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        # Decoding path with skip connections
        u0 = self.upsample0(d4)
        c0 = torch.cat((u0, d3), dim=1)
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        u4 = self.upsample4(c3)

        return u4

    def training_step(self, batch, batch_idx):
        noisy, _ = batch
        g1, g2 = subsample(noisy)

        pred1 = g1 - self.forward(g1)
        pred2 = g2 - self.forward(g2)

        loss_res = 0.5 * (torch.nn.MSELoss()(g1, pred2) + torch.nn.MSELoss()(g2, pred1))

        denoised = noisy - self.forward(noisy)
        dg1, dg2 = subsample(denoised)

        loss_cons = 0.5 * (torch.nn.MSELoss()(pred1, dg1) + torch.nn.MSELoss()(pred2, dg2))

        loss = loss_res + loss_cons

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def predict_step(self, batch, batch_idx):

        noisy, clean = batch
        pred = noisy - self.forward(noisy)
        # メトリクスの計算
        metrics = Metrics(self.n_fft, self.hop_length, self.sampling_rate, clean, pred)

        pesq_wb = metrics.pesq(mode="wb")
        pesq_nb = metrics.pesq(mode="nb")
        snr = metrics.snr()
        stoi = metrics.stoi()
        si_sdr = metrics.si_sdr()
        # ログの記録
        if self.logger is not None:

            writer = self.logger.experiment  # SummaryWriterオブジェクト
            current_step = self.global_step
            # メトリクスをテキストとしてログ
            metrics_text = f"""
    
            pesq_wb: {pesq_wb}
            pesq_nb: {pesq_nb}
            snr: {snr}
            stoi: {stoi}
            si_sdr: {si_sdr}
            """
            writer.add_text("Metrics", metrics_text, current_step)
            # オーディオファイルをTensorBoardにログ
            writer.add_audio("audio/noisy", noisy[0], current_step, sample_rate=self.sampling_rate)

            writer.add_audio("audio/clean", clean[0], current_step, sample_rate=self.sampling_rate)
            writer.add_audio("audio/pred", pred[0], current_step, sample_rate=self.sampling_rate)
            # STFTを計算
            noisy_stft = stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length)

            pred_stft = stft(pred, n_fft=self.n_fft, hop_length=self.hop_length)
            clean_stft = stft(clean, n_fft=self.n_fft, hop_length=self.hop_length)
            # 振幅スペクトログラムを計算
            # 振幅 = sqrt(real^2 + imag^2)

            noisy_mag = torch.sqrt(
                noisy_stft[..., 0] ** 2 + noisy_stft[..., 1] ** 2
            ).cpu()  # [batch, channel, freq, time]
            pred_mag = torch.sqrt(pred_stft[..., 0] ** 2 + pred_stft[..., 1] ** 2).cpu()
            clean_mag = torch.sqrt(clean_stft[..., 0] ** 2 + clean_stft[..., 1] ** 2).cpu()
            # 振幅スペクトログラムをデシベル単位に変換（0除算防止のための1e-6を加算）
            noisy_spectrogram = 20 * torch.log10(noisy_mag + 1e-6)

            pred_spectrogram = 20 * torch.log10(pred_mag + 1e-6)
            clean_spectrogram = 20 * torch.log10(clean_mag + 1e-6)

            # スペクトログラムをRGB画像に変換する関数
            def spectrogram_to_rgb(spectrogram_db):

                # バッチ次元やチャネル次元を削除
                spectrogram_db = spectrogram_db.squeeze()  # [freq, time]
                # データが2次元（周波数, 時間）であることを確認
                if spectrogram_db.ndim != 2:
                    raise ValueError(f"spectrogram_db should be 2D, but got shape {spectrogram_db.shape}")
                spectrogram_db = spectrogram_db.numpy()
                # 値を [0, 1] に正規化
                spectrogram_db_min = spectrogram_db.min()

                spectrogram_db_max = spectrogram_db.max()
                spectrogram_db_norm = (spectrogram_db - spectrogram_db_min) / (
                    spectrogram_db_max - spectrogram_db_min + 1e-8
                )  # 1e-8を加えてゼロ除算を防止

                # カラーマップを適用
                cmap = plt.get_cmap("viridis")
                spectrogram_rgb = cmap(spectrogram_db_norm)  # 出力形状は (freq, time, 4)

                # アルファチャンネルを削除
                spectrogram_rgb = spectrogram_rgb[:, :, :3]  # 形状は (freq, time, 3)

                # PyTorchテンソルに変換し、次元を並べ替え
                spectrogram_rgb = torch.from_numpy(spectrogram_rgb).permute(2, 0, 1).float()  # 形状は (3, freq, time)

                return spectrogram_rgb

            # スペクトログラムをRGB画像に変換
            # バッチとチャネルを選択（例: 最初のバッチとチャネル）
            noisy_rgb = spectrogram_to_rgb(noisy_spectrogram[0, 0])  # [freq, time]
            clean_rgb = spectrogram_to_rgb(clean_spectrogram[0, 0])
            pred_rgb = spectrogram_to_rgb(pred_spectrogram[0, 0])

            # TensorBoardにRGBスペクトログラムをログ
            writer.add_image("spectrogram/noisy_rgb", noisy_rgb, current_step)
            writer.add_image("spectrogram/clean_rgb", clean_rgb, current_step)
            writer.add_image("spectrogram/pred_rgb", pred_rgb, current_step)

        return pred

    def configure_optimizers(self):
        # オプティマイザの選択
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # スケジューラの選択
        if self.scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        elif self.scheduler_type == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **self.scheduler_params)
        elif self.scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        elif self.scheduler_type == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_params)
        elif self.scheduler_type is None:
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
