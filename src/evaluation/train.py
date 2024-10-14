import sys

sys.path.append("ZS-N2N/src")
import argparse
import lightning as l
from omegaconf import OmegaConf
from data.dataset import SpeechDataset
from models.model import ZSN2N
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number to use")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name for logging")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory for TensorBoard")
    parser.add_argument("--noise_type", type=str, required=True, help="Type of noise to use in training")
    parser.add_argument("--noise_file", type=str, required=True, help="Specific noise file to use")
    parser.add_argument("--clean_file", type=str, required=True, help="Specific clean file to use")
    parser.add_argument("--snr_levels", nargs="+", type=int, help="List of SNR levels separated by spaces")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # ハイパーパラメータの設定
    hparams = {
        "loss_type": config.training.loss.type,
        "optimizer_type": config.training.optimizer.type,
        "learning_rate": config.training.optimizer.params.lr,
        "scheduler_type": config.training.scheduler.type if config.training.get("scheduler") else None,
        "max_epoch": config.training.max_epoch,
        "batch_size": config.training.batch_size,
        "n_fft": config.data.n_fft,
        "hop_length": config.data.hop_length,
        "sampling_rate": config.data.sample_rate,
    }

    # Parent Runの名前を作成
    parent_run_name = f"optimizer={hparams['optimizer_type']}_lr={hparams['learning_rate']}_loss={hparams['loss_type']}"

    for snr_level in args.snr_levels:
        # Child Runの名前を作成
        child_run_name = f"snr_{snr_level}"

        # データセットの作成
        dataset = SpeechDataset(
            noise_file=args.noise_file,
            clean_file=args.clean_file,
            noise_type=args.noise_type,
            snr_level=snr_level,
        )

        # モデルの初期化
        model = ZSN2N(config)

        # デバイスの設定
        accelerator = "gpu" if args.gpu >= 0 else "cpu"
        # accelerator = "mps"

        # プログレスバーの設定
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green_yellow",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="cyan",
                processing_speed="#ff1493",
                metrics="#ff1493",
                metrics_text_delimiter="\n",
            )
        )

        # ログディレクトリの設定
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.experiment_name,  # Experiment名（例: experiment_1）
            version=f"{parent_run_name}/{child_run_name}",  # Parent RunとChild Runの階層を表現
        )

        logger.log_hyperparams(hparams)

        # トレーナーの初期化
        trainer = l.Trainer(
            accelerator=accelerator,
            max_epochs=config.training.max_epoch,
            callbacks=[progress_bar],
            logger=logger,
            log_every_n_steps=1,
        )

        data_loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)

        # トレーニングの実行
        trainer.fit(model, data_loader)

        # 予測の実行
        trainer.predict(model, data_loader)


if __name__ == "__main__":
    main()
