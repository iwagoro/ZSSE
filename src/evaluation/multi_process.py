import os
import subprocess
import multiprocessing
from pathlib import Path
from omegaconf import OmegaConf, ListConfig


def run_single_gpu_script(
    script_path, config_path, gpu, experiment_name, log_dir, noise_type, noise_file, clean_file, snr_levels
):
    snr_levels_args = ["--snr_levels"] + list(map(str, snr_levels))
    command = [
        "python",
        script_path,
        "--config_path",
        config_path,
        "--gpu",
        str(gpu),
        "--experiment_name",
        experiment_name,
        "--log_dir",
        log_dir,
        "--noise_type",
        noise_type,
        "--noise_file",
        noise_file,
        "--clean_file",
        clean_file,
    ] + snr_levels_args
    subprocess.run(command)


def run_train_script():
    #!  並列処理を行うためのコンフィグファイルを読み込む
    config_path = "/Users/rockwell/Documents/python/ZS-N2N/src/evaluation/multi_process_params.yml"
    config = OmegaConf.load(config_path)

    #!  並列処理を行うためのコンフィグファイルに記載されたパラメータを取得
    script_path = config.multi_process.script_path
    training_config = config.multi_process.training_config
    gpus = config.multi_process.gpus
    num_workers = config.multi_process.num_workers
    log_dir = config.multi_process.log_dir
    noise_type = config.multi_process.noise_type
    noise_path = config.multi_process.noise_path
    noise_samples = config.multi_process.noise_samples
    clean_path = config.multi_process.clean_path
    clean_samples = config.multi_process.clean_samples
    snr_levels = config.multi_process.snr_levels
    is_test = config.multi_process.is_test

    #!  script_pathの存在確認
    if not Path(script_path).exists():
        raise FileNotFoundError(f"Script path {script_path} does not exist")

    #!  各種パラメータのエラーチェック
    if not isinstance(num_workers, int):
        raise ValueError("num_workers must be an integer")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config path {config_path} does not exist")
    if not isinstance(log_dir, str):
        raise ValueError("log_dir must be a string")
    if noise_type != "white" and not Path(os.path.join(noise_path, noise_type)).exists():
        raise FileNotFoundError(f"Noise path {noise_path} does not exist")
    if not isinstance(noise_samples, int):
        raise ValueError("noise_samples must be an integer")
    if not Path(clean_path).exists():
        raise FileNotFoundError(f"Clean path {clean_path} does not exist")
    if not isinstance(clean_samples, int):
        raise ValueError("clean_samples must be an integer")
    if not isinstance(snr_levels, ListConfig):
        raise ValueError("snr_levels must be a list of integers")
    if not all(isinstance(snr_level, int) for snr_level in snr_levels):
        raise ValueError("All elements in snr_levels must be integers")

    #! ノイズファイルをソートして取得
    noise_files = []
    if noise_type == "white":
        noise_files = [""] * noise_samples
    else:
        noise_files = sorted(Path(os.path.join(noise_path, noise_type)).glob("*.wav"))
        if len(noise_files) < noise_samples:
            raise ValueError(f"Not enough noise files found. Required: {noise_samples}, Found: {len(noise_files)}")

    selected_noise_files = noise_files[:noise_samples]

    #! cleanファイルをソートして取得
    clean_files = sorted(Path(clean_path).glob("*.wav"))
    if len(clean_files) < clean_samples:
        raise ValueError(f"Not enough clean files found. Required: {clean_samples}, Found: {len(clean_files)}")

    selected_clean_files = clean_files[:clean_samples]

    #! experiment_nameの設定
    experiment_name = f"{noise_type}" if not is_test else f"{noise_type}_test"

    #! gpusのエラーチェック
    if gpus == "None":
        #! 単一プロセスで実行
        for noise_file in selected_noise_files:
            for clean_file in selected_clean_files:
                run_single_gpu_script(
                    script_path,
                    training_config,
                    -1,
                    experiment_name,
                    log_dir,
                    noise_type,
                    noise_file,
                    clean_file,
                    snr_levels,
                )
    elif not isinstance(gpus, ListConfig):
        raise ValueError("gpus must be a list of integers")
    elif not all(isinstance(gpu, int) for gpu in gpus):
        raise ValueError("All elements in gpus must be integers")
    else:
        #! 複数GPUが指定された場合は並列実行
        processes = []
        for idx, gpu in enumerate(gpus):
            #! ノイズとクリーンファイルを各GPUに割り当てる
            noise_file = selected_noise_files[idx % len(selected_noise_files)]
            clean_file = selected_clean_files[idx % len(selected_clean_files)]
            process = multiprocessing.Process(
                target=run_single_gpu_script,
                args=(
                    script_path,
                    training_config,
                    gpu,
                    experiment_name,
                    log_dir,
                    noise_type,
                    noise_file,
                    clean_file,
                    snr_levels,
                ),
            )
            process.start()
            processes.append(process)

        #! すべてのプロセスが終了するのを待機
        for process in processes:
            process.join()


if __name__ == "__main__":
    run_train_script()
