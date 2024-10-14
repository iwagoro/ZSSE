import torch


def stft(wav, n_fft, hop_length):
    # wav shape: [batch, channel, data]
    batch_size, num_channels, data_length = wav.shape
    window = torch.hann_window(n_fft, device=wav.device)

    # Initialize a list to store the STFT results for each sample
    stft_results = []

    # Loop through each batch and channel to apply STFT
    for b in range(batch_size):
        channel_stft = []
        for c in range(num_channels):
            single_stft = torch.stft(
                input=wav[b, c], n_fft=n_fft, hop_length=hop_length, normalized=True, return_complex=True, window=window
            )
            channel_stft.append(torch.view_as_real(single_stft))
        # Concatenate channels and append to the result
        stft_results.append(torch.stack(channel_stft, dim=0))

    # Combine the batch and channel results into a single tensor
    stft_results = torch.stack(stft_results, dim=0)  # Shape: [batch, channel, width, height, complex]
    return stft_results


def istft(stft, n_fft, hop_length):
    # stft shape: [batch, channel, width, height, complex]
    batch_size, num_channels, _, _, _ = stft.shape
    window = torch.hann_window(n_fft, device=stft.device)

    # Initialize a list to store the ISTFT results for each sample
    wav_results = []

    # Loop through each batch and channel to apply ISTFT
    for b in range(batch_size):
        channel_wav = []
        for c in range(num_channels):
            single_wav = torch.istft(
                input=torch.view_as_complex(stft[b, c]),
                n_fft=n_fft,
                hop_length=hop_length,
                normalized=True,
                window=window,
            )
            channel_wav.append(single_wav)
        # Concatenate channels and append to the result
        wav_results.append(torch.stack(channel_wav, dim=0))

    # Combine the batch and channel results into a single tensor
    wav_results = torch.stack(wav_results, dim=0)  # Shape: [batch, channel, data]
    return wav_results
