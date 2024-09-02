from pathlib import Path

import librosa
import torch
import torchaudio
from torch.nn import functional as F

from simple_ssm_autoencoder import SSMAutoencoder

torch.set_grad_enabled(False)


def denoise(model, noisy):    
    noisy = noisy[:, None, :]  # unsqueeze channel dim
    
    padding = 256 - noisy.shape[-1] % 256
    noisy = F.pad(noisy, (0, padding))
    denoised = model(noisy)
    
    return denoised.squeeze(1)[..., :-padding]


load_dir = Path("noisy_samples")
save_dir = Path("denoised_samples")

noisy_files = [fn for fn in load_dir.glob('*.wav')]
noisy_samples = [torch.tensor(librosa.load(wav_file, sr=16000)[0]) for wav_file in noisy_files]
audio_lens = [noisy.shape[-1] for noisy in noisy_samples]
max_len = max(audio_lens)
noisy_samples = torch.stack([F.pad(noisy, (0, max_len - noisy.shape[-1])) for noisy in noisy_samples])

model = SSMAutoencoder()
model.load_state_dict(torch.load('weights.pt', map_location='cpu'))
model = model.eval()
denoised_samples = denoise(model, noisy_samples)

for i, (audio_len, denoised, noisy_fn) in enumerate(zip(audio_lens, denoised_samples, noisy_files)):
    denoised = denoised_samples[[i], :audio_len]
    torchaudio.save(save_dir / f"{noisy_fn.stem}.wav", denoised, 16000)
