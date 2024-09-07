import math
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from einops.layers.torch import EinMix
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from huggingface_hub import hf_hub_download


@torch.compiler.disable
def fft_conv(equation, input, kernel, *args):
    input, kernel = input.float(), kernel.float()
    args = tuple(arg.cfloat() for arg in args)
    n = input.shape[-1]
    
    kernel_f = torch.fft.rfft(kernel, 2 * n)
    input_f = torch.fft.rfft(input, 2 * n)
    output_f = torch.einsum(equation, input_f, kernel_f, *args)
    output = torch.fft.irfft(output_f, 2 * n)
    
    return output[..., :n]


def ssm_basis_kernels(A, B, log_dt, length):
    log_A_real, A_imag = A.T  # (2, num_coeffs)
    lrange = torch.arange(length, device=A.device)
    dt = log_dt.exp()
    
    dtA_real, dtA_imag = -dt * F.softplus(log_A_real), dt * A_imag        
    return (dtA_real[:, None] * lrange).exp() * torch.cos(dtA_imag[:, None] * lrange), B * dt[:, None]


def opt_ssm_forward(input, K, B_hat, C):
    """SSM ops with einsum contractions
    """
    batch, c_in, _ = input.shape
    c_out, coeffs = C.shape
    
    if (1 / c_in + 1 / c_out) > (1 / batch + 1 / coeffs):
        if c_in * c_out <= coeffs:
            kernel = torch.einsum('dn,nc,nl->dcl', C, B_hat, K)
            return fft_conv('bcl,dcl->bdl', input, kernel)
    else:
        if coeffs <= c_in:
            x = torch.einsum('bcl,nc->bnl', input, B_hat)
            x = fft_conv('bnl,nl->bnl', x, K)
            return torch.einsum('bnl,dn->bdl', x, C)
        
    return fft_conv('bcl,nl,nc,dn->bdl', input, K, B_hat, C)


class SSMLayer(nn.Module):
    def __init__(self, 
                 num_coeffs: int, 
                 in_channels: int, 
                 out_channels: int, 
                 repeat: int):
        from torch.backends import opt_einsum
        assert opt_einsum.is_available()
        opt_einsum.strategy = 'optimal'
        
        super().__init__()

        init_parameter = lambda mat: Parameter(torch.tensor(mat, dtype=torch.float))
        normal_parameter = lambda fan_in, shape: Parameter(torch.randn(*shape) * math.sqrt(2 / fan_in))
        
        A_real, A_imag = 0.5 * np.ones(num_coeffs), math.pi * np.arange(num_coeffs)
        log_A_real = np.log(np.exp(A_real) - 1)  # inv softplus
        B = np.ones(num_coeffs)
        A = np.stack([log_A_real, A_imag], -1)
        log_dt = np.linspace(np.log(0.001), np.log(0.1), repeat)
        
        A = np.tile(A, (repeat, 1))
        B = np.tile(B[:, None], (repeat, in_channels)) / math.sqrt(in_channels)
        log_dt = np.repeat(log_dt, num_coeffs)
            
        self.log_dt, self.A, self.B = init_parameter(log_dt), init_parameter(A), init_parameter(B)
        self.C = normal_parameter(num_coeffs * repeat, (out_channels, num_coeffs * repeat))
    
    def forward(self, input):
        K, B_hat = ssm_basis_kernels(self.A, self.B, self.log_dt, input.shape[-1])
        return opt_ssm_forward(input, K, B_hat, self.C)
                

class LayerNormFeature(nn.Module):
    """Apply LayerNorm to the channel dimension
    """
    def __init__(self, features):
        super().__init__()
        self.layer_norm = nn.LayerNorm(features)
    
    def forward(self, input):
        return self.layer_norm(input.moveaxis(-1, -2)).moveaxis(-1, -2)


class Denoiser(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 channels=[16, 32, 64, 96, 128, 256], 
                 num_coeffs=16, 
                 repeat=16, 
                 resample_factors=[4, 4, 2, 2, 2, 2], 
                 pre_conv=True):
        super().__init__()
        
        depth = len(channels)
        self.depth = depth
        self.channels = [in_channels] + channels
        self.num_coeffs = num_coeffs
        self.repeat = repeat
        self.pre_conv = pre_conv
        
        self.down_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=True) for (c_in, c_out, r) in zip(self.channels[:-1], self.channels[1:], resample_factors)
        ])
        self.up_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=False) for (c_in, c_out, r) in zip(self.channels[1:], self.channels[:-1], resample_factors)
        ])
        self.hid_ssms = nn.Sequential(
            self.ssm_block(self.channels[-1], True), self.ssm_block(self.channels[-1], True), 
        )
        self.last_ssms = nn.Sequential(
            self.ssm_block(self.channels[0], True), self.ssm_block(self.channels[0], False), 
        )
    
    def ssm_pool(self, in_channels, out_channels, resample_factor, downsample=True):
        if downsample:
            return nn.Sequential(
                self.ssm_block(in_channels, use_activation=True), 
                EinMix('b c (t r) -> b d t', weight_shape='c d r', c=in_channels, d=out_channels, r=resample_factor), 
            )
        else:
            return nn.Sequential(
                EinMix('b c t -> b d (t r)', weight_shape='c d r', c=in_channels, d=out_channels, r=resample_factor), 
                self.ssm_block(out_channels, use_activation=True), 
            )
    
    def ssm_block(self, channels, use_activation=False):
        block = nn.Sequential()
        if channels > 1 and self.pre_conv:
            block.append(nn.Conv1d(channels, channels, 3, 1, 1, groups=channels))
        block.append(SSMLayer(self.num_coeffs, channels, channels, self.repeat))
        if use_activation:
            if channels > 1:
                block.append(LayerNormFeature(channels))
            block.append(nn.SiLU())
        
        return block
    
    def forward(self, input):
        x, skips = input, []
        
        # encoder
        for ssm in self.down_ssms:
            skips.append(x)
            x = ssm(x)
        
        # neck
        x = self.hid_ssms(x)
        
        # decoder
        for (ssm, skip) in zip(self.up_ssms[::-1], skips[::-1]):
            x = ssm[0](x)
            x = x + skip
            x = ssm[1](x)
            
        return self.last_ssms(x)
    
    def denoise_single(self, noisy):
        assert noisy.ndim == 2, f"noisy input should be shaped (samples, length)"
        noisy = noisy[:, None, :]  # unsqueeze channel dim

        padding = 256 - noisy.shape[-1] % 256
        noisy = F.pad(noisy, (0, padding))
        denoised = self.forward(noisy)

        return denoised.squeeze(1)[..., :-padding]

    def denoise_multiple(self, noisy_samples):
        audio_lens = [noisy.shape[-1] for noisy in noisy_samples]
        max_len = max(audio_lens)
        noisy_samples = torch.stack([F.pad(noisy, (0, max_len - noisy.shape[-1])) for noisy in noisy_samples])
        denoised_samples = self.denoise_single(noisy_samples)

        return [denoised[..., :audio_len] for (denoised, audio_len) in zip(denoised_samples, audio_lens)]
    
    def denoise(self, noisy_dir, denoised_dir=None):
        noisy_dir = Path(noisy_dir)
        denoised_dir = None if denoised_dir is None else Path(denoised_dir)
        
        noisy_files = [fn for fn in noisy_dir.glob('*.wav')]
        noisy_samples = [torch.tensor(librosa.load(wav_file, sr=16000)[0]) for wav_file in noisy_files]
        print("denoising...")
        denoised_samples = self.denoise_multiple(noisy_samples)
        
        if denoised_dir is not None:
            print("saving audio files...")
            for (denoised, noisy_fn) in zip(denoised_samples, noisy_files):
                torchaudio.save(denoised_dir / f"{noisy_fn.stem}.wav", denoised[None, :], 16000)
                
        return denoised_samples
    
    def from_pretrained(self, repo_id):
        print(f"loading weights from {repo_id}...")
        model_weights_path = hf_hub_download(repo_id=repo_id, filename="weights.pt")
        self.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    
    
# def opt_ssm_forward_with_kernels(input, B_hat, C, K=None, kernel=None):
#     batch, c_in, _ = input.shape
#     c_out, coeffs = C.shape
    
#     if (1 / c_in + 1 / c_out) > (1 / batch + 1 / coeffs):
#         if c_in * c_out <= coeffs:
#             return fft_conv('bcl,dcl->bdl', input, kernel)
#     else:
#         if coeffs <= c_in:
#             x = torch.einsum('bcl,nc->bnl', input, B_hat)
#             x = fft_conv('bnl,nl->bnl', x, K)
#             return torch.einsum('bnl,dn->bdl', x, C)
        
#     return fft_conv('bcl,nl,nc,dn->bdl', input, K, B_hat, C)