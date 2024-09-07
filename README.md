# aTENNuate ([paper](https://arxiv.org/abs/2409.03377))

aTENNuate is a network that can be configured for real-time speech enhancement on raw audio waveforms. It can perform tasks such as audio denoising, super-resolution, and de-quantization. This repo contains the network definition and a set of pre-trained weights for the aTENNuate model.

Note that the repo is meant for denoising performance evaluation on custom audio samples, and is not optimized for inference. It also does not contain the recurrent configuration of the network, so it cannot be directly used for real-time inference by itself. Evaluation should ideally be done on a batch of .wav files at once as expected by the `denoise.py` script.

Please contact Brainchip Inc. to learn more on the full real-time audio denoising solution. And please consider citation [our work](https://arxiv.org/abs/2409.03377) if you find this repo useful.

## Quickstart

One simply needs a working Python environment, and run the following
```
pip install attenuate
```

To run the pre-trained network on custom audio samples, simply put the `.wav` files (or other format supported by `librosa`) into the `noisy_samples` directory (or any directory of your choice), and run the following
```python
import torch
from attenuate import Denoiser

model = Denoiser()
model.eval()

with torch.no_grad():
    model.from_pretrained("PeaBrane/aTENNuate")
    model.denoise('noisy_samples', denoised_dir='test_samples')

# denoised_samples = model.denoise('noisy_samples')  # return torch tensors instead
```
The denoised samples will then be saved as `.wav` files in the `denoised_samples` directory.

## Training

The network should be easily interfaced with your custom training pipeline. The network expects an input of shape `(batch, 1, length)` and an output of the same shape, which can be sampled at any frequency (though the pre-trained weights operate at 16000 Hz). Note that length should be a multiple of 256, due to the downsampling behavior of the network.

The model supports `torch.compile` for training, but the FFT operations will be performed in eager mode still, due to [complex numbers not being supported](https://github.com/pytorch/pytorch/issues/98161). The model does not fully support `torch.amp` yet stably, due to the sensitivity of the SSM layers. It is recommended to train the model with `tensorfloat32` instead, which can be enabled by
```python
from torch.backends import cudnn
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
```

## Denoising samples

### DNS1 synthetic test samples, no reverb

| Noisy Sample | Denoised Sample |
|--------------|----------------|
| [Noisy Sample 1](noisy_samples/clnsp1_train_69005_1_snr15_tl-21_fileid_158.wav) | [Denoised Sample 1](denoised_samples/clnsp1_train_69005_1_snr15_tl-21_fileid_158.wav) |
| [Noisy Sample 2](noisy_samples/clnsp44_wind_97396_2_snr14_tl-26_fileid_271.wav) | [Denoised Sample 2](denoised_samples/clnsp44_wind_97396_2_snr14_tl-26_fileid_271.wav) |
| [Noisy Sample 3](noisy_samples/clnsp52_amMeH4u6AO4_snr5_tl-18_fileid_19.wav) | [Denoised Sample 3](denoised_samples/clnsp52_amMeH4u6AO4_snr5_tl-18_fileid_19.wav) |

### DNS1 real recordings

| Noisy Sample | Denoised Sample |
|--------------|----------------|
| [Noisy Sample 1](noisy_samples/ms_realrec_headset_cafe_spk2_3.wav) | [Denoised Sample 1](denoised_samples/ms_realrec_headset_cafe_spk2_3.wav) |
| [Noisy Sample 2](noisy_samples/audioset_realrec_babycry_2x43exdQ5bo.wav) | [Denoised Sample 2](denoised_samples/audioset_realrec_babycry_2x43exdQ5bo.wav) |
| [Noisy Sample 3](noisy_samples/audioset_realrec_printer_IZHuH27jLUQ.wav) | [Denoised Sample 3](denoised_samples/audioset_realrec_printer_IZHuH27jLUQ.wav) |

## Contribution

Please submit a Github issue if you find any bugs. If you'd like to contribute a new feature, feel free to open a Github issue to discuss, or email yanrpei@gmail.com.

<!-- ## DNS1 synthetic test samples, no reverb

| Noisy Sample | Denoised Sample |
|--------------|----------------|
| <audio controls><source src="noisy_samples/clnsp1_train_69005_1_snr15_tl-21_fileid_158.wav" type="audio/wav"></audio> | <audio controls><source src="denoised_samples/clnsp1_train_69005_1_snr15_tl-21_fileid_158.wav" type="audio/wav"></audio> |
| <audio controls><source src="noisy_samples/clnsp44_wind_97396_2_snr14_tl-26_fileid_271.wav" type="audio/wav"></audio> | <audio controls><source src="denoised_samples/clnsp44_wind_97396_2_snr14_tl-26_fileid_271.wav" type="audio/wav"></audio> |
| <audio controls><source src="noisy_samples/clnsp52_amMeH4u6AO4_snr5_tl-18_fileid_19.wav" type="audio/wav"></audio> | <audio controls><source src="denoised_samples/clnsp52_amMeH4u6AO4_snr5_tl-18_fileid_19.wav" type="audio/wav"></audio> |

## DNS1 real recordings

| Noisy Sample | Denoised Sample |
|--------------|----------------|
| <audio controls><source src="noisy_samples/ms_realrec_headset_cafe_spk2_3.wav" type="audio/wav"></audio> | <audio controls><source src="denoised_samples/ms_realrec_headset_cafe_spk2_3.wav" type="audio/wav"></audio> |
| <audio controls><source src="noisy_samples/audioset_realrec_babycry_2x43exdQ5bo.wav" type="audio/wav"></audio> | <audio controls><source src="denoised_samples/audioset_realrec_babycry_2x43exdQ5bo.wav" type="audio/wav"></audio> |
| <audio controls><source src="noisy_samples/audioset_realrec_printer_IZHuH27jLUQ.wav" type="audio/wav"></audio> | <audio controls><source src="denoised_samples/audioset_realrec_printer_IZHuH27jLUQ.wav" type="audio/wav"></audio> | -->
