This repo contains the network definition and a set of pre-trained weights for the `aTENNuate` model. It is meant for users to evaluate the performance of the network on custom audio samples.

Note that the repo does not contain the recurrent configuration of the network, hence it by itself cannot be efficiently used for real-time inference. In addition, the pre-trained network is not quantized or sparsified.

# Denoising samples

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