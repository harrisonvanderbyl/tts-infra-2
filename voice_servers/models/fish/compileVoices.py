
import torch


from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.utils.spectrogram import LogMelSpectrogram
from fish_speech.models.vqgan.modules.firefly import ConvNeXtEncoder
from fish_speech.models.vqgan.modules.firefly import HiFiGANGenerator
from fish_speech.models.vqgan.modules.fsq import DownsampleFiniteScalarQuantize


from transformers import WhisperProcessor, WhisperForConditionalGeneration
layers =  6*2
# from datasets import load_dataset

# # load model and processor

whisperprocessor = WhisperProcessor.from_pretrained("openai/whisper-small")

whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

whisper.config.forced_decoder_ids = None


def load_model(config_name, checkpoint_path, device="cuda"):
    

    model = FireflyArchitecture(  
        backbone=ConvNeXtEncoder(
            input_channels=160,
            depths=[3, 3, 9, 3],
            dims=[128, 256, 384, 512],
            drop_path_rate=0.2,
            kernel_size=7,
        ),
        head=HiFiGANGenerator(
            hop_length=512,
            upsample_rates=[8, 8, 2, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            num_mels=512,
            upsample_initial_channel=512,
            pre_conv_kernel_size=13,
            post_conv_kernel_size=13,
        ),
        quantizer=DownsampleFiniteScalarQuantize(
            input_dim=512,
            n_groups=8,
            n_codebooks=1,
            levels=[8, 5, 5, 5],
            downsample_factor=[2, 2],
        ),
        spec_transform=LogMelSpectrogram(
            sample_rate=44100,
            n_mels=160,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
        ),
    )
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    return model


model = load_model("fish-speech/configs/firefly_gan_vq.yaml", "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth", device="cuda")

# encode reference audio
#         logger.info(f"Processing in-place reconstruction of {input_path}")
import torchaudio
# Load audio

import os
# go into ./voices/*.wav and compile the voices

files = os.listdir("voices/wav/")
files = [f for f in files if f.endswith(".wav")]

for f in files:

    audio, sr = torchaudio.load(str("voices/wav/" + f))
    resampled,*_ = torchaudio.transforms.Resample(sr, 16000)(audio)
    txt = whisperprocessor.batch_decode(whisper.generate(whisperprocessor(resampled, sampling_rate=16000, return_tensors="pt").input_features ), skip_special_tokens=True)[0]

    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    audio = torchaudio.functional.resample(
        audio, sr, model.spec_transform.sample_rate
    )

    audios = audio[None].to("cuda")

    # VQ Encoder
    audio_lengths = torch.tensor([audios.shape[2]], device="cuda", dtype=torch.long)
    indices = model.encode(audios, audio_lengths)[0][0]

    import numpy as np
    # Save indices
    np.save("voices/npy/"+f+".npy", indices.cpu().detach().numpy())
    # save text
    with open("voices/npy/"+f+".txt", "w") as f:
        f.write(txt)


