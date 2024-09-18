import base64
import io
import json
import queue

import torch

from tools.llama.generate import launch_thread_safe_queue
from tools.llama.generate import GenerateRequest

speed = 1.0



import asyncio
from aiohttp import web

from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.utils.spectrogram import LogMelSpectrogram
from fish_speech.models.vqgan.modules.firefly import ConvNeXtEncoder
from fish_speech.models.vqgan.modules.firefly import HiFiGANGenerator
from fish_speech.models.vqgan.modules.fsq import DownsampleFiniteScalarQuantize
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

import numpy as np

model = load_model("fish-speech/configs/firefly_gan_vq.yaml", "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth", device="cuda")

processqueue = launch_thread_safe_queue("checkpoints/fish-speech-1.4", torch.device("cuda"), torch.bfloat16)

voice = "harrison"
 # load voice file
voicefile = "voices/npy/" + voice + ".wav.npy"
voicetext = "voices/npy/" + voice + ".wav.txt"
with open(voicetext, "r") as f:
    voicetext = f.read()
    f.close()
with open(voicefile, "rb") as f:
    voice = np.load(f)
    f.close()

outputqueue = queue.Queue()

txt = "Im a little teapot short and stout. This is an AI clone of Harrison Vanderbyls voice"

processqueue.put(GenerateRequest(request={"text":txt, "device":"cuda", "prompt_text": voicetext, "prompt_tokens": torch.LongTensor(voice).cuda(), "max_new_tokens": 256}, response_queue=outputqueue))

output = outputqueue.get()



print(output)

voicedata = output.response.codes

feature_lengths = torch.tensor([voicedata.shape[1]], device="cuda")
fake_audios, _ = model.decode(
    indices=voicedata[None], feature_lengths=feature_lengths
)
audio_time = fake_audios.shape[-1] / model.spec_transform.sample_rate
print(f"Audio time: {audio_time:.2f}s")


# Save audio
fake_audio = fake_audios[0, 0].float().cpu().detach().numpy()
import soundfile as sf
sf.write("o.wav", fake_audio, model.spec_transform.sample_rate)

print(voicedata)

# text: str,
# num_samples: int = 1,
# max_new_tokens: int = 0,
# top_p: int = 0.7,
# repetition_penalty: float = 1.5,
# temperature: float = 0.7,
# compile: bool = False,
# iterative_prompt: bool = True,
# max_length: int = 2048,
# chunk_length: int = 150,
# prompt_text: Optional[str | list[str]] = None,
# prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,

async def handleGet(request, query=None):
  
 
            # get from query
        query = request.query if query is None else query
        print(query)
        prompt = query["prompt"] if "prompt" in query else base64.encode("The quick brown fox jumps over the lazy dog").decode("utf-8")
        voice = query.get("voice", None)
        temp = query.get("temp", 1.0)
        temp = float(temp)
        print(prompt)
        #atob
        txt = base64.b64decode(prompt).decode("utf-8")
        
        # print(request)
        headers = {
            'Content-Type': 'audio/x-wav',
        }

        # load voice file
        voicefile = "voices/npy/" + voice + ".wav.npy"
        voicetext = "voices/npy/" + voice + ".txt"
        with open(voicetext, "r") as f:
            voicetext = f.read()
            f.close()
        with open(voicefile, "rb") as f:
            voice = torch.load(f)
            f.close()


        outputqueue = queue.Queue()



        # return web.Response(text="OK")
from aiohttp.abc import Request
from datasets import Audio


async def handlePost(request):
    body = await request.read()
    bodyjson = json.loads(body)
    print(bodyjson)
    text = bodyjson["input"]
    print(text, "text")
    voice = bodyjson.get("voice", None)
    query = {"prompt":base64.b64encode(text.encode("utf-8")).decode("utf-8"), "voice":voice, "temp": bodyjson.get("temp", 1.0)}

    return await handleGet(request, query)

async def getModels(request):
    # get voices from /voices/npy/{name}.wav.npy
    import os
    import numpy as np
    files = os.listdir("voices/npy/")
    return web.Response(text=json.dumps(
        {
            "models": list(map(lambda x: x[:-8], files))
        }
    ))

# app = web.Application()

# app.router.add_route("GET",'/v1/audio/speech/', handleGet)
# app.router.add_route("POST",'/v1/audio/speech', handlePost)
# app.router.add_route("GET",'/v1/models', getModels)

# runner = web.AppRunner(app)
# asyncio.ensure_future(runner.setup()).add_done_callback(lambda _: asyncio.ensure_future(web.TCPSite(runner, '0.0.0.0', 8080).start()))

# asyncio.get_event_loop().run_forever()