import base64
import io
import json
from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# American accent
# output_path = 'en-us.wav'
# model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

# # British accent
# output_path = 'en-br.wav'
# model.tts_to_file(text, speaker_ids['EN-BR'], output_path, speed=speed)

# # Indian accent
# output_path = 'en-india.wav'
# model.tts_to_file(text, speaker_ids['EN_INDIA'], output_path, speed=speed)

# # Australian accent
# output_path = 'en-au.wav'
# model.tts_to_file(text, speaker_ids['EN-AU'], output_path, speed=speed)

# # Default accent
# output_path = 'en-default.wav'
# model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed)




import asyncio
from aiohttp import web



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

        # ccreate temp file
        path = "temp.wav"
        with open(path, "wb") as f:
            model.tts_to_file(txt, speaker_ids[voice], f, speed=temp)
            f.close()
            f = open(path, "rb")
            data = f.read()
            f.close()
        return web.Response(body=data, headers=headers)


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
    return web.Response(text=json.dumps(
        {
            "models": list(map(lambda x: x, speaker_ids.keys()))
        }
    ))

app = web.Application()

app.router.add_route("GET",'/v1/audio/speech/', handleGet)
app.router.add_route("POST",'/v1/audio/speech', handlePost)
app.router.add_route("GET",'/v1/models', getModels)

runner = web.AppRunner(app)
asyncio.ensure_future(runner.setup()).add_done_callback(lambda _: asyncio.ensure_future(web.TCPSite(runner, '0.0.0.0', 8080).start()))

asyncio.get_event_loop().run_forever()