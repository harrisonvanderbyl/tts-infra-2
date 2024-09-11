

import asyncio
import base64
import json
import openai

vocal_server_map = {
     
}

async def register_vocal_server(request):
    # get from query
    query = request.query
    voice = query["voice"]
    address = query["address"]
    vocal_server_map[voice] = {"address":address}
    return web.Response(text="OK")


async def handleGet(request, query=None):
   
            # get from query
            query = request.query if query is None else query
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

            if voice is None:
                voice = "default"

            if voice in vocal_server_map:
                voice_server = vocal_server_map[voice]
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{voice_server['address']}/v1/audio/speech/", params={"prompt":prompt, "voice":voice, "temp":temp}) as resp:
                        # print(resp)
                        return web.Response(body=await resp.read(), headers=headers)
                    
            else:
                return web.Response(text="No voice server found for voice: "+voice, status=404)

         
        # return web.Response(text="OK")

async def getModels(request):
    return web.Response(text=json.dumps(
        {
            "models": list(vocal_server_map.keys())
        }
    ))

import aiohttp
from aiohttp.abc import Request
from aiohttp import web



async def handlePost(request):
    body = await request.read()
    bodyjson = json.loads(body)
    print(bodyjson)
    text = bodyjson["input"]
    voice = bodyjson.get("voice", None)
    query = {"prompt":base64.b64encode(text.encode("utf-8")).decode("utf-8"), "voice":voice, "temp": bodyjson.get("temp", 1.0)}

    return await handleGet(request, query)

app = web.Application()

app.router.add_route("GET",'/v1/audio/speech/', handleGet)
app.router.add_route("POST",'/v1/audio/speech/', handlePost)
app.router.add_route("POST",'/v1/register/', register_vocal_server)
app.router.add_route("GET",'/v1/models', getModels)
# add default to log other requests
def default(request):
    print(request)
    return web.Response(text="OK")
app.router.add_route("GET", '/{tail:.*}', default)

async def run():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    print("Server started at http://0.0.0.0:8080")

    # run loop
    webo = site.start()

    # do both simultaneously
    async def loop():
        while True:
            await asyncio.sleep(1)
            print("Running")
    tog = asyncio.gather(webo, loop())

    await tog


asyncio.run(run())