# from enviroment variables, list the voice models to server
# and start the server


# cd models/melotts and run python start.py
import asyncio
import base64
import json

from aiohttp import web

# start the server
import os
import sys
# start process cd models/melotts && python start.py 
os.system("tmux new-session -d -s melotts 'cd models/melotts && python start.py'") # 8080

# start the server
os.system("tmux new-session -d -s ct1 'cd models/cottontail-lg && python expxl.py'") # 8081


os.system("tmux new-session -d -s ct2 'cd models/cottontail-sm && python exp.py'") # 8081


def getModels(request):

    import requests
    # get models from melotts
    melotts = requests.get("http://0.0.0.0:8080/v1/models")
    melotts = melotts.json()
    melotts = melotts["models"]

    # get models from cottontail
    cottontail = requests.get("http://0.0.0.0:8085/v1/models")
    cottontail = cottontail.json()
    cottontail = cottontail["models"]

    # get models from cottontail-lg
    cottontail2 = requests.get("http://0.0.0.0:8090/v1/models")
    cottontail2 = cottontail2.json()
    cottontail2 = cottontail2["models"]


    return web.Response(text=json.dumps(
        {
            "models": ["mixed"],
            "voices": [
                *["melotts/" + x for x in melotts],
                *["cottontail/" + x for x in cottontail],
                *["cottontail-lg/" + x for x in cottontail2]
            ]
        }
    ))

async def handleGet(request):
    query = request.query
    query = dict(query)
    voice = query.get("voice", None)
    if voice:
        voice = voice.split("/")

    query["voice"] = voice[1] if voice else None
    query["input"] = base64.b64decode(query["prompt"]).decode("utf-8")

    if voice and voice[0] == "melotts":
        import requests
        response = requests.post("http://0.0.0.0:8080/v1/audio/speech", json=query, stream=True)
        
        return web.Response(body=response.content, headers={"Content-Type": "audio/mpeg"})
    
    elif voice and voice[0] == "cottontail":
        import requests
        response = requests.post("http://0.0.0.0:8085/v1/audio/speech", json=query, stream=True)

        return web.Response(body=response.content, headers={"Content-Type": "audio/mpeg"}, status=response.status_code)
    elif voice and voice[0] == "cottontail-lg":
        import requests
        response = requests.post("http://0.0.0.0:8090/v1/audio/speech", json=query, stream=True)
    
        return web.Response(body=response.content, headers={"Content-Type": "audio/mpeg"}, status=response.status_code)
    else:
        return web.Response(text="Error: Voice not found")

        

async def handlePost(request):
    body = await request.read()
    bodyjson = json.loads(body)
    print(bodyjson)
    text = bodyjson["input"]
    print(text, "text")
    voice = bodyjson.get("voice", None)
    query = {"prompt":base64.b64encode(text.encode("utf-8")).decode("utf-8"), "voice":voice, "temp": bodyjson.get("temp", 1.0)}
    class reqcopy:
        def __init__(self, query):
            self.query = query
    ns = reqcopy(query)
    return await handleGet(ns)

async def handleSTPost(request):
    body = await request.read()
    bodyjson = json.loads(body)
    print(bodyjson)
    text = bodyjson["input"]
    print(text, "text")
    voice = bodyjson.get("voice", None)
    query = {"prompt":base64.b64encode(text.encode("utf-8")).decode("utf-8"), "voice":voice, "temp": bodyjson.get("temp", 1.0)}
    redirect = "http://0.0.0.0:8079/v1/audio/speech?" + "&".join([f"{k}={v}" for k,v in query.items()])
    return web.Response(text=redirect,headers={"Content-Type": "text/plain"})

app = web.Application()

app.router.add_route("GET",'/v1/models', getModels)
app.router.add_route("GET",'/v1/audio/speech', handleGet)
app.router.add_route("POST",'/v1/audio/speech', handlePost)
app.router.add_route("POST",'/v1/audio/speech/sillytavern', handleSTPost)


runner = web.AppRunner(app)
asyncio.ensure_future(runner.setup()).add_done_callback(lambda _: asyncio.ensure_future(web.TCPSite(runner, '0.0.0.0', 80).start()))

asyncio.get_event_loop().run_forever()