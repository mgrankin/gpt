import httpx
import hashlib
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware
from os import environ
from storage import log, get_ban
import random
from common import Prompt
import restproxy

app = FastAPI(title="Russian GPT", version="0.3",)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.post("/generate/")
async def gen_sample(prompt: Prompt, request: Request):
    hash = hashlib.sha1(str(prompt).encode()).hexdigest()
    log(prompt.model, request, 0, hash, prompt)
    ban = get_ban(request)
    if not ban:
        if prompt.model == 'frida':
            result = {"replies": restproxy.get_sample(prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak)}
        else:
            port = ''
            if prompt.model == 'poetry': port = '7000'
            if prompt.model == 'gpt3': port = '7001'
            if prompt.model == 'xlarge': port = '7002'

            host_url = f'http://127.0.0.1:{port}/generate/'
            async with httpx.AsyncClient() as client:
                response = await client.post(host_url, json=prompt.dict())
            if response.status_code == 200:
                result = response.json()
            else:
                raise HTTPException(status_code=response.status_code)

        log(prompt.model, request, 1+ban, hash, result)
    else:
        result = result = {"replies": [''.join(random.choice('вам отказано') for i in range(random.randint(10,30)))] }
    return result

@app.get("/models")
def get_models():
    return ['xlarge', 'gpt3', 'frida']

@app.get("/health")
def healthcheck():
    return True

# uvicorn api:app --host 0.0.0.0 --port 8280