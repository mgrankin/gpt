import httpx
import hashlib
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware
from os import environ
from storage import log, get_ban
import random
from common import Prompt
import restproxy

MODEL_PORTS = {
    'lawa': '7005',
    'mig': '7004',
    'original': '7002',
    'gpt3': '7001',
    'frida': '7003',
}

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
        if prompt.model == 'frida не работает':
            result = {"replies": restproxy.get_sample(prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak, prompt.temperature)}
        else:
            port = MODEL_PORTS.get(prompt.model)
            if port is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown model: {prompt.model}",
                )
            
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
    return list(MODEL_PORTS)

@app.get("/health")
def healthcheck():
    return True

# uvicorn api:app --host 0.0.0.0 --port 8280
