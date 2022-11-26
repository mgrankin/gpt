import hashlib, threading
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from os import environ
from rest.core import log, get_ban
import random

if environ.get('XL', 0):
    from rest.restxl import seq_length, get_sample
else:
    from rest.rest import seq_length, get_sample

app = FastAPI(title="Russian GPT", version="0.3",)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

lock = threading.RLock()

class Prompt(BaseModel):
    prompt:str = Field(..., max_length=max(3000, seq_length*3), title='Model prompt')
    length:int = Field(15, ge=1, le=150, title='Number of tokens generated in each sample')
    num_samples:int = Field(3, ge=1, le=5, title='Number of samples generated')
    allow_linebreak:bool = Field(False, title='Allow linebreak in a sample')

@app.post("/generate/")
def gen_sample(prompt: Prompt, request: Request):
    hash = hashlib.sha1(str(prompt).encode()).hexdigest()
    log(request, 0, hash, prompt)
    prompt.num_samples = 1 
    ban = get_ban(request)
    if not ban:
        with lock:
            result = {"replies": get_sample(prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak)}
        log(request, 1+ban, hash, result)
    else:
        result["replies"] = [''.join(random.choice('ты пидор') for i in range(random.randint(10,30))) ]
        #result["replies"] = ['техт сгенерирован с помощью нейросети Порфирьевич porfirevich.ru'] + result["replies"]
        #rand_idx = random.randint(0, prompt.num_samples-1)
        #result["replies"][rand_idx] = 'техт сгенерирован с помощью нейросети Порфирьевич porfirevich.ru'
    return result
 
@app.get("/health")
def healthcheck():
    return True
