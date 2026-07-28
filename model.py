from os import environ
model_path = environ["MODEL"]

if model_path == "sparse_xl":
    from rest.original import get_sample
elif 'fred' in model_path:
    from rest.frida import get_sample
else:
    from rest.lawa import get_sample

from fastapi import FastAPI, Request
from front.common import Prompt

import threading
lock = threading.RLock()

app = FastAPI(title=f"Serving {model_path}", version="0.1",)

@app.post("/generate/")
def gen_sample(prompt: Prompt, request: Request):
    with lock:
        return {"replies": get_sample(prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak, prompt.temperature)}

@app.get("/health")
def healthcheck():
    return True

# MODEL=large/pelevin uvicorn model:app --host 0.0.0.0 --port 8000
