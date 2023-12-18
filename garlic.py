
#from fastapi import FastAPI, Request
from typing import Union
import requests
from transformers import pipeline
import io
from io import BytesIO
import base64
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
global prompt
prompt = "Oragami Christmas Tree"
img = Image.new('RGB', (512, 512))
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def toB64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())

@app.get("/")
def read_root():
    return "Garlic Phoney Server Online!"

@app.get("/check")
def check():
    return {"prompt": prompt, "image": toB64(img)}

@app.post("/prompt")
async def setprompt(pr: Request):
    global prompt
    input = await pr.json()
    prompt = input["prompt"]
    return {"result": 1}

#
#Setup AIs
#

kwords = pipeline("text2text-generation", model="google/flan-t5-large", device="cpu")
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device="cuda")

#
#Setup Functions
#

def turn(prompt):
    response = requests.post("http://192.168.1.231:7860/sdapi/v1/txt2img", json={"prompt": prompt, "steps": 1, "cfg_scale": 1})
    img = Image.open(io.BytesIO(base64.b64decode(response.json()['images'][0])))
    return (img, pipe(img)[0]['generated_text'])
def gen(prompt):
    response = requests.post("http://192.168.1.231:7860/sdapi/v1/txt2img", json={"prompt": prompt, "steps": 1, "cfg_scale": 1})
    img = Image.open(io.BytesIO(base64.b64decode(response.json()['images'][0])))
    return img



def fail(success):
    words = kwords("Write keywords to go with the following sentence. " + success)[0]['generated_text']
    uncompiled = []
    for w in words.split(" "):
        uncompiled.append(w)
        uncompiled.append(kwords("Q: what is similar to " + w)[0]['generated_text'])
    uc =""
    for w in uncompiled:
        uc = uc + w + " "
    uc.removesuffix(" ")

    return kwords("Form a scene using these keywords: " + uc)[0]['generated_text']

@app.get("/step/")
def step():
    global prompt
    global img
    print(prompt)
    img = gen(prompt)

    
    prompt = pipe(img)[0]['generated_text']
    prompt = fail(prompt)
    print("Last output \"" + prompt + "\"")
    return {"result": 1}