import requests
from transformers import pipeline
from PIL import Image
import io
import base64
import flask
kwords = pipeline("text2text-generation", model="google/flan-t5-large", device="cpu")

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device="cuda")

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

prompt = "Dog laying under a christmas tree"

for i in range(3):
    img = gen(prompt)

    clear_output()
    display(img)
    
    prompt = pipe(img)[0]['generated_text']
    prompt = fail(prompt)
    print("OH! I think I see \"" + prompt + "\"")
