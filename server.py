import io
import yaml
import tqdm
import torch
import base64
import traceback
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from database import Distance, add, get_similar, auth
import wespeaker 

app = FastAPI()

# Allowing CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model_name = "english"
model = wespeaker.load_model(model_name)


def base64_to_file(sound_base64: str) -> str:
    sound_bytes = base64.b64decode(sound_base64)
    filename = 'data/tmp_sound.wav'
    with open(filename, 'wb') as f:
        f.write(sound_bytes)
    return filename


class Request(BaseModel):
    username: str
    sound_base64: str


@app.post("/login")
async def login(body: Request):
    try:
        sound_file = base64_to_file(body.sound_base64)
        embedding = model.extract_embedding(sound_file)
        successfull = auth(model_name, embedding, body.username, Distance.COSINE)
        return { 'successfull': successfull, 'message': '' }
    except Exception as ex:
        traceback.print_exc()
        return { 'successfull': False, 'message': str(ex) }
        

@app.post("/register")
async def register(body: Request):
    try:
        sound_file = base64_to_file(body.sound_base64)
        embedding = model.extract_embedding(sound_file)
        add(model_name, body.username, embedding)
        return { 'successfull': True, 'message': '' }
    except Exception as ex:
        traceback.print_exc()
        return { 'successfull': False, 'message': str(ex) }


@app.post("/identify")
async def identify(body: Request):
    try:
        sound_file = base64_to_file(body.sound_base64)
        embedding = model.extract_embedding(sound_file)
        username = get_similar(model_name, embedding, Distance.COSINE).replace('user_', '')
        return { 'successfull': True, 'message': username }
    except Exception as ex:
        traceback.print_exc()
        return { 'successfull': False, 'message': str(ex) }
