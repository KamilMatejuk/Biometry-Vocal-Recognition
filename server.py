import io
import yaml
import tqdm
import torch
import base64
from PIL import Image
from fastapi import FastAPI
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

class Request(BaseModel):
    username: str
    sound_file: str


@app.post("/login")
async def login(body: Request):
    try:
        embedding = model.extract_embedding(body.sound_file)

        successfull = auth(model_name, embedding, f'user_{body.username}')
        return { 'successfull': successfull, 'message': '' }
    except Exception as ex:
        return { 'successfull': False, 'message': str(ex) }
        

@app.post("/register")
async def register(body: Request):
    try:
        embedding = model.extract_embedding(body.sound_file)

        add(model_name, f'user_{body.username}', embedding)
        return { 'successfull': True, 'message': '' }
    except Exception as ex:
        return { 'successfull': False, 'message': str(ex) }

@app.post("/identify")
async def identify(body: Request):
    try:
        embedding = model.extract_embedding(body.sound_file)

        user = get_similar(model_name, embedding)
        username = user.replace('user_', '')
        return { 'successfull': True, 'message': username }
    except Exception as ex:
        return { 'successfull': False, 'message': str(ex) }