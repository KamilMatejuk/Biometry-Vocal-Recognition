import io
import yaml
import tqdm
import torch
import base64
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from dataset import get_dl_db_ui_ii
from database import auth, Distance
from models.ghost_face import GhostFaceModel as Model
from models.ghost_face import GhostFacePreprocessorTest as PreprocessorTest


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
dl_db_ui_ii = get_dl_db_ui_ii('data/inputs', 'cpu', None)

with open('config.yml') as f:
    config = yaml.safe_load(f)
config = config.get('ghost', {}).get('all', {})

model = Model('cpu', config)
model.load_model_and_optimizer(f'data/checkpoints/{model}/full_ds.chpt')

# load database
database = []
for embedding, label, _ in tqdm.tqdm(dl_db_ui_ii, desc='Db'):
    database.append((embedding, label.item()))


def base64_to_embed(image_base64: str) -> torch.Tensor:
    image_bytes = base64.b64decode(image_base64)
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_transformed = PreprocessorTest.preprocess(image_pil).unsqueeze(0)
    embedding = model.get_embedding(image_transformed)
    print(f'Got embedding of shape {embedding.shape}: [{", ".join(map(lambda x: f"{x.item():.2f}", embedding[:15]))} ...')
    return embedding



class Request(BaseModel):
    username: str
    image_base64: str


@app.post("/login")
async def login(body: Request):
    try:
        embedding = base64_to_embed(body.image_base64)
        successfull = auth(database, embedding, body.username, 0.44, Distance.EUCLIDEAN)
        return { 'successfull': successfull, 'message': '' }
    except Exception as ex:
        return { 'successfull': False, 'message': str(ex) }
        

@app.post("/register")
async def register(body: Request):
    try:
        embedding = base64_to_embed(body.image_base64)
        database.append((embedding, body.username))
        return { 'successfull': True, 'message': '' }
    except Exception as ex:
        raise ex
        return { 'successfull': False, 'message': str(ex) }
