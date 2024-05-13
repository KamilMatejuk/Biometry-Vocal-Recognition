import os
import torch
from enum import Enum, auto

from loggers import db_logger as logger


def _dist_euclidean(v1: torch.Tensor, v2: torch.Tensor): return torch.norm(v1 - v2)
def _dist_manhattan(v1: torch.Tensor, v2: torch.Tensor): return torch.sum(torch.abs(v1 - v2))
def _dist_cosine(v1: torch.Tensor, v2: torch.Tensor): 
    v1 = v1.flatten()
    v2 = v2.flatten()
    return -torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))


class Distance(Enum):
    EUCLIDEAN = auto()
    MANHATTAN = auto()
    COSINE = auto()



def _path(model_name: str) -> str:
    return f'data/embeddings/{model_name}.db'


def init_empty(model_name: str) -> None:
    os.makedirs(os.path.dirname(_path(model_name)), exist_ok=True)
    torch.save([], _path(model_name))
    logger.info(f'Initialized empty database at {_path(model_name)}')

def get_db(model_name: str):
    return torch.load(_path(f'{model_name}'))
def get_db_raw(model_name: str):
    return torch.load(_path(f'{model_name}_raw'))
def save_db(model_name: str, vectors):
    os.makedirs(os.path.dirname(_path(f'{model_name}')), exist_ok=True)
    torch.save(vectors, _path(f'{model_name}'))
def save_db_raw(model_name: str, vectors_raw):
    os.makedirs(os.path.dirname(_path(f'{model_name}_raw')), exist_ok=True)
    torch.save(vectors_raw, _path(f'{model_name}_raw'))

def add(model_name: str, label: str, embedding: torch.Tensor) -> None:
    vectors_raw = get_db_raw(model_name)
    vectors = get_db(model_name)

    user_vectors = vectors_raw[label] if vectors_raw.get(label) is not None else []
    user_vectors.apppend(embedding)

    vectors[label] = torch.mean(torch.stack(user_vectors), dim=0)
    vectors_raw[label] = user_vectors

    save_db_raw(model_name, vectors_raw)
    save_db(model_name, vectors)

def get_similar(model_name: str, embedding: torch.Tensor, distance: Distance) -> str:
    vectors = get_db(model_name)

    best_label = None
    best_distance = 10000000

    if distance == Distance.EUCLIDEAN: dist_func = _dist_euclidean
    if distance == Distance.MANHATTAN: dist_func = _dist_manhattan
    if distance == Distance.COSINE: dist_func = _dist_cosine

    for label, avg_embedding in vectors.items():
        d = dist_func(avg_embedding, embedding)
        if d < best_distance:
            best_distance = d
            best_label = label

    return best_label

def auth(model_name: str, embedding: torch.Tensor, label: str, distance: Distance) -> bool:
    best_label = get_similar(model_name, embedding, distance)

    return best_label == label
