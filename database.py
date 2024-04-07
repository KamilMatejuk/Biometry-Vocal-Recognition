import os
import torch
from enum import Enum, auto

from loggers import db_logger as logger


def _dist_euclidean(v1: torch.Tensor, v2: torch.Tensor): return torch.norm(v1 - v2)
def _dist_manhattan(v1: torch.Tensor, v2: torch.Tensor): return torch.sum(torch.abs(v1 - v2))
def _dist_cosine(v1: torch.Tensor, v2: torch.Tensor):    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))


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


def add(model_name: str, label: str, embedding: torch.Tensor) -> None:
    vectors: list[tuple[str, torch.Tensor]] = torch.load(_path(model_name))
    vectors.append((label, embedding))
    torch.save(vectors, _path(model_name))


def get_similar(model_name: str, embedding: torch.Tensor, threshold: float, distance: Distance) -> list[str]:
    vectors: list[tuple[str, torch.Tensor]] = torch.load(_path(model_name))
    similar = []
    if distance == Distance.EUCLIDEAN: dist_func = _dist_euclidean
    if distance == Distance.MANHATTAN: dist_func = _dist_manhattan
    if distance == Distance.COSINE: dist_func = _dist_cosine
    for label, vector in vectors:
        d = dist_func(embedding, vector)
        if d < threshold:
            similar.append(label)
    return similar

def auth(db: list[tuple[str, torch.Tensor]], embedding: torch.Tensor, label: str, threshold: float, distance: Distance):
    similar = set()
    if distance == Distance.EUCLIDEAN: dist_func = _dist_euclidean
    if distance == Distance.MANHATTAN: dist_func = _dist_manhattan
    if distance == Distance.COSINE: dist_func = _dist_cosine
    for dbi in db:
        d = dist_func(embedding, dbi[1])
        if d < threshold:
            similar.add(dbi[0])
    # logger.info(f'Found {len(similar)} in range {threshold} -> {label} {label in similar}')
    return label in similar
