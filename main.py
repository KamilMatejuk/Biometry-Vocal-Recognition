import torch
import argparse
from tqdm import tqdm

from dataset import get_dl_train, get_dl_test, get_dl_val, get_dl_db
from database import init_empty, add, get_similar
from loggers import main_logger as logger
from train import train


def action_train():
    dl_train = get_dl_train('data/inputs', device, 16, PreprocessorTrain)
    dl_test = get_dl_test('data/inputs', device, 64, PreprocessorTest)
    dl_val = get_dl_val('data/inputs', device, 64, PreprocessorTest)
    model = Model(device)
    train(model, 50, 'arc_face_init', device, dl_train, dl_test, dl_val, None, True)


def action_init_db():
    dl_db = get_dl_db('data/inputs', device, 1, PreprocessorTest)
    model = Model(device)
    model.load_model_and_optimizer()
    init_empty(Model.__name__)
    for image, label in tqdm(dl_db):
        embedding = model.get_embedding(image)
        add(label, embedding)


def action_add():
    # get image from 
    # get label from cmd
    label = 'random'
    image = torch.rand((128, 128))
    image = PreprocessorTest.preprocess(image)
    image = image.to(device)
    model = Model(device)
    model.load_model_and_optimizer()
    embedding = model.get_embedding(image)
    add(label, embedding)


def action_auth():
    # get image from 
    # get label from cmd
    label = 'random'
    image = torch.rand((128, 128))
    image = PreprocessorTest.preprocess(image)
    image = image.to(device)
    model = Model(device)
    model.load_model_and_optimizer()
    embedding = model.get_embedding(image)
    similar = get_similar(embedding, threshold=0.1)
    for s in similar:
        if s == label:
            logger.info('Identity confirmed')
            return
    logger.error('Identity unknown')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['arc', 'deep', 'ghost', 'insight'], required=True)
    parser.add_argument('-a', '--action', choices=['train', 'init_db', 'add', 'auth'], required=True)
    args = parser.parse_args()
    
    if args.model == 'arc':
        try:
            from models.arc_face import ArcFaceModel as Model
            from models.arc_face import ArcFacePreprocessorTrain as PreprocessorTrain
            from models.arc_face import ArcFacePreprocessorTest as PreprocessorTest
        except Exception as ex:
            raise ImportError(f'Cannot import wrapper for ArcFace model: {ex}')
    elif args.model == 'deep':
        try:
            from models.deep_face import DeepFaceModel as Model
            from models.deep_face import DeepFacePreprocessor as PreprocessorTrain
            from models.deep_face import DeepFacePreprocessor as PreprocessorTest
        except Exception as ex:
            raise ImportError(f'Cannot import wrapper for DeepFace model: {ex}')
    elif args.model == 'ghost':
        try:
            from models.ghost_face import GhostFaceModel as Model
            from models.ghost_face import GhostFacePreprocessor as PreprocessorTrain
            from models.ghost_face import GhostFacePreprocessor as PreprocessorTest
        except Exception as ex:
            raise ImportError(f'Cannot import wrapper for GhostFace model: {ex}')
    elif args.model == 'insight':
        try:
            from models.insight_face import InsightFaceModel as Model
            from models.insight_face import InsightFacePreprocessor as PreprocessorTrain
            from models.insight_face import InsightFacePreprocessor as PreprocessorTest
        except Exception as ex:
            raise ImportError(f'Cannot import wrapper for InsightFace model: {ex}')
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')
    logger.info(f'Using device {device}')

    if args.action == 'train': action_train()
    elif args.action == 'init_db': action_init_db()
    elif args.action == 'add': action_add()
    elif args.action == 'auth': action_auth()
