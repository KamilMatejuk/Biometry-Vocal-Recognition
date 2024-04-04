import yaml
import torch
import argparse
from tqdm import tqdm

from dataset import get_dl_train, get_dl_test, get_dl_val, get_dl_db
from database import init_empty, add, get_similar
from loggers import main_logger as logger
from train import train


def action_train(model_name: str, config: dict):
    dl_train = get_dl_train('data/inputs', device, config['bs'], PreprocessorTrain)
    dl_test = get_dl_test('data/inputs', device, config['bs'], PreprocessorTest)
    dl_val = get_dl_val('data/inputs', device, config['bs'], PreprocessorTest)
    model = Model(device, config)
    train(model, 1000, f'{model}/{model_name}', device, dl_train, dl_test, dl_val, None, True)


def action_init_db(model_name: str, config: dict):
    dl_db = get_dl_db('data/inputs', device, 1, PreprocessorTest)
    model = Model(device, config)
    model.load_model_and_optimizer(f'data/checkpoints/{model}/{model_name}.chpt')
    init_empty(f'{model}/{model_name}')
    for image, label in tqdm(dl_db):
        embedding = model.get_embedding(image)
        add(label, embedding)


def action_add(model_name: str, config: dict):
    # get image from 
    # get label from cmd
    label = 'random'
    image = torch.rand((128, 128))
    image = PreprocessorTest.preprocess(image)
    image = image.to(device)
    model = Model(device, config)
    model.load_model_and_optimizer(f'data/checkpoints/{model}/{model_name}.chpt')
    embedding = model.get_embedding(image)
    add(f'{model}/{model_name}', label, embedding)


def action_auth(model_name: str, config: dict):
    # get image from 
    # get label from cmd
    label = 'random'
    image = torch.rand((128, 128))
    image = PreprocessorTest.preprocess(image)
    image = image.to(device)
    model = Model(device, config)
    model.load_model_and_optimizer(f'data/checkpoints/{model}/{model_name}.chpt')
    embedding = model.get_embedding(image)
    similar = get_similar(f'{model}/{model_name}', embedding, threshold=0.1)
    for s in similar:
        if s == label:
            logger.info('Identity confirmed')
            return
    logger.error('Identity unknown')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['arc', 'deep', 'ghost', 'insight'], required=True)
    parser.add_argument('-n', '--name', default='init')
    parser.add_argument('-a', '--action', choices=['train', 'init_db', 'add', 'auth'], required=True)
    parser.add_argument('-c', '--config', default='config.yml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = config.get(args.model, {})
    config = {**config.get('all', {}), **config.get(args.name, {})}
    logger.info(f'Loaded config from {args.config}')
    logger.debug(config)
    
    if args.model == 'arc':
        try:
            from models.arc_face import ArcFaceModel as Model
            from models.arc_face import ArcFacePreprocessorTrain as PreprocessorTrain
            from models.arc_face import ArcFacePreprocessorTest as PreprocessorTest
        except Exception as ex:
            logger.exception(f'Cannot import wrapper for ArcFace model: {ex}')
            exit(1)
    elif args.model == 'deep':
        try:
            from models.deep_face import DeepFaceModel as Model
            from models.deep_face import DeepFacePreprocessor as PreprocessorTrain
            from models.deep_face import DeepFacePreprocessor as PreprocessorTest
        except Exception as ex:
            logger.exception(f'Cannot import wrapper for DeepFace model: {ex}')
            exit(1)
    elif args.model == 'ghost':
        try:
            from models.ghost_face import GhostFaceModel as Model
            from models.ghost_face import GhostFacePreprocessorTrain as PreprocessorTrain
            from models.ghost_face import GhostFacePreprocessorTest as PreprocessorTest
        except Exception as ex:
            logger.exception(f'Cannot import wrapper for GhostFace model: {ex}')
            exit(1)
    elif args.model == 'insight':
        try:
            from models.insight_face import InsightFaceModel as Model
            from models.insight_face import InsightFacePreprocessor as PreprocessorTrain
            from models.insight_face import InsightFacePreprocessor as PreprocessorTest
        except Exception as ex:
            logger.exception(f'Cannot import wrapper for InsightFace model: {ex}')
            exit(1)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')
    # device = torch.device('cpu')
    logger.info(f'Using device {device}')

    if args.action == 'train': action_train(args.name, config)
    elif args.action == 'init_db': action_init_db(args.name, config)
    elif args.action == 'add': action_add(args.name, config)
    elif args.action == 'auth': action_auth(args.name, config)
