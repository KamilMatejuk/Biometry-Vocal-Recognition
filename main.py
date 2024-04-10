import yaml
import torch
import argparse
from tqdm import tqdm

from dataset import get_dl_train, get_dl_test, get_dl_val, get_dl_db_ui_ii
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
    dl_db = get_dl_db_ui_ii('data/inputs', device, 1, PreprocessorTest)
    model = Model(device, config)
    model.load_model_and_optimizer(f'data/checkpoints/{model}/{model_name}.chpt')
    counter = 0
    init_empty(f'{model}/{model_name}')
    for image, label, _ in tqdm(dl_db):
        embedding = model.get_embedding(image).cpu().detach().numpy()
        label = f'user_{label.item()}'
        add(f'{model}/{model_name}', label, embedding)
        counter += 1
    logger.info(f'Added {counter} users')


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
    parser.add_argument('-m', '--model', choices=['example'], required=True, help='Name of model to use, e.g. example')
    parser.add_argument('-n', '--name', default='init', help='Name of specific experiment')
    parser.add_argument('-a', '--action', choices=['train', 'init_db', 'add', 'auth'], required=True, help='What to do?')
    parser.add_argument('-c', '--config', default='config.yml', help='YAML file with configuration')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = config.get(args.model, {})
    config = {**config.get('all', {}), **config.get(args.name, {})}
    logger.info(f'Loaded config from {args.config}')
    logger.debug(config)
    
    if args.model == 'example':
        try:
            from models.example import ExampleModel as Model
            from models.example import ExamplePreprocessorTrain as PreprocessorTrain
            from models.example import ExamplePreprocessorTest as PreprocessorTest
        except Exception as ex:
            logger.exception(f'Cannot import wrapper for GhostFace model: {ex}')
            exit(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cuda')
    # device = torch.device('cpu')
    logger.info(f'Using device {device}')

    if args.action == 'train': action_train(args.name, config)
    elif args.action == 'init_db': action_init_db(args.name, config)
    elif args.action == 'add': action_add(args.name, config)
    elif args.action == 'auth': action_auth(args.name, config)
