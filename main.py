import torch

from models.arc_face import ArcFaceModel, ArcFacePreprocessorTrain, ArcFacePreprocessorTest
from dataset import get_dl_train, get_dl_test, get_dl_val
from loggers import main_logger as logger
from train import train


if __name__ == '__main__':
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')
    logger.info(f'Using device {device}')

    dl_train = get_dl_train('data/inputs', device, 16, ArcFacePreprocessorTrain)
    dl_test = get_dl_test('data/inputs', device, 64, ArcFacePreprocessorTest)
    dl_val = get_dl_val('data/inputs', device, 64, ArcFacePreprocessorTest)
    model = ArcFaceModel(device)
    train(model, 10, 'arc_face_init', device, dl_train, dl_test, dl_val, None, True)
    