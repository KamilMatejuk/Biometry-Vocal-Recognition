import torch

from models.arc_face import ArcFaceModel, ArcFacePreprocessorTrain, ArcFacePreprocessorTest
from train import train
from dataset import partition
from loggers import main_logger as logger


if __name__ == '__main__':
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')
    logger.info(f'Using device {device}')

    set_train, set_test, set_val, _ = partition(
        'data/inputs', device, [0.7, 0.1, 0.1, 0.1],
        [ArcFacePreprocessorTrain] + [ArcFacePreprocessorTest] * 3)
    model = ArcFaceModel(device)
    dl_train = torch.utils.data.DataLoader(set_train, batch_size=16, shuffle=True)
    dl_test = torch.utils.data.DataLoader(set_test, batch_size=16, shuffle=False)
    dl_val = torch.utils.data.DataLoader(set_val, batch_size=16, shuffle=False)
    train(model, 10, 'arc_face_init', device, dl_train, dl_test, dl_val, None, True)
    