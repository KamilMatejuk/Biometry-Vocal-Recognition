import os
import torch
import random
from PIL import Image

from loggers import dataset_logger as logger
from models import Preprocessor


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, name: str, device: str, labels: list[int], preprocessor: Preprocessor | None):
        self.dir = dir
        self.name = name
        self.device = device
        self.labels = labels
        self.preprocessor = preprocessor
        self.data = []
        with open(os.path.join(self.dir, 'identities.txt')) as f:
            for line in f.readlines():
                self._load_labels_line(line)

    def _load_labels_line(self, line: str):
        img_name = line.split(' ')[0].strip()
        label = int(line.split(' ')[1])
        # check if correct dataset
        if label not in self.labels: return
        # check existance
        assert os.path.exists(os.path.join(self.dir, 'images_resized', img_name)),\
            logger.error(f'Couldn\'t find {img_name}')
        # save reference to memmory
        self.data.append((img_name, label))

    def __getitem__(self, i: int):
        img_name, label = self.data[i]
        img = Image.open(os.path.join(self.dir, 'images_resized', img_name))
        if self.preprocessor is not None:
            img = self.preprocessor.preprocess(img)
        label = torch.tensor(label).long()
        return img.to(self.device), label.to(self.device)
    
    def __len__(self):
        return len(self.data)
    
    def stats(self):
        images = len(self.data)
        labels = len(set(label for _, label in self.data))
        logger.debug(f'{self.name} has {images} images of {labels} celebrities (avg {images/labels:.2f} img/celeb)')


def partition(root_dir: str, ratios: list[float]):
    logger.info('Load and partition dataset into: ' + ' '.join(map(lambda x: f'{x:.0%}', ratios)))
    ratio_train, ratio_test, ratio_val, ratio_db = ratios
    if abs(1.0 - sum(ratios)) > 0.01:
        msg = f'All test ratios should sum up to 1.0, not {sum(ratios)}'
        logger.error(msg)
        raise ValueError(msg)
    # gather labels
    labels = set()
    with open(os.path.join(root_dir, 'identities.txt')) as f:
        for line in f.readlines():
            labels.add(int(line.split(' ')[1]))
    # create set sizes
    k_train = int(len(labels) * ratio_train)
    k_test = int(len(labels) * ratio_test)
    k_val = int(len(labels) * ratio_val)
    # create label sets
    labels_train = set(random.sample(list(labels), k=k_train))
    labels -= labels_train
    labels_test = set(random.sample(list(labels), k=k_test))
    labels -= labels_test
    labels_val = set(random.sample(list(labels), k=k_val))
    labels -= labels_val
    labels_db = labels
    # save labels
    torch.save(labels_train, os.path.join(root_dir, 'partition_train.lst'))
    torch.save(labels_test, os.path.join(root_dir, 'partition_test.lst'))
    torch.save(labels_val, os.path.join(root_dir, 'partition_val.lst'))
    torch.save(labels_db, os.path.join(root_dir, 'partition_db.lst'))


def get_dl(root_dir: str, device: str, stage: str, bs: int, shuffle: bool, preprocessor: Preprocessor | None):
    labels = torch.load(os.path.join(root_dir, f'partition_{stage.lower()}.lst'))
    dataset = CelebADataset(root_dir, stage, device, labels, preprocessor)
    dataset.stats()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle)
    return dataloader


def get_dl_train(root_dir: str, device: str, bs: int, preprocessor: Preprocessor | None):
    return get_dl(root_dir, device, 'Train', bs, False, preprocessor)
def get_dl_test(root_dir: str, device: str, bs: int, preprocessor: Preprocessor | None):
    return get_dl(root_dir, device, 'Test', bs, False, preprocessor)
def get_dl_val(root_dir: str, device: str, bs: int, preprocessor: Preprocessor | None):
    return get_dl(root_dir, device, 'Val', bs, False, preprocessor)
def get_dl_db(root_dir: str, device: str, bs: int, preprocessor: Preprocessor | None):
    return get_dl(root_dir, device, 'Db', bs, False, preprocessor)


if __name__ == '__main__':
    partition('data/inputs', [0.69, 0.15, 0.15, 0.01])
