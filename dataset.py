import os
import torch
import random
from PIL import Image
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

from loggers import dataset_logger as logger
from models import Preprocessor


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dir: str, name: str, device: str,
                 data: list[tuple[str, int]],
                 preprocessor: Preprocessor | None):
        self.dir = dir
        self.name = name
        self.device = device
        self.preprocessor = preprocessor
        if data is not None:
            self.data = data
        else:
            self.data = []
            with open(os.path.join(self.dir, 'identities.txt')) as f:
                for line in f.readlines():
                    self._load_labels_line(line)
        # ids = sorted(list(map(lambda x: x[1], self.data)))[:100_000]
        # self.data = list(filter(lambda x: x[1] in ids, self.data))

    def _load_labels_line(self, line: str):
        img_name = line.split(' ')[0].strip()
        label = int(line.split(' ')[1])
        # check if correct dataset
        if label not in self.labels: return
        if img_name not in self.imgs: return
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


def partition(root_dir: str):
    logger.info('Load and partition dataset')
    # gather labels
    data = []
    with open(os.path.join(root_dir, 'identities.txt')) as f:
        for line in f.readlines():
            img_name = line.split(' ')[0].strip()
            label = int(line.split(' ')[1])
            data.append((img_name, label))
    # filter at least 10 images per user
    logger.debug(f'Data length before filtering: {len(data)}')
    labels = list(map(lambda x: x[1], data))
    label_counts = Counter(labels)
    filtered_labels = [label for label, count in label_counts.items() if count >= 10]
    data = set(filter(lambda x: x[1] in filtered_labels, data))
    labels = list(map(lambda x: x[1], data))
    logger.debug(f'Data length after filtering: {len(data)}')
    CelebADataset(root_dir, 'all', 'cpu', data, None).stats()
    # exclude db (100 users)
    labels_db = sorted(list(set(labels)))[-100:]
    data_db = list(filter(lambda x: x[1] in labels_db, data))
    torch.save(data_db, os.path.join(root_dir, f'partition_db.data'))
    # remap labels
    data = list(filter(lambda x: x[1] not in labels_db, data))
    old_labels = sorted(list(set(map(lambda x: x[1], data))))
    new_labels = list(range(len(old_labels)))
    label_mapping = {o: n for o, n in zip(old_labels, new_labels)}
    data = [(img, label_mapping[lab]) if lab in label_mapping else None for (img, lab) in data]
    data = list(filter(lambda x: x is not None, data))
    labels = list(map(lambda x: x[1], data))
    # split train test val
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.15)
    train_ids, test_ids = next(splitter.split(data, labels))
    data_train = []
    data_test = []
    data_val = []
    for i, x in enumerate(data):
        if i in train_ids: data_train.append(x)
        elif i in test_ids: data_test.append(x)
        else: data_val.append(x)
    # save labels
    torch.save(data_train, os.path.join(root_dir, f'partition_train_full.data'))
    torch.save(data_test, os.path.join(root_dir, f'partition_test_full.data'))
    torch.save(data_val, os.path.join(root_dir, f'partition_val_full.data'))


def get_dl(root_dir: str, device: str, stage: str, bs: int, shuffle: bool, preprocessor: Preprocessor | None):
    datafile = f'partition_db.data' if stage == 'Db' else f'partition_{stage.lower()}_full.data'
    data = torch.load(os.path.join(root_dir, datafile))
    dataset = CelebADataset(root_dir, stage, device, data, preprocessor)
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
    partition('data/inputs')
    dl_train = get_dl_train('data/inputs', 'cpu', 1, None)
    dl_test = get_dl_test('data/inputs', 'cpu', 1, None)
    dl_val = get_dl_val('data/inputs', 'cpu', 1, None)
    dl_db = get_dl_db('data/inputs', 'cpu', 1, None)