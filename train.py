import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable

from .models import Model


# TODO define metrics and how to measure them


def _load_logs(name: str) -> tuple[int, pd.DataFrame]:
    try:
        logs = pd.read_csv(f'data/metrics/{name}.csv')
        start_epoch = int(logs['epoch'].max()) + 1
        print(f'Loaded existing logs for {start_epoch} epochs')
    except:
        logs = pd.DataFrame(columns=['epoch', 'loss train', 'loss val'])
        start_epoch = 0
        print('Failed loading existing logs')
    return start_epoch, logs


def _load_checkpoint(start_epoch: int, pretrained_weights: str, name: str, device: str, model: Model) -> None:
    try:
        if start_epoch == 0 and pretrained_weights is not None:
            checkpoint_file = f'data/checkpoints/{pretrained_weights}'
        else: checkpoint_file = f'data/checkpoints{name}.chpt'
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
        model.load_model_and_optimizer(checkpoint)
        print('Loaded existing checkpoints')
    except:
        print('Failed loading existing checkpoints')
    return model


def _train(device: str, model: Model) -> float:
    loss_train_per_epoch = []
    for image, label in tqdm(model.train_dl, leave=False):
        image = image.to(device)
        label = label.to(device)
        pred = model.get_classification(image, label)
        loss = model.get_loss(pred, label)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        loss_train_per_epoch.append(loss.item())
    return float(np.mean(loss_train_per_epoch))


def _test(device: str, model: Model) -> float:
    metrics_per_epoch = []
    for image, label in tqdm(model.test_dl, leave=False):
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model.get_classification(image, label)
            # pred = pred.detach().cpu().numpy()
            # pred = np.argmax(pred, axis = 1)
            # precision_per_epoch.append(precision_score(label, pred, average='weighted'))
            # recall_per_epoch.append(recall_score(label, pred, average='weighted'))
            # f1_per_epoch.append(f1_score(label, pred, average='weighted'))
        metrics_per_epoch.append(0)
    return float(np.mean(metrics_per_epoch))


def _validate(device: str, model: Model) -> float:
    loss_val_per_epoch = []
    for image, label in tqdm(model.val_dl, leave=False):
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model.get_classification(image, label)
            loss = model.get_loss(pred, label)
        loss_val_per_epoch.append(loss.item())
    return float(np.mean(loss_val_per_epoch))


def _is_early_stop(prev_losses: list[float], cur_loss: float, max_epochs_without_improvement: int) -> tuple[bool, int]:
    val_losses = prev_losses + [cur_loss]
    if len(val_losses) <= max_epochs_without_improvement: return False, -1
    best_loss = min(val_losses)
    best_loss_idx = val_losses.index(best_loss)
    epochs_without_improvement = len(val_losses) - best_loss_idx - 1
    return epochs_without_improvement > max_epochs_without_improvement, best_loss_idx


def _save_logs(epochs: int, name: str, curr_epoch: int, early_stop: bool, best_idx: int,
               logs: pd.DataFrame, train_loss: float, val_loss: float, test_metrics: float) -> pd.DataFrame:
    logs = pd.concat([logs, pd.DataFrame([{
        'epoch': curr_epoch,
        'loss train': train_loss,
        'loss val': val_loss,
        'test metrics': test_metrics,
    }])])
    logs.to_csv(f'data/metrics/{name}.csv', index=False)
    # duplicate for early stop 
    if early_stop:
        empty_epochs = list(range(curr_epoch + 1, epochs))
        if len(empty_epochs) == 0: return logs
        last_log = logs.iloc[best_idx]
        duplicated_logs = pd.concat([last_log.to_frame().T] * len(empty_epochs), ignore_index=True)
        duplicated_logs['epoch'] = empty_epochs
        logs = pd.concat([logs, duplicated_logs], ignore_index=True) 
        logs.to_csv(f'data/metrics/{name}.csv', index=False)
    return logs


def _save_checkpoints(epochs: int, name: str, curr_epoch: int, early_stop: bool, best_idx: int,
                      model: Model, save_per_epoch: bool):
    # save epoch intermediate checkpoint
    if save_per_epoch:
        model.save_model_and_optimizer(f'data/checkpoints/{name}.{curr_epoch + 1:>03}.chpt')
    # save main checkpoint
        model.save_model_and_optimizer(f'data/checkpoints/{name}.chpt')
    # duplicate for early stop 
    if early_stop and save_per_epoch:
        empty_epochs = list(range(curr_epoch + 1, epochs))
        for e in empty_epochs:
            os.system(f'cp data/checkpoints/{name}.{best_idx+1:>03}.chpt data/checkpoints/{name}.{e:>03}.chpt')

    
def train(model: Model, epochs: int, name: str, device: str, pretrained_weights: str = None, save_per_epoch: bool = False):
    # create dirs
    os.makedirs('data/metrics', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)
    # load history
    start_epoch, logs = _load_logs(name)
    if start_epoch >= epochs:
        print(f'Found finished training, skipping')
        return
    _load_checkpoint(start_epoch, pretrained_weights, name, device, model)
    # train
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=10, gamma=0.1)
    for epoch in tqdm(range(epochs)):
        if epoch < start_epoch: continue
        scheduler.step()
        train_loss = _train(device, model)
        test_metrics = _test(device, model)
        val_loss = _validate(device, model)
        # early stop
        early_stop, best_idx = _is_early_stop(list(logs['loss val']), val_loss, 3)
        logs = _save_logs(epochs, name, epoch, early_stop, best_idx, logs, train_loss, val_loss, test_metrics)
        _save_checkpoints(epochs, name, epoch, early_stop, best_idx, model, save_per_epoch)
        if early_stop:
            print(f'Early stop after {epoch} epochs')
            break
