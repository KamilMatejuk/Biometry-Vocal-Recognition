import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)

from models import Model
from loggers import train_logger as logger


def _load_logs(name: str) -> tuple[int, pd.DataFrame]:
    logs = pd.DataFrame(columns=['epoch', 'loss train', 'loss val'])
    start_epoch = 0
    try:
        logs = pd.read_csv(f'data/metrics/{name}.csv')
        start_epoch = int(logs['epoch'].max()) + 1
        logger.debug(f'Loaded existing logs for {start_epoch} epochs')
    except FileNotFoundError:
        logger.warn(f'Failed loading existing logs: File not found')
    except Exception as ex:
        logger.exception(f'Failed loading existing logs: {ex}')
        exit(1)
    return start_epoch, logs


def _load_checkpoint(start_epoch: int, pretrained_weights: str, name: str, device: str, model: Model) -> None:
    try:
        if start_epoch == 0 and pretrained_weights is not None:
            checkpoint_file = f'data/checkpoints/{pretrained_weights}'
        else: checkpoint_file = f'data/checkpoints/{name}.chpt'
        model.load_model_and_optimizer(checkpoint_file)
        logger.debug('Loaded existing checkpoints')
    except FileNotFoundError:
        logger.warn(f'Failed loading existing checkpoints: File not found')
    except Exception as ex:
        logger.exception(f'Failed loading existing checkpoints: {ex}')
        exit(1)
    return model


def _train(model: Model, dl: torch.utils.data.DataLoader) -> float:
    loss_train_per_epoch = []
    for image, label in tqdm(dl, leave=False):
        pred = model.get_classification(image, label)
        loss = model.get_loss(pred, label)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        loss_train_per_epoch.append(loss.item())
    return float(np.mean(loss_train_per_epoch))


def _test(model: Model, dl: torch.utils.data.DataLoader) -> dict[str, float]:
    f1_per_epoch = []
    accuracy_per_epoch = []
    for image, label in tqdm(dl, leave=False):
        with torch.no_grad():
            pred = model.get_classification(image, label)
            label = label.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            pred = np.argmax(pred, axis = 1)
            f1_per_epoch.append(f1_score(label, pred, average='micro'))
            accuracy_per_epoch.append(accuracy_score(label, pred))
    return {
        'f1': float(np.mean(f1_per_epoch)),
        'accuracy': float(np.mean(accuracy_per_epoch)),
    }

 
def _validate(model: Model, dl: torch.utils.data.DataLoader) -> float:
    loss_val_per_epoch = []
    for image, label in tqdm(dl, leave=False):
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
               logs: pd.DataFrame, train_loss: float, val_loss: float, test_metrics: dict[str, float]) -> pd.DataFrame:
    logs = pd.concat([logs, pd.DataFrame([{
        'epoch': curr_epoch,
        'loss train': train_loss,
        'loss val': val_loss,
        **test_metrics,
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

    
def train(model: Model, epochs: int, name: str, device: str, 
          train_dl: torch.utils.data.DataLoader,
          test_dl: torch.utils.data.DataLoader, 
          val_dl: torch.utils.data.DataLoader, 
          pretrained_weights: str = None,
          save_per_epoch: bool = False):
    # create dirs
    logger.info('Create dirs for data')
    os.makedirs('data/metrics', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)
    # load history
    logger.info('Load history')
    start_epoch, logs = _load_logs(name)
    if start_epoch >= epochs:
        logger.debug(f'Found finished training, skipping')
        return
    _load_checkpoint(start_epoch, pretrained_weights, name, device, model)
    # train
    logger.info('Start training')
    # scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=10, gamma=0.1)
    for epoch in tqdm(range(epochs)):
        if epoch < start_epoch: continue
        train_loss = _train(model, train_dl)
        test_metrics = _test(model, test_dl)
        val_loss = _validate(model, val_dl)
        # scheduler.step()
        # early stop
        # early_stop, best_idx = _is_early_stop(list(logs['loss val']), val_loss, 3)
        early_stop, best_idx = False, -1
        logs = _save_logs(epochs, name, epoch, early_stop, best_idx, logs, train_loss, val_loss, test_metrics)
        _save_checkpoints(epochs, name, epoch, early_stop, best_idx, model, save_per_epoch)
        if early_stop:
            logger.debug(f'Early stop after {epoch} epochs')
            break
    logger.info(f'Finish training with {logs.iloc[-1]["accuracy"]:.2%} accuracy')
