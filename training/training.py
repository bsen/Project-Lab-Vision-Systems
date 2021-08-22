import numpy as np
import torch
from torchvision.utils import save_image
import os
from .loss_functions import three_pixel_err, smoothL1
import time

import sys
sys.path.insert(0, '../')
from my_utils import device, base_path

f_smoothL1 = smoothL1(1.0)

def train_model(model, optimizer, scheduler, train_loader,
                num_epochs, valid_loader=None, savefile=None, mes_time=False,
                pretrain_optimizer=None, pretrain_scheduler=None,
                pretrain_loader=None, pretrain_epochs=None):
    """
    Training a model for a given number of epochs.

    :param model: The model this function trains
    :param optimizer: The optimizer that gets used for training
    :param scheduler: The learning rate scheduler that gets used
    :param train_loader: The DataLoader for the training dataset
    :param valid_loader: The DataLoader for the validation dataset (optional).
                         If this is None, also no other losses get stored.
    :param num_epochs: The number of epochs we train for
    :param savefile: If given, the model, training loss, validation loss,
                     the loss in the different iterations and the measured
                     time get stored in the file savefile.
    :param mes_time: If True, the time how long the training procedure takes gets
                     measured and returned (this measures only a rough approximation)
    :param pretrain_optimizer: The optimizer used for pretraining (optional)
    :param pretrain_scheduler: The scheduler used for pretraining (optional)
    :param pretrain_loader: The DataLoader for the pretrain dataset (optional)
    :param pretrain_epochs: The number of epochs for pretraining (optional)
    """

    if mes_time:
        start = time.time()

    if pretrain_loader is not None:
        pre_losses = _train_model_no_time(model, pretrain_optimizer, pretrain_scheduler,
                             pretrain_loader, valid_loader, pretrain_epochs);
    losses = _train_model_no_time(model, optimizer, scheduler,
                         train_loader, valid_loader, num_epochs)

    if mes_time:
        end = time.time()
        time_taken = end-start
    else:
        time_taken = None


    if savefile is not None:
        path = os.path.join(base_path, savepath)
        if valid_loader is None:
            torch.save(
                {
                    'model': model,
                },
                f=path)
        else:
            torch.save(
                {
                    'model': model,
                    'pre_losses': pre_losses,
                    'losses': losses
                },
                f=path)

    if valid_loader is not None:
        return pre_losses, losses, time_taken
    return time_taken


def _train_model_no_time(model, optimizer, scheduler, train_loader,
                         valid_loader, num_epochs):
    """Train the model not measuring time"""
    keep_loss = valid_loader is not None

    if keep_loss:
        train_loss = []
        loss_iters = []
        val_loss =  []
        val_err = []
    print('Epoch:')
    for epoch in range(num_epochs):
        print(str(epoch), end=', ')

        # validation epoch
        model.eval()  # important for dropout and batch norms
        if keep_loss and (epoch % 5 == 0 or epoch == num_epochs - 1):
            loss_err = _eval_model(model=model, valid_loader=valid_loader)
            val_loss.append(loss_err[0])
            val_err.append(loss_err[1])

        # training epoch
        model.train()  # important for dropout and batch norms
        mean_list = _train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                keep_loss=keep_loss
            )
        scheduler.step()

        if keep_loss:
            train_loss.append(mean_list[0])
            loss_iters += mean_list[1]

    print("\nTraining completed")

    if keep_loss:
        return train_loss, loss_iters, val_loss, val_err
    return None


def _train_epoch(model, train_loader, optimizer, keep_loss):
    """ Training a model for one epoch """
    if keep_loss:
        loss_list = []

    for (left, right, true_disp) in train_loader:
        left = left.to(device)
        right = right.to(device)
        true_disp = true_disp.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass
        pred_disp = model(left, right)

        # Calculate Loss
        loss = f_smoothL1(true_disp, pred_disp)
        if keep_loss:
            loss_list.append(loss.item())

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

    if keep_loss:
        mean_loss = np.mean(loss_list)

        return mean_loss, loss_list

    return None


@torch.no_grad()
def _eval_model(model, valid_loader):
    """ Evaluating the model for either validation or test """
    loss_list = []
    err_list = []

    for i, (left, right, true_disp) in enumerate(valid_loader):
        left = left.to(device)
        right = right.to(device)
        true_disp = true_disp.to(device)

        # Forward pass
        pred_disp = model(left, right)

        loss = f_smoothL1(true_disp, pred_disp)
        err = three_pixel_err(true_disp, pred_disp)
        loss_list.append(loss.item())
        err_list.append(err.item())

    # Total correct predictions and loss
    loss = np.mean(loss_list)
    err = np.mean(err_list)

    return loss, err
