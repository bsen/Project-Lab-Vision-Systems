import numpy as np
import torch
from torchvision.utils import save_image
import os
from .loss_functions import three_pixel_err, smoothL1
import time
import torch.cuda.amp as amp
import torch.utils.tensorboard as tb

import sys
sys.path.insert(0, '../')
from my_utils import device, base_path

f_smoothL1 = smoothL1(1.0)

def train_model(model, optimizer, scheduler, train_loader,
                num_epochs, log_dir,
                valid_loader=None, savefile=None, mes_time=False,
                pretrain_optimizer=None, pretrain_scheduler=None,
                pretrain_loader=None, pretrain_epochs=None, use_amp=True,
                show_graph=False):
    """
    Training a model for a given number of epochs.

    :param model: The model this function trains
    :param optimizer: The optimizer that gets used for training
    :param scheduler: The learning rate scheduler that gets used
    :param train_loader: The DataLoader for the training dataset
    :param valid_loader: The DataLoader for the validation dataset (optional).
                         If this is None, also no other losses get stored.
    :param num_epochs: The number of epochs we train for
    :param log_dir: The directory, tensorboard should log to
                    (in the folders runs/train/ and runs/pretrain/ )
    :param savefile: If given, the model, training loss, validation loss,
                     the loss in the different iterations and the measured
                     time get stored in the file savefile.
    :param mes_time: If True, the time how long the training procedure takes gets
                     measured and returned (this measures only a rough approximation)
    :param pretrain_optimizer: The optimizer used for pretraining (optional)
    :param pretrain_scheduler: The scheduler used for pretraining (optional)
    :param pretrain_loader: The DataLoader for the pretrain dataset (optional)
    :param pretrain_epochs: The number of epochs for pretraining (optional)
    :param use_amp: Determines whether or not to use automatic mixed precision
                    for training the network (as explained here:
                    https://pytorch.org/docs/stable/notes/amp_examples.html)
    :param show_graph: whether or not to show the network graph in the tensor board
    """

    writer = tb.SummaryWriter(os.path.join(base_path, 'runs/train/', log_dir))

    if show_graph:
        sample = next(iter(train_loader))
        left = sample[0].to(device)
        right = sample[1].to(device)
        writer.add_graph(model, (left, right))

    if mes_time:
        start = time.time()

    if pretrain_loader is not None:
        pre_writer = tb.SummaryWriter(os.path.join('runs/pretrain/', log_dir))
        pre_losses = _train_model_no_time(model, pretrain_optimizer, pretrain_scheduler,
                             pretrain_loader, valid_loader, pretrain_epochs,
                             use_amp=use_amp, writer=pre_writer)
    else:
        pre_losses = None
    losses = _train_model_no_time(model, optimizer, scheduler,
                         train_loader, valid_loader, num_epochs,
                         use_amp=use_amp, writer=writer)

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
            if pre_losses is None:
                torch.save(
                    {
                        'model': model,
                        'losses': losses
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
        if pre_losses is None:
            return losses, time_taken
        else:
            return pre_losses, losses, time_taken
    return time_taken


def _train_model_no_time(model, optimizer, scheduler, train_loader,
                         valid_loader, num_epochs, use_amp, writer):
    """Train the model not measuring time"""
    keep_loss = valid_loader is not None

    scaler = amp.GradScaler(enabled=use_amp)

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
        if keep_loss:
            v_loss, v_err = _eval_model(model=model, valid_loader=valid_loader)
            val_loss.append(v_loss)
            val_err.append(v_err)
            writer.add_scalar('validation loss', v_loss)
            writer.add_scalar('validation error', v_err)

        # training epoch
        model.train()  # important for dropout and batch norms
        loss_list = _train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                keep_loss=keep_loss, scaler=scaler, use_amp=use_amp
            )
        scheduler.step()

        if keep_loss:
            mean_loss = np.mean(loss_list)
            print(f' train loss: {mean_loss}')
            train_loss.append(mean_loss)
            loss_iters += loss_list
            writer.add_scalar('training loss', mean_loss)

    print("\nTraining completed")

    if keep_loss:
        return train_loss, loss_iters, val_loss, val_err
    return None


def _train_epoch(model, train_loader, optimizer, keep_loss, scaler, use_amp):
    """ Training a model for one epoch """
    if keep_loss:
        loss_list = []

    for (left, right, true_disp) in train_loader:
        left = left.to(device)
        right = right.to(device)
        true_disp = true_disp.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass and calculate loss
        with amp.autocast(enabled=use_amp):
            pred_disp = model(left, right)
            loss = f_smoothL1(true_disp, pred_disp)

        if keep_loss:
            loss_list.append(loss.item())

        # Getting gradients w.r.t. parameters
        scaler.scale(loss).backward()
        # Updating parameters
        scaler.step(optimizer)
        # Update the scale
        scaler.update()


    if keep_loss:
        return loss_list

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
    print(f' eval loss: {loss}, eval err: {err}')

    return loss, err
