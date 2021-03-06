import numpy as np
import torch
from torchvision.utils import save_image
import os
from .loss_functions import three_pixel_err, smoothL1
import time
import datetime
import torch.cuda.amp as amp
import torch.utils.tensorboard as tb
from ray import tune

import sys
sys.path.insert(0, '../')
from my_utils import device, base_path

f_smoothL1 = smoothL1(1.0)


def train_model(model, optimizer, scheduler, train_loader,
                num_epochs, log_dir,
                init_epoch=0,
                valid_loader=None, savefile=None, mes_time=False,
                use_amp=True, show_graph=False, use_tune=False, 
                tune_interval=3,
                warmup_lr=False, sched_iters=None):
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
                    (in the folders runs/log_dir/)
                    If set to None, tensorboard will not log
    :param init_epoch: The number of epochs this model was already trained 
                       (important when we load a raytune checkpoint and want to train it)
    :param savefile: If given, the model, training loss, validation loss and
                     the losses in the different iterations
                     get stored in the file savefile.
    :param mes_time: If True, the time how long the training procedure takes gets
                     measured and printed to stdout
                     (this measures only a rough approximation)
    :param use_amp: Determines whether or not to use automatic mixed precision
                    for training the network (as explained here:
                    https://pytorch.org/docs/stable/amp.html)
    :param use_tune: set to True when this function is called by raytune for hyperparameter optimization 
                     (then training stores checkpoints and reports correctly)
    :param tune_interval: if use_tune is True, raytune stores the configuration and reports every
                          tune_interval-th epoch.
    :param warmup_lr: If True, scheduler.step() gets called before the optimizer gets called
                      and some other things are done a bit differently
                      (needed for the warmup learning implementation by Le)
    :param sched_iters: Determines after how many iterations the scheduler should be called
                        in each epoch. 
                        E.g. if sched_iters=4, scheduler.step() will be called
                        in the first iteration, 4-th iteration, 8-th iteration, ... each epoch.
                        If this is not set, the scheduler gets called once per epoch.
                        Can only be used with warmup_lr.
                        (This is needed for Scene Flow, since the epochs there contain many elements.)
    """
    assert (sched_iters is None) or warmup_lr
    assert (sched_iters is None) or (sched_iters < len(train_loader))
    
    if log_dir is None:
        writer = None
    else:
        writer = tb.SummaryWriter(os.path.join(base_path, 'runs/', log_dir))

    if show_graph:
        sample = next(iter(train_loader))
        left = sample[0].to(device)
        right = sample[1].to(device)
        writer.add_graph(model, (left, right))

    if mes_time:
        start = time.time()

    losses = _train_model_no_time(model, optimizer, scheduler,
                         train_loader, valid_loader, num_epochs,
                         init_epoch=init_epoch,
                         use_amp=use_amp, writer=writer, use_tune=use_tune,
                         tune_interval=tune_interval,
                         warmup_lr=warmup_lr, sched_iters=sched_iters)

    if mes_time:
        torch.cuda.synchronize()
        end = time.time()
        time_taken = end-start
        print(f'Time taken for training: {str(datetime.timedelta(seconds=time_taken))}')

    if savefile is not None:
        path = os.path.join(base_path, savefile)
        if valid_loader is None:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                },
                f=path)
        else:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'losses': losses
                },
                f=path)

    return losses


def _train_model_no_time(model, optimizer, scheduler, train_loader,
                         valid_loader, num_epochs, init_epoch, use_amp, writer,
                         use_tune, tune_interval, warmup_lr, sched_iters):
    """ Train the model not measuring time """
    keep_loss = valid_loader is not None

    scaler = amp.GradScaler(enabled=use_amp)

    if keep_loss:
        train_loss = []
        loss_iters = []
        val_loss =  []
        val_err = []
    
    if warmup_lr:
        optimizer.zero_grad()
        optimizer.step()
        
    print('Epoch:')
    for epoch in range(init_epoch, num_epochs):
        print(str(epoch), end=', ')

        # training epoch
        model.train()  # important for dropout and batch norms
        
        loss_list = _train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                scheduler=scheduler, warmup_lr=warmup_lr,
                keep_loss=keep_loss, scaler=scaler, use_amp=use_amp,
                sched_iters=sched_iters
            )
        
        if keep_loss:
            mean_loss = np.mean(loss_list)
            print(f' train loss: {mean_loss}')
            train_loss.append(mean_loss)
            loss_iters += loss_list
            if writer is not None:
                writer.add_scalar('training loss', mean_loss)

        # validation epoch
        model.eval()  # important for dropout and batch norms
        if keep_loss:
            v_loss, v_err = _eval_model(model=model, valid_loader=valid_loader)
            val_loss.append(v_loss)
            val_err.append(v_err)
            if writer is not None:
                writer.add_scalar('validation loss', v_loss)
                writer.add_scalar('validation error', v_err)
                # is see now that i forgot to add the global_step ..
                # I only used this once in the experiments in the report (for pretraining in experiment 3), so 
                # I let it as it was, but the right way of doing it would be:
                # writer.add_scalar('validation loss', v_loss, epoch)
                # and
                # writer.add_scalar('validation error', v_err, epoch)

                
        if use_tune and ((epoch%tune_interval == 0) or (epoch == num_epochs-1)):
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), epoch), path)
            tune.report(loss=v_loss, err=v_err, loss_train=mean_loss, epoch=epoch)

    print("\nTraining completed")

    if keep_loss:
        return train_loss, loss_iters, val_loss, val_err
    return None


def _train_epoch(model, train_loader, optimizer, scheduler, warmup_lr, keep_loss, scaler, use_amp, sched_iters):
    """ Training a model for one epoch """
    if keep_loss:
        loss_list = []
    
    if warmup_lr and (sched_iters is None):
        scheduler.step()

    for i,(left, right, true_disp) in enumerate(train_loader):
        
        if (sched_iters is not None) and i%sched_iters == 0:
            scheduler.step()
            if len(loss_list) >= sched_iters:
                print('train loss:', np.mean(loss_list[-sched_iters:]))
        
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

    if (not warmup_lr) and (sched_iters is None):
        scheduler.step()

    if keep_loss:
        return loss_list

    return None


@torch.no_grad()
def _eval_model(model, valid_loader):
    """ Evaluating the model for either validation or test """
    loss_list = []
    err_list = []

    for left, right, true_disp in valid_loader:
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
