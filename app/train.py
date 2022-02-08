import torch
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np
from tqdm import tqdm
import mlflow
from mlflow import log_metric, log_param, log_artifacts

import config
from utils import *
from model import TSCResNet

def mlflow_setup():

    mlflow.set_tracking_uri(uri = config.DATA_DIR+'mlruns/')
    mlflow.set_experiment(experiment_name = config.EXPERIMENT)

    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    mlflow.start_run()
    run = mlflow.active_run()
    print("Active run_id: {}; status: {}".format(run.info.run_id, run.info.status))

    log_param("config.BATCH_SIZE", config.BATCH_SIZE)
    log_param("config.NUM_EPOCHS", config.NUM_EPOCHS)
    log_param("config.LEARNING_RATE", config.LEARNING_RATE)
    log_param("config.WEIGHT_DECAY", config.WEIGHT_DECAY)

    log_param("config.LOAD_MODEL", config.LOAD_MODEL)
    log_param("config.SAVE_MODEL", config.SAVE_MODEL)

    return experiment, run

def end_run(run):
    # End run and get status
    mlflow.end_run()
    run = mlflow.get_run(run.info.run_id)
    print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
    print("--")

def training_run(run, model, optimizer, loss_fn, train_loader, val_loader, test_loader):

    val_acc_max, test_acc_max = 0, 0
    val_Fm_max, test_Fm_max = 0, 0
    val_Fw_max, test_Fw_max = 0, 0
    
    for epoch in range(config.NUM_EPOCHS):
        
        print ('###### EPOCH: ', epoch)

        t0 = time.time()
        train_fn(train_loader, model, optimizer, loss_fn)
        log_metric("time_train_fn", time.time() - t0, step = epoch)

        print("On Val loader:")
        y_pred, y_true = get_predictions(model, val_loader)
    
        acc = check_class_accuracy(y_pred, y_true)
        print(f"Val accuracy is: {acc:2f}%")

        Fm, Fw = get_f1_scores(y_pred, y_true)
        print(f"Val Fm is: {Fm:2f}%")
        print(f"Val Fw is: {Fw:2f}%")

        log_metric("val_acc", acc, step = epoch)
        log_metric("val_Fm", Fm, step = epoch)
        log_metric("val_Fw", Fw, step = epoch)
        
        if acc > val_acc_max:
            log_metric("val_acc_max", acc, step = epoch)
            val_acc_max = acc

            if config.SAVE_MODEL:
                save_checkpoint(
                    model,
                    optimizer, 
                    filename=config.DATA_DIR+f'models/{config.EXPERIMENT}_run_{run.info.run_id}_epoch_{epoch}.pth.tar'
                    )
        
        if Fm > val_Fm_max:
            log_metric("val_Fm_max", Fm, step = epoch)
            val_Fm_max = Fm

            if config.SAVE_MODEL:
                save_checkpoint(
                    model,
                    optimizer, 
                    filename=config.DATA_DIR+f'models/{config.EXPERIMENT}_run_{run.info.run_id}_epoch_{epoch}.pth.tar'
                    )
        
        if Fw > val_Fw_max:
            log_metric("val_Fw_max", Fw, step = epoch)
            val_Fw_max = Fw

            if config.SAVE_MODEL:
                save_checkpoint(
                    model,
                    optimizer, 
                    filename=config.DATA_DIR+f'models/{config.EXPERIMENT}_run_{run.info.run_id}_epoch_{epoch}.pth.tar'
                    )

        print("On Test loader:")
        y_pred, y_true = get_predictions(model, test_loader)
    
        acc = check_class_accuracy(y_pred, y_true)
        print(f"Test accuracy is: {acc:2f}%")

        Fm, Fw = get_f1_scores(y_pred, y_true)
        print(f"Test Fm is: {Fm:2f}%")
        print(f"Test Fw is: {Fw:2f}%")

        log_metric("test_acc", acc, step = epoch)
        log_metric("test_Fm", Fm, step = epoch)
        log_metric("test_Fw", Fw, step = epoch)

        if acc > test_acc_max:
            log_metric("test_acc_max", acc, step = epoch)
            test_acc_max = acc

            if config.SAVE_MODEL:
                save_checkpoint(
                    model,
                    optimizer, 
                    filename=config.DATA_DIR+f'models/{config.EXPERIMENT}_run_{run.info.run_id}_epoch_{epoch}.pth.tar'
                    )

        if Fm > test_Fm_max:
            log_metric("test_Fm_max", Fm, step = epoch)
            test_Fm_max = Fm

            if config.SAVE_MODEL:
                save_checkpoint(
                    model,
                    optimizer, 
                    filename=config.DATA_DIR+f'models/{config.EXPERIMENT}_run_{run.info.run_id}_epoch_{epoch}.pth.tar'
                    )
        
        if Fw > test_Fw_max:
            log_metric("test_Fw_max", Fw, step = epoch)
            test_Fw_max = Fw

            if config.SAVE_MODEL:
                save_checkpoint(
                    model,
                    optimizer, 
                    filename=config.DATA_DIR+f'models/{config.EXPERIMENT}_run_{run.info.run_id}_epoch_{epoch}.pth.tar'
                    )

def train_fn(loader, model, optimizer, loss_fn):

    loop = tqdm(loader, leave=True)

    losses = []
    for batch_idx, (x, y) in enumerate(loop):

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        out = model(x)

        #print (batch_idx)
        #print ('DATA SHAPES:: INPUT x, y: ', x.shape, y.shape, ' MODEL out: ', out.shape)

        loss = loss_fn(out, y.type(torch.float))
        if batch_idx//2 == 0:
            log_metric("loss", loss.item())

        loss.backward()  # x.grad += dloss/dx
        optimizer.step() # x += -lr * x.grad

        #track losses
        losses.append(loss.item())
        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main():

    if config.OPEN_SET:

        for LABEL in config.LABELS:
            
            model = TSCResNet(in_channels=config.FEATURES).to(config.DEVICE)

            optimizer = optim.Adam(
                params = model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY
            )

            loss_fn = nn.CrossEntropyLoss()

            if config.LOAD_MODEL:
                load_checkpoint(
                    config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
                )

            print ('running LABEL:', LABEL)
            experiment, run = mlflow_setup()
            log_param("config.LABEL", LABEL)

            t0 = time.time()
            train_loader, val_loader, test_loader = get_data_loaders(label = LABEL)
            log_metric('time_get_data_loaders', time.time() - t0)

            training_run(run, model, optimizer, loss_fn, train_loader, val_loader, test_loader)

            end_run(run)
    else:
        model = TSCResNet(in_channels=config.FEATURES).to(config.DEVICE)

        optimizer = optim.Adam(
            params = model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY
        )

        loss_fn = nn.CrossEntropyLoss()

        if config.LOAD_MODEL:
            load_checkpoint(
                config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
            )

        experiment, run = mlflow_setup()
        log_param("config.LABEL", 'Closed Set')

        t0 = time.time()
        train_loader, val_loader, test_loader = get_data_loaders()
        log_metric('time_get_data_loaders', time.time() - t0)

        training_run(run, model, optimizer, loss_fn, train_loader, val_loader, test_loader)

        end_run(run)
    
if __name__ == "__main__":
    main()