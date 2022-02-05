from tqdm import tqdm

import config
import numpy as np
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader

import mlflow

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    mlflow.log_artifact(filename, 'models')
    print("=> Finished Saving checkpoint")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    print("=> Loaded checkpoint")

def get_data_loaders(label = None):
    from dataset import PAMAP2Dataset

    train_dataset = PAMAP2Dataset(config.DATA_DIR, 'train', label)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    val_dataset  = PAMAP2Dataset(config.DATA_DIR, 'val', label)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    test_dataset  = PAMAP2Dataset(config.DATA_DIR, 'test', label)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

def get_predictions(model, loader):

    model.eval() #switch between train/inference

    ############ get_predictions - start
    y_pred, y_true = [], []
    for idx, (x, y) in enumerate(tqdm(loader)):
        
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        
        with torch.no_grad(): #turn off gradient computation
            out = model(x)

        y_pred.append(out)
        y_true.append(y)

        # print ('IDX: ', idx)
        # print ('X shape: ', x.shape, 'y shape: ', y.shape)
        # print ('out shape: ', out.shape)
    ############ get_predictions - end

    model.train() #switch model back to train

    #we need the argmax only
    y_pred = torch.argmax(torch.cat(y_pred), dim = -1)
    y_true = torch.argmax(torch.cat(y_true), dim = -1)
    
    # print ('argmax shape: ', y_pred.shape, y_true.shape)
    # print (y_pred)
    # print (y_true)
    # print (np.unique(y_pred.numpy()))
    # print (np.unique(y_true.numpy()))

    return y_pred, y_true

def check_class_accuracy(y_pred, y_true):
    
    tot_class_preds, correct_class = 0, 0
    correct_class += torch.sum(y_pred == y_true)
    tot_class_preds += y_true.shape[0]

    acc = (correct_class/(tot_class_preds+1e-16))*100

    return acc.item()

def get_f1_scores(y_pred, y_true):

    #convert to labels
    y_pred_labeled = [config.ONE_HOT_LABELS[pred] for pred in y_pred.numpy()]
    y_true_labeled = [config.ONE_HOT_LABELS[true] for true in y_true.numpy()]

    # print ('CHECKING PRED LABELS')
    # print (type(y_pred_labeled))
    # print (np.unique(y_pred_labeled))
    
    if config.OPEN_SET:
        Fm = f1_score(y_true_labeled, y_pred_labeled, labels = config.LABELS_SET, average = 'macro')
        Fw = f1_score(y_true_labeled, y_pred_labeled, labels = config.LABELS_SET, average = 'weighted')
    else:
        Fm = f1_score(y_true_labeled, y_pred_labeled, average = 'macro')
        Fw = f1_score(y_true_labeled, y_pred_labeled, average = 'weighted')

    return Fm*100, Fw*100

def test():

    import torch.optim as optim
    from model import TSCResNet

    #CLOSED-SET
    BEST_MODEL_RUN = '009872e235fa4687901b729567ec3593'
    BEST_EPOCH = '29'

    if config.OPEN_SET:
        BEST_MODEL_RUN = '64e31ccb74ac4d929ff81dc3a91d3993'
        BEST_EPOCH = '37'

    BEST_MODEL_PATH = f'{config.EXPERIMENT}_run_{BEST_MODEL_RUN}_epoch_{BEST_EPOCH}.pth.tar'
    print (BEST_MODEL_PATH)

    model = TSCResNet(in_channels=config.FEATURES).to(config.DEVICE)

    optimizer = optim.Adam(
        params = model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY
    )

    print ('LOADING MODEL!')
    load_checkpoint(
        '/Users/dgonzalez/Documents/dissertation/models/%s'%(BEST_MODEL_PATH), 
        model, 
        optimizer, 
        config.LEARNING_RATE
    )       

    #train_loader, val_loader, test_loader = get_data_loaders()
    train_loader, val_loader, test_loader = get_data_loaders(label = 'lying')

    print("On Test loader:")
    y_pred, y_true = get_predictions(model, test_loader)
    
    acc = check_class_accuracy(y_pred, y_true)
    print(f"Test accuracy is: {acc:2f}%")

    Fm, Fw = get_f1_scores(y_pred, y_true)
    print(f"Test Fm is: {Fm:2f}%")
    print(f"Test Fw is: {Fw:2f}%")

if __name__ == "__main__":
    test()