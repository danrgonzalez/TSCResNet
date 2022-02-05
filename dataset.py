import os
import numpy as np

import torch
from torch.utils.data import Dataset

class PAMAP2Dataset(Dataset):

    def __init__(self, data_dir, data_type, label = None, transform=None):

        self.data_dir = data_dir
        self.data_type = data_type
        self.label = label

        if self.label is None:
            self.labels_path = os.path.join(self.data_dir, self.data_type, 'y_%s.csv'%self.data_type)
        elif self.data_type == 'val' or self.data_type == 'test':
            self.labels_path = os.path.join(self.data_dir, self.data_type, 'y_%s_open_set.csv'%(self.data_type))
        else:
            self.labels_path = os.path.join(self.data_dir, self.data_type, 'y_%s_%s.csv'%(self.data_type, self.label))

        self.labels = np.loadtxt(self.labels_path , delimiter = ',')
        self.labels = torch.tensor(self.labels.astype(np.float32)).type(torch.LongTensor)
        print ('IMPORTED: ', self.labels_path)

        if self.label is None:
            self.X_filepath = os.path.join(self.data_dir, self.data_type, 'X_%s.csv'%(self.data_type))
        elif self.data_type == 'val' or self.data_type == 'test':
            self.X_filepath = os.path.join(self.data_dir, self.data_type, 'X_%s_open_set.csv'%(self.data_type))
        else:
            self.X_filepath = os.path.join(self.data_dir, self.data_type, 'X_%s_%s.csv'%(self.data_type, self.label))

        self.X = np.loadtxt(self.X_filepath, delimiter = ',')
        self.X = torch.tensor(self.X.astype(np.float32))
        print ('IMPORTED: ', self.X_filepath)

        #reshape
        self.X = self.X.reshape(self.X.shape[0], 40, 172)
        self.labels = self.labels.reshape(self.X.shape[0], 18)

        self.transform=None
        self.target_transform=None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        #print ('IDX: ', idx)

        if self.transform:
            pass
        if self.target_transform:
            pass

        #apply index
        return self.X[idx], self.labels[idx]

def test():

    data_dir = '/Users/dgonzalez/Documents/dissertation/'

    #dataset = PAMAP2Dataset(data_dir, 'train', label = None)
    #dataset = PAMAP2Dataset(data_dir, 'train', label = 'Nordic walking')
    dataset = PAMAP2Dataset(data_dir, 'test', label = 'Nordic walking')

    X, y = dataset.__getitem__(17)
    
    print ('IN TEST')
    print (type(X), type(y))
    print (X.shape, y.shape)

    print (X[2], y)
    print ('')
    print (y)
    print (y.type(torch.LongTensor))

if __name__ == "__main__":
    test()

