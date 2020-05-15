import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset ,DataLoader

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data,label=None, transform=None):
        self.transform = transform
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        #out_label =  self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data


    

def mamall_mixrun_create(datapath,group1_sub_path,group2_sub_path):
    sample = pd.read_csv(datapath+group1_sub_path.iloc[0],sep='\t',header=None)
    sample = sample.values[None,:,:]
    for subnum in tqdm(np.sort(group1_sub_path.values[1:40])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        sample = np.append(sample,A.values[None,:,:],axis=0)

    test_sample = pd.read_csv(datapath+group2_sub_path.iloc[0],sep='\t',header=None)
    test_sample = test_sample.values[None,:,:]
    for subnum in tqdm(np.sort(group2_sub_path.values[1:80])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        test_sample = np.append(test_sample,A.values[None,:,:],axis=0)



    sample = sample[:,:,:-1]
    sample = np.reshape(sample.transpose(0,2,1),(40*316,-1))
    test_sample = test_sample[:,:,:-1]
    test_sample = np.reshape(test_sample.transpose(0,2,1),(80*316,-1))
    scaler = StandardScaler()
    scaler.fit(np.append(sample,test_sample,axis=0))
    sample = scaler.transform(sample)
    test_sample = scaler.transform(test_sample)
    
    return sample,test_sample


def bandpassed_mixrun_create(datapath,group1_sub_path,group2_sub_path):
    group1_bandsub_path = group1_sub_path.str.replace("MSMALL","BandPassed")
    group2_bandsub_path = group2_sub_path.str.replace("MSMALL","BandPassed")

    train_bandsample = pd.read_csv(datapath+group1_bandsub_path.iloc[0],sep='\t',header=None)
    train_bandsample = train_bandsample.values[None,:,:]

    for subnum in tqdm(np.sort(group1_bandsub_path.values[1:40])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        train_bandsample = np.append(train_bandsample,A.values[None,:,:],axis=0)

    test_bandsample = pd.read_csv(datapath+group2_bandsub_path.iloc[0],sep='\t',header=None)
    test_bandsample = test_bandsample.values[None,:,:]

    for subnum in tqdm(np.sort(group2_bandsub_path.values[1:80])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        test_bandsample = np.append(test_bandsample,A.values[None,:,:],axis=0)


    test_bandsample = test_bandsample[:,:,:-1]
    train_bandsample = train_bandsample[:,:,:-1]
    train_bandsample = np.reshape(train_bandsample.transpose(0,2,1),(40*316,-1))
    test_bandsample = np.reshape(test_bandsample.transpose(0,2,1),(80*316,-1))
    scaler = StandardScaler()
    scaler.fit(np.append(train_bandsample,test_bandsample,axis=0))
    train_bandsample = scaler.transform(train_bandsample)
    test_bandsample = scaler.transform(test_bandsample)
    
    return train_bandsample,test_bandsample
    
def mamall_separaterun_create(datapath,group1_sub_path,group2_sub_path):
    group1_sub_path_run1 = group1_sub_path[group1_sub_path.str.contains("run1")] #run1のファイル名パス
    group1_sub_path_run2 = group1_sub_path[group1_sub_path.str.contains("run2")]

    sample = pd.read_csv(datapath+group1_sub_path_run1.iloc[0],sep='\t',header=None)
    sample = sample.values[:,:-1].T
    scaler = StandardScaler()
    scaler.fit(sample)
    sample = scaler.transform(sample)
    sample = sample[None,:,:]
    for subnum in tqdm(np.sort(group1_sub_path_run1.values[1:40])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        A = A.values[:,:-1].T
        scaler = StandardScaler()
        scaler.fit(A)
        A = scaler.transform(A)
        A = A[None,:,:]
        sample = np.append(sample,A,axis=0)


    test_sample = pd.read_csv(datapath+group1_sub_path_run2.iloc[0],sep='\t',header=None)
    test_sample = test_sample.values[:,:-1].T
    scaler = StandardScaler()
    scaler.fit(test_sample)
    test_sample = scaler.transform(test_sample)
    test_sample = test_sample[None,:,:]

    for subnum in tqdm(np.sort(group1_sub_path_run2.values[1:80])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        A = A.values[:,:-1].T
        scaler = StandardScaler()
        scaler.fit(A)
        A = scaler.transform(A)
        A = A[None,:,:]
        test_sample = np.append(test_sample,A,axis=0)
    sample = np.reshape(sample,(40*316,-1))
    test_sample = np.reshape(test_sample,(80*316,-1))
    return sample,test_sample

def bandpassed_separaterun_create(datapath,group1_sub_path,group2_sub_path):
    group1_bandsub_path = group1_sub_path.str.replace("MSMALL","BandPassed")
    group1_sub_path_run1 = group1_bandsub_path[group1_bandsub_path.str.contains("run1")] #run1のファイル名パス
    group1_sub_path_run2 = group1_bandsub_path[group1_bandsub_path.str.contains("run2")]

    sample = pd.read_csv(datapath+group1_sub_path_run1.iloc[0],sep='\t',header=None)
    sample = sample.values[:,:-1].T
    scaler = StandardScaler()
    scaler.fit(sample)
    sample = scaler.transform(sample)
    sample = sample[None,:,:]
    for subnum in tqdm(np.sort(group1_sub_path_run1.values[1:40])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        A = A.values[:,:-1].T
        scaler = StandardScaler()
        scaler.fit(A)
        A = scaler.transform(A)
        A = A[None,:,:]
        sample = np.append(sample,A,axis=0)


    test_sample = pd.read_csv(datapath+group1_sub_path_run2.iloc[0],sep='\t',header=None)
    test_sample = test_sample.values[:,:-1].T
    scaler = StandardScaler()
    scaler.fit(test_sample)
    test_sample = scaler.transform(test_sample)
    test_sample = test_sample[None,:,:]

    for subnum in tqdm(np.sort(group1_sub_path_run2.values[1:80])):

        A = pd.read_csv(datapath+subnum,sep='\t',header=None)
        A = A.values[:,:-1].T
        scaler = StandardScaler()
        scaler.fit(A)
        A = scaler.transform(A)
        A = A[None,:,:]
        test_sample = np.append(test_sample,A,axis=0)
    sample = np.reshape(sample,(40*316,-1))
    test_sample = np.reshape(test_sample,(80*316,-1))
    return sample,test_sample


__factory = {
    'MSMALL': {'mixrun':mamall_mixrun_create,
              'separate_run':mamall_separaterun_create},
    'BandPassed': {'mixrun':bandpassed_mixrun_create,
                   'separate_run':bandpassed_separaterun_create},
}

def create(datatype, runtype,batch_size,train_shuffle,
           test_shuffle,datapath,group1_sub_path,group2_sub_path):
    
    if datatype not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(datatype))
    data_load_func = __factory[datatype][runtype]
    sample,test_sample = data_load_func(datapath,group1_sub_path,group2_sub_path)
    train_dataset = MyDataset(torch.tensor(sample,dtype=torch.float32))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=train_shuffle)

    test_dataset = MyDataset(torch.tensor(test_sample,dtype=torch.float32))
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=test_shuffle)
    return train_dataloader,test_dataloader












