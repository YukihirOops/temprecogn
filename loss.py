import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fuc(x,output,weight,metanalysis,alpha,deco_weight=None):
    """weight 59412x4"""
    mseloss = nn.MSELoss()
    num = len(metanalysis)
    weightdata = weight[:num,:]
    meta_std = torch.std(metanalysis,dim=1)
    weight_std = torch.std(weightdata,dim=1)
    meta_mean = torch.mean(metanalysis,dim=1)[:,None]
    weight_mean = torch.mean(weightdata,dim=1)[:,None]
    covar = torch.mean((weightdata - weight_mean)*torch.tensor((metanalysis-meta_mean)),dim=1)
    corr = covar/(weight_std*meta_std)
    
    corr_mean = torch.mean(corr)
    rloss = 1-corr_mean
    
    
    mse = mseloss(x,output)
    
    if deco_weight is not None:
        weightdata = deco_weight.t()[:num,:]
        weight_std = torch.std(weightdata,dim=1)
        weight_mean = torch.mean(weightdata,dim=1)[:,None]
        covar = torch.mean((weightdata - weight_mean)*torch.tensor((metanalysis-meta_mean)),dim=1)
        corr = covar/(weight_std*meta_std)
        corr_mean_de= torch.mean(corr)
        rloss_de = 1-corr_mean
        return mse + alpha*rloss + alpha*rloss_de,corr_mean ,mse,corr_mean_de
        
    return mse + alpha*rloss,corr_mean ,mse


def sparse_loss(autoencoder, images):
    loss = 0
    values = images

    h = autoencoder.encoder(images)
    
    loss += torch.mean(torch.abs(h))
    return loss