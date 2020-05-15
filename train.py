import numpy as np
import torch
import torch.nn as nn
from torch import optim
from loss import loss_fuc,sparse_loss

def group_trainer(num_epochs,model,lr,train_dataloader,
                 test_dataloader,is_sparse)

    mseloss = nn.MSELoss()
    optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, weight_decay=0)
    model = model.to("cuda")

    weight_list = list()
    for epoch in range(num_epochs):
        loss_list = list()
        mseloss_list = list()
        corr_list = list()

        test_loss_list = list()
        test_mseloss_list = list()
        test_corr_list = list()
        for data in train_dataloader:

            data = data.to("cuda")

            # ===================forward=====================
            _,output = model(data)
            loss,corr,mse_loss= loss_fuc(data,output,model.encoder.weight,metanalysis,1.5)
            spa_loss = sparse_loss(model,data)
            loss = loss + 0.0001*spa_loss
            loss_list.append(loss.item())
            mseloss_list.append(mse_loss.item())
            corr_list.append(corr.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], sumloss:{:.4f},mseloss:{:.4f}'
              .format(epoch + 1, num_epochs, np.mean(loss_list),np.mean(mseloss_list)))



        for data in test_dataloader:

            data = data.to("cuda")
            model.eval()
            # ===================forward=====================
            _,output = model(data)
            loss,corr,mse_loss = loss_fuc(data,output,model.encoder.weight,metanalysis,1.5)
            test_loss_list.append(loss.item())
            test_mseloss_list.append(mse_loss.item())
            test_corr_list.append(corr.item())
        print('test_epoch [{}/{}], loss:{:.4f},corr:{:.4f},mseloss:{:.4f}'
              .format(epoch + 1, num_epochs, np.mean(test_loss_list),np.mean(test_corr_list),np.mean(test_mseloss_list)))
        weight_list.append(model.encoder.weight.cpu().t())