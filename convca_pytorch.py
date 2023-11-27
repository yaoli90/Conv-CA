import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.data as data_utils

from convca_read import prepare_data_as, prepare_template, normalize


class ConvCA(nn.Module):
    def __init__(self, params, device):
        super(ConvCA, self).__init__()
        self.params = params
        self.device = device
        self.conv11 = nn.Conv2d(1, 16, 9, padding='same')
        self.conv12 = nn.Conv2d(16, 1, (1,9), padding='same')
        self.conv13 = nn.Conv2d(1, 1, (1,9), padding='valid')
        self.drop1 = nn.Dropout(p=0.75)
        self.flatten = nn.Flatten(start_dim=2)

        self.conv21 = nn.Conv2d(self.params['ch'], 40, (9,1), padding='same')
        self.conv22 = nn.Conv2d(40, 1, (9,1), padding='same')
        self.drop2 = nn.Dropout(p=0.15)

        self.fc = nn.Linear(params['cl'], params['cl'])

    def Corr(self, signal, temp):
        corr_xt = torch.bmm(signal,temp)
        corr_xt = torch.squeeze(corr_xt, 1)
        corr_xx = torch.bmm(signal, torch.transpose(signal,1,2))
        corr_xx = torch.squeeze(corr_xx, 1)
        corr_tt = torch.sum(temp*temp, dim=1)
        corr = corr_xt/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)
        return corr

    def forward(self, signal, temp):
        if torch.is_tensor(signal) != True:
            signal = torch.from_numpy(signal).float().to(self.device)
        if torch.is_tensor(temp) != True:
            temp = torch.from_numpy(temp).float().to(self.device)
        signal = self.conv11(signal)
        signal = self.conv12(signal)
        signal = self.conv13(signal)
        signal = self.drop1(signal)
        signal = self.flatten(signal)

        temp = self.conv21(temp)
        temp = self.conv22(temp)
        temp = self.drop2(temp)
        temp = torch.squeeze(temp)

        corr = self.Corr(signal, temp)
        out = self.fc(corr)
        return out


## parameters
# channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
permutation = [47,53,54,55,56,57,60,61,62]
params = {'tw':250,'Fs':250,'cl':40,'ch':len(permutation)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for subj in range(1,2):
    print(subj)
    train_run = [1,2,3,4,5]
    test_run = [0]

    ## build dataset
    x_train,y_train,freq = prepare_data_as(subj,train_run,params['tw']) ## [?,tw,ch]
    x_test,y_test,__ = prepare_data_as(subj,test_run,params['tw']) # [?,tw,ch]
    x_train = x_train.reshape((x_train.shape[0],1, params['tw'],params['ch']))
    x_test = x_test.reshape((x_test.shape[0],1, params['tw'],params['ch']))
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train).type(torch.LongTensor)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test).type(torch.LongTensor)

    ## build reference signal
    template = prepare_template(subj,train_run,params['tw']) # [cl*sample,cl,tw,ch]
    template = np.transpose(template, axes=(0,3,2,1)) # [cl*sample,ch,tw,cl]
    template = torch.tensor(template)

    train_dataset = data_utils.TensorDataset(x_train,template.repeat(len(train_run),1,1,1),y_train)
    test_dataset = data_utils.TensorDataset(x_test,template,y_test)

    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    ## build model
    model = ConvCA(params, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    epochs = 1000


    for e in range(epochs):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(training_loader):
            signal, temp, label = data
            optimizer.zero_grad()
            output = model(signal, temp)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
        print("training epoch", e, "loss:", last_loss)


    torch.save(model,"test.pth")

    model = torch.load("test.pth")
    model.eval()
    acc = 0
    for i, data in enumerate(test_loader):
        signal, temp, label = data
        optimizer.zero_grad()
        output = model(signal, temp)
        _, pred = torch.max(output, 1)
        diff = (pred==label).detach().cpu().numpy()
        acc += np.sum(diff)
    print(acc/y_test.shape[0])

    
