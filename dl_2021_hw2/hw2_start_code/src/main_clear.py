import torch
import torch.nn as nn
import torch.optim as optim
import data_unshuffle
import models
import os
import sys
import time
import numpy as np

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

logstring = ""

def logprint(s,argmap):
    global logstring
    print(s)
    logstring += s + '\n'
    f = open('{0}_logstring'.format(argmap['who']),"w")
    f.write(logstring)
    return

def train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=20,argmap={}):

    def train(model, train_loader,optimizer,criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0
        cnt = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            
            cnt += len(inputs)
            if (cnt>=argmap['trainlim']): break

        epoch_loss = total_loss / cnt #len(train_loader.dataset)
        epoch_acc = total_correct.double() / cnt #len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        cnt = 0
        conf_mat = [([0]*10) for i in range(10)]
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            
            for i in range(len(inputs)):
                conf_mat[int(labels.data[i])][int(predictions[i])] += 1
                # print(labels.data.shape,predictions.shape)
                # print(type(labels.data),int(labels.data[i]),type(predictions),int(predictions[i]))
            
            cnt += len(inputs)
            if (cnt>=argmap['validlim']): break
        
        epoch_loss = total_loss / cnt #len(valid_loader.dataset)
        epoch_acc = total_correct.double() / cnt #len(valid_loader.dataset)
        # print(conf_mat)
        return conf_mat, epoch_loss, epoch_acc.item()

    best_acc = 0.0
    logprint('*' * 60,argmap)
    
    Loss_Acc = []
    Conf_Mat = []
    
    for epoch in range(num_epochs):
        logprint('epoch:{:d}/{:d}'.format(epoch, num_epochs) + "   " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),argmap)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        for T in range(argmap['rounds_per_train']-1):
            train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        logprint("training: {:.4f}, {:.4f}".format(train_loss, train_acc),argmap)
        conf_mat, valid_loss, valid_acc = valid(model, valid_loader,criterion)
        logprint("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc),argmap)
        
        print(np.array(conf_mat))
        
        Loss_Acc.append([train_loss,train_acc,valid_loss,valid_acc])
        Conf_Mat.append(conf_mat)
        
        np.save('{0}_Loss_Acc.npy'.format(argmap['who']),np.array(Loss_Acc))
        np.save('{0}_Conf_Mat.npy'.format(argmap['who']),np.array(Conf_Mat))
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            # torch.save(best_model, '{0}_best_model.pt'.format(argmap['who']))
            torch.save(best_model.state_dict(), '{0}_best_model.pkl'.format(argmap['who']))
            logprint('Saved best model!',argmap)
        logprint('*' * 100,argmap)


if __name__ == '__main__':
    argmap = eval(sys.argv[1])
    print(argmap)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "../hw2_dataset/" ## You need to specify the data_dir first
    inupt_size = 224
    batch_size = argmap['batch_size']

    ## about training
    num_epochs = argmap['epochs'] #100
    lr = argmap['lr'] #0.001

    ## model initialization
    exec("model = models.model_{0}(num_classes=num_classes)".format(argmap['model']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpudevice = torch.device("cpu")
    model = model.to(device)

    ## data preparation
    pretrain_loader, valid_loader = data_unshuffle.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size, argmap=argmap)
    
#     print(type(train_loader))

#     from torch.utils.data import TensorDataset, DataLoader
#     import random
#     premodel = models.model_Myhighway(num_classes=num_classes)
#     premodel.load_state_dict(torch.load('Task5_modelMyhighway_best_model.pkl'))

    premodel = models.model_Myhighway(num_classes=num_classes)
    premodel.to(device)
    premodel.load_state_dict(torch.load('Task5_modelMyhighway_best_model.pkl'))
    premodel.train(False)
    
    ID = []
    cnt = 0
    lastlabel = -1
    for inputs, labels in pretrain_loader:
        inputs = inputs.to(device)
        outputs = premodel(inputs)
        outputs = outputs.to(cpudevice)
        _, predictions = torch.max(outputs, 1)
        for i in range(len(inputs)):
            if lastlabel != int(labels[i]):
                cnt = 0
                lastlabel = int(labels[i])
            if labels[i] != predictions[i]:
                ID.append([int(labels[i]),cnt])
#                 print("false", labels[i], predictions[i])
#             else:
#                 print("true")
            cnt += 1
#         if cnt>100: break
    
    np.save('ID.npy',np.array(ID))
    
#     A = torch.tensor(A)
#     B = torch.tensor(B)

#     S = TensorDataset(A,B)
#     train_loader = DataLoader(S, batch_size=32, shuffle=True)

#     ## optimizer
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#     ## loss function
#     criterion = nn.CrossEntropyLoss()
#     train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs,argmap=argmap)
