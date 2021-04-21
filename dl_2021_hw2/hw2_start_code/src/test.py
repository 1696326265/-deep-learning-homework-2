import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import time

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

logstring = ""

def logprint(s):
    global logstring
    print(s)
    logstring += s + '\n'
    f = open('logstring',"w")
    f.write(logstring)
    return

def train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=20):

    def train(model, train_loader,optimizer,criterion):
        return 0, 0
        model.train(True)
        total_loss = 0.0
        total_correct = 0

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

            print(total_correct)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            
            print(total_correct)
            
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        logprint('epoch:{:d}/{:d}'.format(epoch, num_epochs) + "   " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        logprint('*' * 60)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        logprint("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        logprint("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            # torch.save(best_model, 'best_model.pt')
            logprint('Saved best model!')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    argmap = { 'model':'A' , 'train_path':'os.path.join(data_dir, \'1-Large-Scale\', \'train\')' , 'valid_path':'os.path.join(data_dir,\'test\')' , 'trainlim':10000000 , 'validlim':10000000, 'batch_size':32 , 'lr':0.001 , 'epochs':100 , 'who':'Task1_modelResnet' , 'rounds_per_train':1 }
    
    ## about model
    num_classes = 10

    ## about data
    data_dir = "../hw2_dataset/" ## You need to specify the data_dir first
    inupt_size = 224
    batch_size = 36

    ## about training
    num_epochs = 100
    lr = 0.001

    ## model initialization
    model = models.model_A(num_classes=num_classes)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)
    model.load_state_dict(torch.load("Task1_modelResnet_best_model.pkl",map_location=torch.device('cpu')))

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size, argmap=argmap)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)
