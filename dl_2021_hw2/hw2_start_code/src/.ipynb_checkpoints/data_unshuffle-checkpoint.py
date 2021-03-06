from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir = "../data/",input_size = 224,batch_size = 36, argmap={}):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    ## The default dir is for the first task of large-scale deep learning
    ## For other tasks, you may need to modify the data dir or even rewrite some part of 'data.py'
#     image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, '1-Large-Scale', 'train'), data_transforms['train'])
#     image_dataset_valid = datasets.ImageFolder(os.path.join(data_dir,'test'), data_transforms['test'])
#     exec("image_dataset_train = datasets.ImageFolder({0}, data_transforms['train'])".format(argmap['train_path']))
    image_dataset_train = eval("datasets.ImageFolder({0}, data_transforms['train'])".format(argmap['train_path']))
#     exec("image_dataset_valid = datasets.ImageFolder({0}, data_transforms['test'])".format(argmap['valid_path']))
    image_dataset_valid = eval("datasets.ImageFolder({0}, data_transforms['test'])".format(argmap['valid_path']))

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=False, num_workers=1)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, valid_loader
