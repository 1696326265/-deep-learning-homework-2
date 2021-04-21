from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


def model_A(num_classes):
    model_resnet = models.resnet50(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_B(num_classes):
    ## your code here
    pass


def model_C(num_classes):
    ## your code here
    pass

def model_TEST(num_classes):
    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()   #size=(3*224*224)
            self.conv1 = nn.Sequential(   
                nn.Conv2d(
                    in_channels= 3,   # 输入的高度
                    out_channels= 8, # 输出的高度
                    kernel_size =7,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 3       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #size=(8*112*112)
            )
            self.conv2=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 8,   # 输入的高度
                    out_channels= 16, # 输出的高度
                    kernel_size =7,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 3       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #size=(16*56*56)
            )
            self.conv3=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 16,   # 输入的高度
                    out_channels= 32, # 输出的高度
                    kernel_size =5,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 2       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #size=(32*28*28)
            )
            self.conv4=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 32,   # 输入的高度
                    out_channels= 64, # 输出的高度
                    kernel_size =5,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 2       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #size=(64*14*14)
            )
            self.conv5=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 64,   # 输入的高度
                    out_channels= 128, # 输出的高度
                    kernel_size =3,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 1       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #size=(128*7*7)
            )
            self.conv6=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 128,   # 输入的高度
                    out_channels= 512, # 输出的高度
                    kernel_size =3,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 1       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=7, stride=7), #size=(512*1*1)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(512 * 1 * 1, 120),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(120, 10),
            )
            self.mp=nn.MaxPool2d(2)
            self.fc=nn.Linear(320,10)
        def forward(self,x):
#             x=F.relu(self.mp(self.conv1(x)))
#             x=F.relu(self.mp(self.conv2(x)))
#             x=x.view(x.size()[0], -1)
#             x=self.fc(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = x.view(x.size(0), -1)  # 图片的维度：（batch,32*7*7）
            x = self.fc1(x)
            x = self.fc2(x)
            return x
    model=Net()
    return model


def model_SIMPLE(num_classes):
    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()   #size=(3*224*224)
            self.conv1 = nn.Sequential(   
                nn.Conv2d(
                    in_channels= 3,   # 输入的高度
                    out_channels= 12, # 输出的高度
                    kernel_size =11,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 5       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=4), #size=(12*56*56)
            )
            self.conv2=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 12,   # 输入的高度
                    out_channels= 24, # 输出的高度
                    kernel_size =9,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 4       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=4), #size=(24*14*14)
            )
            self.conv3=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 24,   # 输入的高度
                    out_channels= 48, # 输出的高度
                    kernel_size =5,   # 
                    stride = 1,       # 每次扫描跳的范围
                    padding = 2       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #size=(48*7*7)
            )
            self.conv4=nn.Sequential(  
                nn.Conv2d(
                    in_channels= 48,   # 输入的高度
                    out_channels= 1024, # 输出的高度
                    kernel_size =3,   # 
                    stride = 2,       # 每次扫描跳的范围
                    padding = 1       # 补全边缘像素点
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=4), #size=(1024*1*1)
            )
            
            self.fc1 = nn.Sequential(
                nn.Linear(1024 * 1 * 1, 500),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(500, 10),
            )
            self.dropout = nn.Dropout(p=0.5)
            self.mp=nn.MaxPool2d(2)
            self.fc=nn.Linear(320,10)
            
        def forward(self,x):
#             x=F.relu(self.mp(self.conv1(x)))
#             x=F.relu(self.mp(self.conv2(x)))
#             x=x.view(x.size()[0], -1)
#             x=self.fc(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)  # 图片的维度：（batch,32*7*7）
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    model=Net()
    return model