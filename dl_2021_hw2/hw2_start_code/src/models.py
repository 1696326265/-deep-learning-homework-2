from torchvision import models
import torch.nn as nn
import torch.nn.functional as F



#----------------------------------------------------------------------------
# https://blog.csdn.net/winycg/article/details/86709991

# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def model_ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
def model_ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])
def model_ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])
def model_ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])
def model_ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

#------------------------------------------------------------------------------
# https://zhuanlan.zhihu.com/p/149387262

def model_another_ResNet50(num_classes):
    class ConvBlock(nn.Module):
        def __init__(self, in_channel, f, filters, s):
            super(ConvBlock,self).__init__()
            F1, F2, F3 = filters
            self.stage = nn.Sequential(
                nn.Conv2d(in_channel,F1,1,stride=s, padding=0, bias=False),
                nn.BatchNorm2d(F1),
                nn.ReLU(True),
                nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=False),
                nn.BatchNorm2d(F2),
                nn.ReLU(True),
                nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F3),
            )
            self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
            self.batch_1 = nn.BatchNorm2d(F3)
            self.relu_1 = nn.ReLU(True)

        def forward(self, X):
            X_shortcut = self.shortcut_1(X)
            X_shortcut = self.batch_1(X_shortcut)
            X = self.stage(X)
            X = X + X_shortcut
            X = self.relu_1(X)
            return X    

    class IndentityBlock(nn.Module):
        def __init__(self, in_channel, f, filters):
            super(IndentityBlock,self).__init__()
            F1, F2, F3 = filters
            self.stage = nn.Sequential(
                nn.Conv2d(in_channel,F1,1,stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F1),
                nn.ReLU(True),
                nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=False),
                nn.BatchNorm2d(F2),
                nn.ReLU(True),
                nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F3),
            )
            self.relu_1 = nn.ReLU(True)

        def forward(self, X):
            X_shortcut = X
            X = self.stage(X)
            X = X + X_shortcut
            X = self.relu_1(X)
            return X

    class ResModel(nn.Module):
        def __init__(self, n_class):
            super(ResModel,self).__init__()
            self.stage1 = nn.Sequential(
                nn.Conv2d(3,64,7,stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(3,2,padding=1),
            )
            self.stage2 = nn.Sequential(
                ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
                IndentityBlock(256, 3, [64, 64, 256]),
                IndentityBlock(256, 3, [64, 64, 256]),
            )
            self.stage3 = nn.Sequential(
                ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
                IndentityBlock(512, 3, [128, 128, 512]),
                IndentityBlock(512, 3, [128, 128, 512]),
                IndentityBlock(512, 3, [128, 128, 512]),
            )
            self.stage4 = nn.Sequential(
                ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
                IndentityBlock(1024, 3, [256, 256, 1024]),
                IndentityBlock(1024, 3, [256, 256, 1024]),
                IndentityBlock(1024, 3, [256, 256, 1024]),
                IndentityBlock(1024, 3, [256, 256, 1024]),
                IndentityBlock(1024, 3, [256, 256, 1024]),
            )
            self.stage5 = nn.Sequential(
                ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
                IndentityBlock(2048, 3, [512, 512, 2048]),
                IndentityBlock(2048, 3, [512, 512, 2048]),
            )
            self.pool = nn.AvgPool2d(2,2,padding=1)
            self.fc = nn.Sequential(
                nn.Linear(8192,n_class)
            )

        def forward(self, X):
            out = self.stage1(X)
            out = self.stage2(out)
            out = self.stage3(out)
            out = self.stage4(out)
            out = self.stage5(out)
            out = self.pool(out)
            out = out.view(out.size(0),-1)
            out = self.fc(out)
            return out

    model_resnet = ResModel(num_classes)
    return model_resnet


#-----------------------------------------------------------------------------



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

def model_Myhighway(num_classes):
    class res_block(nn.Module):
        def __init__(self, in_channel, filters):
            super(res_block,self).__init__()
            F1, F2, F3 = filters
            self.stage = nn.Sequential(
                nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F1),
                nn.ReLU(True),
                nn.Conv2d(F1, F2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(F2),
                nn.ReLU(True),
                nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F3),
            )
            self.relu_1 = nn.ReLU(True)

        def forward(self, X):
            X_shortcut = X
            X = self.stage(X)
            X = X + X_shortcut
            X = self.relu_1(X)
            return X
    
    class Myhighway(nn.Module):
        def __init__(self):
            super(Myhighway,self).__init__()
            self.stage1 = nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=1, padding=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                res_block(32, [16, 16, 32]),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                res_block(64, [32, 32, 64]),
                res_block(64, [32, 32, 64]),
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(64, 128, 5, stride=1, padding=2, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                res_block(128, [64, 64, 128]),
                res_block(128, [64, 64, 128]),
            )
            self.stage4 = nn.Sequential(
                nn.Conv2d(128, 256, 5, stride=1, padding=2, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                res_block(256, [64, 64, 256]),
                res_block(256, [64, 64, 256]),
                res_block(256, [64, 64, 256]),
            )
            self.stage5 = nn.Sequential(
                nn.Conv2d(256, 256, 5, stride=1, padding=2, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                res_block(256, [64, 64, 256]),
                res_block(256, [64, 64, 256]),
                nn.AvgPool2d(kernel_size=3, stride=3,padding=1),
            )
            self.fc1 = nn.Sequential(
                nn.Linear(2304,300),
            )
            self.dpout = nn.Dropout(p=0.5)
            self.fc2 = nn.Sequential(
                nn.Linear(300,10),
            )

        def forward(self, X):
            out = self.stage1(X)
            out = self.stage2(out)
            out = self.stage3(out)
            out = self.stage4(out)
            out = self.stage5(out)
            out = out.view(out.size(0),-1)
            out = self.fc1(out)
            out = self.dpout(out)
            out = self.fc2(out)
            return out

    model = Myhighway()
    return model
    

def model_SIMPLE(num_classes):
    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()   #size=(3*224*224)
            self.conv1 = nn.Sequential(   
                nn.Conv2d( in_channels= 3, out_channels= 20, kernel_size =5, stride = 1, padding = 2),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(   
                nn.Conv2d( in_channels= 20, out_channels= 30, kernel_size =5, stride = 1, padding = 2 ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=7, stride=7), #size=(30*32*32)
            )
            self.conv3=nn.Sequential(  
                nn.Conv2d( in_channels= 30, out_channels= 40, kernel_size =5, stride = 1, padding = 2 ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=4), #size=(40*8*8)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(40 * 8 * 8, 400),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(400, 200),
            )
            self.fc3 = nn.Sequential(
                nn.Linear(200, 10),
            )
            self.dropout = nn.Dropout(p=0.5)
            
        def forward(self,x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x) 
            x = self.fc3(x)
            return x
    model=Net()
    return model
