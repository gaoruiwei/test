import torch.nn as nn
import torch.nn.functional as F
import torch

class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(32)


        self.fc = nn.Linear(20*20*32, 10)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=x.view(-1,20*20*32)
        x=self.fc(x)
        return x

net=Q()
intput = torch.randn(1,1,32,32)
print('输入',intput)
out=net(intput)
print('输出',out)
