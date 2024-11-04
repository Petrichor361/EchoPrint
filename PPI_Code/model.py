import torch
import torch.nn as nn
from KANLinear import KANLinear

class KAN_PPI(nn.Module):
    def __init__(self,nums=[11,16,8,1]):
        super(KAN_PPI, self).__init__()
        FC=list()
        for i in range(len(nums)-1):
            FC.append(KANLinear(nums[i], nums[i+1]))         
        self.FC = nn.Sequential(*FC)  

    def forward(self, x):
        mask = ~torch.isnan(x)
        x = torch.nan_to_num(x, nan=0.0) * mask
        x = self.FC(x)
        x = torch.sigmoid(x)
        return x
# 定义一个多层感知机模型，添加批归一化层
class MLP_PPI(nn.Module):
    def __init__(self,nums=	[11,32,128,64,16,1]):
        super(MLP_PPI, self).__init__()
        FC=list()
        for i in range(len(nums)-1):
            FC.append(nn.Linear(nums[i], nums[i+1]))         
            if i!=len(nums)-2:
                FC.append(nn.BatchNorm1d(nums[i+1]) )
                FC.append(nn.ReLU())
        self.FC = nn.Sequential(*FC)  

    def forward(self, x):
        mask = ~torch.isnan(x)
        x = torch.nan_to_num(x, nan=0.0) * mask
        x = self.FC(x)
        x = torch.sigmoid(x) 
        return x
if __name__ == '__main__':
    x = [[0,3,2,3,4,1,0,3,2,4,1],[0,3,2,3,4,0,3,2,3,4,1]]
    x = torch.tensor(x, dtype=torch.float32)
    net=MLP_PPI([11,64,128,32,16,1])
    print("Num params: ", sum(p.numel() for p in net.parameters()))
    output=net(x)
    print(output.shape)
    
