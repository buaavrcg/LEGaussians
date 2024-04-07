import torch
import torch.nn as nn
    
class IndexDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(IndexDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class XyzMLP(nn.Module):
    def __init__(self, D=4, W=128, 
                 in_channels_xyz=63, out_channels_xyz=8):
        super(XyzMLP, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.out_channels_xyz = out_channels_xyz
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i}", layer)
        self.xyz_encoding_final = nn.Linear(W, out_channels_xyz)
        
    def forward(self, x):
        for i in range(self.D):
            x = getattr(self, f"xyz_encoding_{i}")(x)
        xyz_encoding_final = self.xyz_encoding_final(x)
        return xyz_encoding_final