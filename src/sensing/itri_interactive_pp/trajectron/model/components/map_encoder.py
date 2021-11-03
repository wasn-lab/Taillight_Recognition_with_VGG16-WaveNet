import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#  {'VEHICLE': 
#  {'heading_state_index': 6, 
#  'patch_size': [50, 10, 50, 90], 
#  'map_channels': 3, (but actually we stack the same mask up to 3)
#  'hidden_channels': [10, 20, 10, 1], 
#  'output_size': 32, 
#  'masks': [5, 5, 5, 3], 
#  'strides': [2, 2, 1, 1], 
#  'dropout': 0.5},}, 

class CNNMapEncoder(nn.Module):
    def __init__(self, map_channels, hidden_channels, output_size, masks, strides, patch_size):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        patch_size_x = patch_size[0] + patch_size[2]
        patch_size_y = patch_size[1] + patch_size[3]
        input_size = (map_channels, patch_size_x, patch_size_y)
        # print("CNN input_size", input_size)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float('nan'))
        # print("CNN x_dummy shape", x_dummy.shape)
        
        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(nn.Conv2d(map_channels if i == 0 else hidden_channels[i-1],
                                        hidden_channels[i], masks[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)
            # print("CNN x_dummy convs[i]", i)
            # print("CNN x_dummy shape", x_dummy.shape)
            # print("CNN weight size", self.convs[i].weight.size())
        self.fc = nn.Linear(x_dummy.numel(), output_size)
    
    def forward(self, x, training):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x