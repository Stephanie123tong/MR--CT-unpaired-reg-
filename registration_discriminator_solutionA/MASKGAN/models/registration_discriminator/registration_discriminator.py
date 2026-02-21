"""
Registration Discriminator Network
"""

import torch
import torch.nn as nn


class RegistrationDiscriminator(nn.Module):
    """Discriminator that uses deformation field complexity."""
    
    def __init__(self, num_blocks=8, gan_mode='lsgan', use_local=True):
        super(RegistrationDiscriminator, self).__init__()
        self.num_blocks = num_blocks
        self.gan_mode = gan_mode
        self.use_local = use_local
        
        # Global branch
        self.global_fc1 = nn.Linear(1, 64)
        self.global_fc2 = nn.Linear(64, 32)
        self.global_fc3 = nn.Linear(32, 1)
        
        if use_local:
            # Local branch
            self.local_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.local_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.local_conv3 = nn.Conv2d(64, 1, kernel_size=1)
            self.local_bn1 = nn.BatchNorm2d(32)
            self.local_bn2 = nn.BatchNorm2d(64)
            self.fusion_fc = nn.Linear(1 + num_blocks * num_blocks, 1)
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, global_complexity, local_complexity=None):
        B = global_complexity.shape[0]
        
        # Global branch
        global_feat = global_complexity.unsqueeze(1)
        global_feat = self.relu(self.global_fc1(global_feat))
        global_feat = self.relu(self.global_fc2(global_feat))
        global_out = self.global_fc3(global_feat)
        
        if self.use_local and local_complexity is not None:
            # Local branch
            local_feat = local_complexity.unsqueeze(1)
            local_feat = self.relu(self.local_bn1(self.local_conv1(local_feat)))
            local_feat = self.relu(self.local_bn2(self.local_conv2(local_feat)))
            local_out = self.local_conv3(local_feat)
            local_out = local_out.view(B, -1)
            
            # Fusion
            combined = torch.cat([global_out, local_out], dim=1)
            output = self.fusion_fc(combined)
        else:
            output = global_out
        
        return output

