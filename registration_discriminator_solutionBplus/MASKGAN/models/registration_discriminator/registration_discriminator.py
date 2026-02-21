"""
Registration Discriminator Network (Block-level)
"""

import torch
import torch.nn as nn


class RegistrationDiscriminator(nn.Module):
    """
    Block-level discriminator that classifies individual blocks as "good" or "bad" registration.
    
    Input: Block features [B, num_blocks*num_blocks, feature_dim]
    Output: Block-level predictions [B, num_blocks*num_blocks, 1]
    """
    
    def __init__(self, num_blocks=8, gan_mode='lsgan', feature_dim=4):
        super(RegistrationDiscriminator, self).__init__()
        self.num_blocks = num_blocks
        self.gan_mode = gan_mode
        self.feature_dim = feature_dim
        
        # MLP for block-level classification
        # Input: [B, num_blocks*num_blocks, feature_dim]
        # Output: [B, num_blocks*num_blocks, 1]
        self.block_fc1 = nn.Linear(feature_dim, 64)
        self.block_fc2 = nn.Linear(64, 32)
        self.block_fc3 = nn.Linear(32, 1)
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, block_features):
        """
        Args:
            block_features: [B, num_blocks*num_blocks, feature_dim]
        
        Returns:
            output: [B, num_blocks*num_blocks, 1] - Predictions for each block
        """
        # Process each block independently
        # [B, num_blocks*num_blocks, feature_dim] -> [B, num_blocks*num_blocks, 1]
        x = self.relu(self.block_fc1(block_features))
        x = self.dropout(x)
        x = self.relu(self.block_fc2(x))
        x = self.dropout(x)
        output = self.block_fc3(x)
        
        return output

