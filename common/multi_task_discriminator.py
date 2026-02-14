"""
Multi-task discriminator for registration-aware CycleGAN.
Supports both standard adversarial task and registration quality regression.
"""
import torch
import torch.nn as nn
import functools


class MultiTaskDiscriminator(nn.Module):
    """
    Discriminator with multiple task heads:
    1. Adversarial head: real/fake classification
    2. Registration quality head: regression of registration quality score
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Construct a multi-task PatchGAN discriminator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(MultiTaskDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        # Shared backbone
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        self.shared_backbone = nn.Sequential(*sequence)
        
        # Task 1: Adversarial head (real/fake)
        self.adv_head = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        
        # Task 2: Registration quality regression head
        self.reg_quality_head = nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 64, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, input):
        """
        Standard forward.
        
        Returns:
            dict with keys:
                - 'adv': adversarial output [B, 1, H, W]
                - 'reg_quality': registration quality score [B, 1]
        """
        features = self.shared_backbone(input)
        adv_output = self.adv_head(features)
        reg_quality = self.reg_quality_head(features).squeeze(-1).squeeze(-1)  # [B, 1]
        if reg_quality.dim() == 1:
            reg_quality = reg_quality.unsqueeze(1)
        return {
            'adv': adv_output,
            'reg_quality': reg_quality
        }

