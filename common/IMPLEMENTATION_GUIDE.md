# Implementation Guide for Ablation Study Versions

This guide explains how to modify the existing CycleGAN code to support different ablation study versions with minimal code duplication.

## Overview

Instead of duplicating code for each version, we use a **configuration-based approach**:
1. Import common utilities from `common/` directory
2. Add registration modules conditionally based on configuration
3. Modify loss functions to include registration losses when enabled

## Step-by-Step Implementation

### Step 1: Add Common Utilities to Path

In each version's model file (e.g., `models/cycle_gan_model.py`), add at the top:

```python
import sys
import os
# Add common directory to path
common_path = os.path.join(os.path.dirname(__file__), '..', '..', 'common')
if common_path not in sys.path:
    sys.path.insert(0, common_path)

from config import get_config
from registration_utils import RegistrationWrapper, compute_deformation_field_similarity
```

### Step 2: Determine Version from Directory Name

In `__init__` method of CycleGANModel, detect version:

```python
def __init__(self, opt):
    # ... existing code ...
    
    # Detect version from directory structure or opt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if 'baseline' in current_dir:
        version = 'baseline'
    elif 'version1' in current_dir:
        version = 'version1'
    # ... etc
    
    # Or pass version via opt:
    version = getattr(opt, 'ablation_version', 'baseline')
    
    # Get configuration
    self.reg_config = get_config(version)
    
    # Initialize registration wrapper if any module is enabled
    if (self.reg_config.enable_module1 or 
        self.reg_config.enable_module2 or 
        self.reg_config.enable_module3):
        self.reg_wrapper = RegistrationWrapper(
            model_name=self.reg_config.reg_model_name,
            device=self.device,
            freeze=True
        )
    else:
        self.reg_wrapper = None
```

### Step 3: Modify Discriminator for Module 3 (Multi-task)

If `enable_module3` is True, modify discriminator to have multiple heads.

In `models/networks.py`, add a new discriminator class:

```python
class MultiTaskDiscriminator(nn.Module):
    """Discriminator with multiple task heads."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Shared backbone (same as NLayerDiscriminator)
        # ... create shared layers ...
        
        # Task 1: Adversarial head (real/fake)
        self.adv_head = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        
        # Task 2: Registration quality regression head
        self.reg_quality_head = nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 64, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def forward(self, input):
        features = self.shared_backbone(input)
        adv_output = self.adv_head(features)
        reg_quality = self.reg_quality_head(features).squeeze()
        return {
            'adv': adv_output,
            'reg_quality': reg_quality
        }
```

Then in `cycle_gan_model.py`, conditionally use MultiTaskDiscriminator:

```python
if self.isTrain:
    if self.reg_config.enable_module3:
        # Use multi-task discriminator
        self.netD_B = MultiTaskDiscriminator(...)
    else:
        # Use standard discriminator
        self.netD_B = networks.define_D(...)
```

### Step 4: Add Registration Losses to Generator

Modify `backward_G` method to include registration losses:

```python
def backward_G(self):
    # ... existing code for standard losses ...
    
    # Module 1: Intra-domain self-supervised registration
    if self.reg_config.enable_module1 and self.reg_wrapper is not None:
        # Get two samples from batch (assuming batch_size >= 2)
        if self.real_A.size(0) >= 2:
            A1, A2 = self.real_A[0:1], self.real_A[1:2]
            B_fake1, B_fake2 = self.fake_B[0:1], self.fake_B[1:2]
            
            # Register within MR domain
            reg_MR = self.reg_wrapper.register_pair(A1, A2)
            # Register within generated CT domain
            reg_CT = self.reg_wrapper.register_pair(B_fake1, B_fake2)
            
            if reg_MR and reg_CT:
                # Compute deformation field similarity
                self.loss_geo = compute_deformation_field_similarity(
                    reg_MR['phi'], reg_CT['phi']
                ) * self.reg_config.lambda_geo
            else:
                self.loss_geo = 0
        else:
            self.loss_geo = 0
    else:
        self.loss_geo = 0
    
    # Module 2: Cycle-registration consistency
    if self.reg_config.enable_module2 and self.reg_wrapper is not None:
        # Register B_fake back to A
        reg_cycle_A = self.reg_wrapper.register_pair(self.fake_B, self.real_A)
        # Register A_fake back to B
        reg_cycle_B = self.reg_wrapper.register_pair(self.fake_A, self.real_B)
        
        if reg_cycle_A and reg_cycle_B:
            # Maximize registration quality (minimize negative quality)
            self.loss_cycle_reg = (
                -reg_cycle_A['quality_score'] * self.reg_config.lambda_cycle_reg +
                -reg_cycle_B['quality_score'] * self.reg_config.lambda_cycle_reg
            )
        else:
            self.loss_cycle_reg = 0
    else:
        self.loss_cycle_reg = 0
    
    # Combine all losses
    self.loss_G = (self.loss_G_A + self.loss_G_B + 
                   self.loss_cycle_A + self.loss_cycle_B +
                   self.loss_idt_A + self.loss_idt_B +
                   self.loss_geo + self.loss_cycle_reg)
```

### Step 5: Modify Discriminator Loss for Module 3

Modify `backward_D_B` (and `backward_D_A` similarly):

```python
def backward_D_B(self):
    """Calculate GAN loss for discriminator D_B"""
    fake_A = self.fake_A_pool.query(self.fake_A)
    
    if self.reg_config.enable_module3 and self.reg_wrapper is not None:
        # Multi-task discriminator
        pred_real = self.netD_B(self.real_A)
        pred_fake = self.netD_B(fake_A.detach())
        
        # Adversarial loss
        loss_D_real_adv = self.criterionGAN(pred_real['adv'], True)
        loss_D_fake_adv = self.criterionGAN(pred_fake['adv'], False)
        loss_D_adv = (loss_D_real_adv + loss_D_fake_adv) * 0.5
        
        # Registration quality regression loss
        # Get registration quality scores
        reg_real = self.reg_wrapper.register_pair(
            self.real_A[0:1], 
            torch.roll(self.real_A, 1, dims=0)[0:1]  # Pair with another real sample
        )
        reg_fake = self.reg_wrapper.register_pair(fake_A[0:1], self.real_A[0:1])
        
        if reg_real and reg_fake:
            target_real_quality = reg_real['quality_score']
            target_fake_quality = reg_fake['quality_score']
            
            loss_reg_quality = (
                torch.nn.functional.mse_loss(
                    pred_real['reg_quality'], 
                    torch.tensor(target_real_quality, device=self.device)
                ) +
                torch.nn.functional.mse_loss(
                    pred_fake['reg_quality'],
                    torch.tensor(target_fake_quality, device=self.device)
                )
            ) * self.reg_config.lambda_reg_quality
        else:
            loss_reg_quality = 0
        
        self.loss_D_B = loss_D_adv + loss_reg_quality
    else:
        # Standard discriminator
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
```

### Step 6: Update Loss Names

Add new loss names to `loss_names` list:

```python
self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
if self.reg_config.enable_module1:
    self.loss_names.append('geo')
if self.reg_config.enable_module2:
    self.loss_names.append('cycle_reg')
if self.reg_config.enable_module3:
    self.loss_names.append('reg_quality')
```

### Step 7: Copy and Configure Each Version

For each version folder:

1. Copy baseline code to version folder
2. Modify `models/cycle_gan_model.py` following steps above
3. If Module 3 is enabled, also modify `models/networks.py` to add MultiTaskDiscriminator
4. Create a simple script or config file indicating the version

## Notes

- Registration operations are computationally expensive. Consider setting `reg_frequency > 1` to perform registration every N iterations.
- MultiGradICON expects specific input formats. You may need to adjust tensor shapes/preprocessing.
- Test each version incrementally to ensure correctness.

## File Structure

```
MaskGAN_ablation/
├── baseline/          # Original code (no changes)
├── version1/         # + Module 1
│   └── models/
│       └── cycle_gan_model.py  # Modified
├── version2/         # + Module 2
├── ...
└── common/           # Shared utilities (symlink or copy)
```

