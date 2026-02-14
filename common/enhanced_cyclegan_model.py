"""
Enhanced CycleGAN model with registration modules support.
This is a template that can be imported and customized for each version.
"""
import torch
import itertools
import os
import sys
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# Add common to path
common_path = os.path.join(os.path.dirname(__file__), '..', 'common')
if common_path not in sys.path:
    sys.path.insert(0, common_path)

try:
    from config import get_config
    from registration_utils import RegistrationWrapper, compute_deformation_field_similarity
    from multi_task_discriminator import MultiTaskDiscriminator
except ImportError as e:
    print(f"Warning: Could not import registration modules: {e}")
    get_config = None
    RegistrationWrapper = None
    compute_deformation_field_similarity = None
    MultiTaskDiscriminator = None


def create_enhanced_cyclegan_model(base_class, version_name):
    """
    Create an enhanced CycleGAN model class for a specific version.
    
    Args:
        base_class: Base CycleGAN model class
        version_name: Version name ('baseline', 'version1', etc.)
    
    Returns:
        Enhanced model class
    """
    
    class EnhancedCycleGANModel(base_class):
        @staticmethod
        def modify_commandline_options(parser, is_train=True):
            parser = base_class.modify_commandline_options(parser, is_train)
            if is_train:
                parser.add_argument('--ablation_version', type=str, default=version_name,
                                  help='Ablation study version')
            return parser
        
        def __init__(self, opt):
            # Detect version
            version = getattr(opt, 'ablation_version', version_name)
            if 'baseline' in version or version == 'baseline':
                version = 'baseline'
            elif 'version1' in version:
                version = 'version1'
            elif 'version2' in version:
                version = 'version2'
            elif 'version3' in version:
                version = 'version3'
            elif 'version4' in version:
                version = 'version4'
            elif 'version5' in version:
                version = 'version5'
            elif 'version6' in version:
                version = 'version6'
            elif 'version7' in version:
                version = 'version7'
            else:
                version = 'baseline'
            
            # Get configuration
            if get_config is not None:
                try:
                    self.reg_config = get_config(version)
                except:
                    self.reg_config = None
            else:
                self.reg_config = None
            
            # Initialize base model
            base_class.__init__(self, opt)
            
            # Initialize registration wrapper if needed
            self.reg_wrapper = None
            if (self.reg_config is not None and 
                (self.reg_config.enable_module1 or 
                 self.reg_config.enable_module2 or 
                 self.reg_config.enable_module3) and
                RegistrationWrapper is not None and self.isTrain):
                try:
                    self.reg_wrapper = RegistrationWrapper(
                        model_name=self.reg_config.reg_model_name,
                        device=self.device,
                        freeze=True
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize registration wrapper: {e}")
                    self.reg_wrapper = None
            
            # Update loss names
            if self.reg_config is not None:
                if self.reg_config.enable_module1:
                    if 'geo' not in self.loss_names:
                        self.loss_names.append('geo')
                if self.reg_config.enable_module2:
                    if 'cycle_reg' not in self.loss_names:
                        self.loss_names.append('cycle_reg')
                if self.reg_config.enable_module3:
                    if 'reg_quality' not in self.loss_names:
                        self.loss_names.append('reg_quality')
            
            # Replace discriminators with multi-task version if Module 3 is enabled
            if (self.reg_config is not None and 
                self.reg_config.enable_module3 and 
                MultiTaskDiscriminator is not None and 
                self.isTrain):
                # Replace D_B with multi-task discriminator
                if hasattr(self, 'netD_B'):
                    # Get input channels
                    input_nc = opt.input_nc
                    ndf = opt.ndf
                    n_layers_D = opt.n_layers_D
                    norm = networks.get_norm_layer(opt.norm)
                    
                    # Create multi-task discriminator
                    mt_disc = MultiTaskDiscriminator(
                        input_nc=input_nc,
                        ndf=ndf,
                        n_layers=n_layers_D if hasattr(opt, 'n_layers_D') else 3,
                        norm_layer=norm
                    )
                    mt_disc = networks.init_net(mt_disc, opt.init_type, opt.init_gain, self.gpu_ids)
                    self.netD_B = mt_disc
                    
                    # Also replace D_A if needed (for symmetry)
                    output_nc = opt.output_nc
                    mt_disc_A = MultiTaskDiscriminator(
                        input_nc=output_nc,
                        ndf=ndf,
                        n_layers=n_layers_D if hasattr(opt, 'n_layers_D') else 3,
                        norm_layer=norm
                    )
                    mt_disc_A = networks.init_net(mt_disc_A, opt.init_type, opt.init_gain, self.gpu_ids)
                    self.netD_A = mt_disc_A
        
        def backward_G(self):
            """Calculate the loss for generators with registration modules."""
            # Call base backward_G first
            base_class.backward_G(self)
            
            # Add registration losses if enabled
            if self.reg_config is None or self.reg_wrapper is None:
                self.loss_geo = 0
                self.loss_cycle_reg = 0
                return
            
            # Module 1: Intra-domain self-supervised registration
            if self.reg_config.enable_module1:
                self.loss_geo = self._compute_module1_loss()
            else:
                self.loss_geo = 0
            
            # Module 2: Cycle-registration consistency
            if self.reg_config.enable_module2:
                self.loss_cycle_reg = self._compute_module2_loss()
            else:
                self.loss_cycle_reg = 0
            
            # Add to total loss
            self.loss_G = self.loss_G + self.loss_geo + self.loss_cycle_reg
        
        def _compute_module1_loss(self):
            """Module 1: Intra-domain self-supervised registration loss."""
            if self.real_A.size(0) < 2:
                return torch.tensor(0.0, device=self.device)
            
            try:
                # Get two samples from batch
                A1 = self.real_A[0:1]
                A2 = self.real_A[1:2] if self.real_A.size(0) > 1 else self.real_A[0:1]
                B_fake1 = self.fake_B[0:1]
                B_fake2 = self.fake_B[1:2] if self.fake_B.size(0) > 1 else self.fake_B[0:1]
                
                # Register within MR domain
                reg_MR = self.reg_wrapper.register_pair(A1, A2)
                # Register within generated CT domain
                reg_CT = self.reg_wrapper.register_pair(B_fake1, B_fake2)
                
                if reg_MR and reg_CT and reg_MR['phi'] is not None and reg_CT['phi'] is not None:
                    # Compute deformation field similarity
                    loss_geo = compute_deformation_field_similarity(
                        reg_MR['phi'], reg_CT['phi']
                    ) * self.reg_config.lambda_geo
                    return loss_geo
            except Exception as e:
                print(f"Warning: Module 1 computation failed: {e}")
            
            return torch.tensor(0.0, device=self.device)
        
        def _compute_module2_loss(self):
            """Module 2: Cycle-registration consistency loss."""
            try:
                # Register B_fake back to A
                reg_cycle_A = self.reg_wrapper.register_pair(self.fake_B, self.real_A)
                # Register A_fake back to B
                reg_cycle_B = self.reg_wrapper.register_pair(self.fake_A, self.real_B)
                
                loss_cycle_reg = torch.tensor(0.0, device=self.device)
                
                if reg_cycle_A and reg_cycle_A['quality_score'] is not None:
                    # Maximize registration quality (minimize negative quality)
                    quality_A = reg_cycle_A['quality_score']
                    if isinstance(quality_A, (int, float)):
                        loss_cycle_reg = loss_cycle_reg - torch.tensor(
                            quality_A, device=self.device
                        ) * self.reg_config.lambda_cycle_reg
                    else:
                        loss_cycle_reg = loss_cycle_reg - quality_A * self.reg_config.lambda_cycle_reg
                
                if reg_cycle_B and reg_cycle_B['quality_score'] is not None:
                    quality_B = reg_cycle_B['quality_score']
                    if isinstance(quality_B, (int, float)):
                        loss_cycle_reg = loss_cycle_reg - torch.tensor(
                            quality_B, device=self.device
                        ) * self.reg_config.lambda_cycle_reg
                    else:
                        loss_cycle_reg = loss_cycle_reg - quality_B * self.reg_config.lambda_cycle_reg
                
                return loss_cycle_reg
            except Exception as e:
                print(f"Warning: Module 2 computation failed: {e}")
            
            return torch.tensor(0.0, device=self.device)
        
        def backward_D_B(self):
            """Calculate GAN loss for discriminator D_B with Module 3 support."""
            fake_A = self.fake_A_pool.query(self.fake_A)
            
            if (self.reg_config is not None and 
                self.reg_config.enable_module3 and 
                isinstance(self.netD_B, MultiTaskDiscriminator)):
                # Multi-task discriminator
                pred_real = self.netD_B(self.real_A)
                pred_fake = self.netD_B(fake_A.detach())
                
                # Adversarial loss
                if isinstance(pred_real, dict):
                    loss_D_real_adv = self.criterionGAN(pred_real['adv'], True)
                    loss_D_fake_adv = self.criterionGAN(pred_fake['adv'], False)
                    loss_D_adv = (loss_D_real_adv + loss_D_fake_adv) * 0.5
                    
                    # Registration quality regression loss
                    if self.reg_wrapper is not None:
                        try:
                            # Get registration quality scores for real pairs
                            if self.real_A.size(0) >= 2:
                                reg_real = self.reg_wrapper.register_pair(
                                    self.real_A[0:1], 
                                    self.real_A[1:2]
                                )
                            else:
                                reg_real = None
                            
                            # Get registration quality for fake-real pair
                            reg_fake = self.reg_wrapper.register_pair(
                                fake_A[0:1], 
                                self.real_A[0:1]
                            )
                            
                            loss_reg_quality = torch.tensor(0.0, device=self.device)
                            
                            if reg_real and reg_real['quality_score'] is not None:
                                target_real = torch.tensor(
                                    reg_real['quality_score'], 
                                    device=self.device
                                ).unsqueeze(0)
                                if pred_real['reg_quality'].dim() > 1:
                                    target_real = target_real.expand_as(pred_real['reg_quality'][0:1])
                                loss_reg_quality = loss_reg_quality + torch.nn.functional.mse_loss(
                                    pred_real['reg_quality'][0:1], target_real
                                )
                            
                            if reg_fake and reg_fake['quality_score'] is not None:
                                target_fake = torch.tensor(
                                    reg_fake['quality_score'], 
                                    device=self.device
                                ).unsqueeze(0)
                                if pred_fake['reg_quality'].dim() > 1:
                                    target_fake = target_fake.expand_as(pred_fake['reg_quality'][0:1])
                                loss_reg_quality = loss_reg_quality + torch.nn.functional.mse_loss(
                                    pred_fake['reg_quality'][0:1], target_fake
                                )
                            
                            loss_reg_quality = loss_reg_quality * self.reg_config.lambda_reg_quality
                            self.loss_D_B = loss_D_adv + loss_reg_quality
                        except Exception as e:
                            print(f"Warning: Module 3 computation failed: {e}")
                            self.loss_D_B = loss_D_adv
                    else:
                        self.loss_D_B = loss_D_adv
                else:
                    # Fallback to standard
                    self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
            else:
                # Standard discriminator
                self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        
        def backward_D_A(self):
            """Calculate GAN loss for discriminator D_A with Module 3 support."""
            fake_B = self.fake_B_pool.query(self.fake_B)
            
            if (self.reg_config is not None and 
                self.reg_config.enable_module3 and 
                isinstance(self.netD_A, MultiTaskDiscriminator)):
                # Multi-task discriminator (similar to D_B)
                pred_real = self.netD_A(self.real_B)
                pred_fake = self.netD_A(fake_B.detach())
                
                if isinstance(pred_real, dict):
                    loss_D_real_adv = self.criterionGAN(pred_real['adv'], True)
                    loss_D_fake_adv = self.criterionGAN(pred_fake['adv'], False)
                    loss_D_adv = (loss_D_real_adv + loss_D_fake_adv) * 0.5
                    
                    # Registration quality regression (similar to D_B)
                    if self.reg_wrapper is not None:
                        try:
                            if self.real_B.size(0) >= 2:
                                reg_real = self.reg_wrapper.register_pair(
                                    self.real_B[0:1], 
                                    self.real_B[1:2]
                                )
                            else:
                                reg_real = None
                            
                            reg_fake = self.reg_wrapper.register_pair(
                                fake_B[0:1], 
                                self.real_B[0:1]
                            )
                            
                            loss_reg_quality = torch.tensor(0.0, device=self.device)
                            
                            if reg_real and reg_real['quality_score'] is not None:
                                target_real = torch.tensor(
                                    reg_real['quality_score'], 
                                    device=self.device
                                ).unsqueeze(0)
                                if pred_real['reg_quality'].dim() > 1:
                                    target_real = target_real.expand_as(pred_real['reg_quality'][0:1])
                                loss_reg_quality = loss_reg_quality + torch.nn.functional.mse_loss(
                                    pred_real['reg_quality'][0:1], target_real
                                )
                            
                            if reg_fake and reg_fake['quality_score'] is not None:
                                target_fake = torch.tensor(
                                    reg_fake['quality_score'], 
                                    device=self.device
                                ).unsqueeze(0)
                                if pred_fake['reg_quality'].dim() > 1:
                                    target_fake = target_fake.expand_as(pred_fake['reg_quality'][0:1])
                                loss_reg_quality = loss_reg_quality + torch.nn.functional.mse_loss(
                                    pred_fake['reg_quality'][0:1], target_fake
                                )
                            
                            loss_reg_quality = loss_reg_quality * self.reg_config.lambda_reg_quality
                            self.loss_D_A = loss_D_adv + loss_reg_quality
                        except Exception as e:
                            print(f"Warning: Module 3 computation failed for D_A: {e}")
                            self.loss_D_A = loss_D_adv
                    else:
                        self.loss_D_A = loss_D_adv
                else:
                    self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
            else:
                self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    
    return EnhancedCycleGANModel

