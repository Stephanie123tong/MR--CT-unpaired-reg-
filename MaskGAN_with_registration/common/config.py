"""
Configuration system for ablation study versions.
Each version enables different registration modules.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegistrationConfig:
    """Configuration for registration modules."""
    # Module 1: Intra-domain self-supervised registration
    enable_module1: bool = False
    
    # Module 2: Cycle-registration consistency
    enable_module2: bool = False
    
    # Module 3: Pseudo-pair registration for discriminator
    enable_module3: bool = False
    
    # Loss weights
    lambda_geo: float = 1.0  # Weight for deformation field consistency
    lambda_cycle_reg: float = 1.0  # Weight for cycle registration consistency
    lambda_reg_quality: float = 1.0  # Weight for registration quality regression
    
    # Registration model settings
    reg_model_name: str = 'multigradicon'  # 'multigradicon' or 'unigradicon'
    reg_device: str = 'cuda'
    
    # Frequency of registration operations (to save computation)
    reg_frequency: int = 1  # Perform registration every N iterations


# Version configurations
VERSION_CONFIGS = {
    'baseline': RegistrationConfig(
        enable_module1=False,
        enable_module2=False,
        enable_module3=False,
    ),
    'version1': RegistrationConfig(
        enable_module1=True,
        enable_module2=False,
        enable_module3=False,
        lambda_geo=1.0,
    ),
    'version2': RegistrationConfig(
        enable_module1=False,
        enable_module2=True,
        enable_module3=False,
        lambda_cycle_reg=1.0,
    ),
    'version3': RegistrationConfig(
        enable_module1=False,
        enable_module2=False,
        enable_module3=True,
        lambda_reg_quality=1.0,
    ),
    'version4': RegistrationConfig(
        enable_module1=True,
        enable_module2=True,
        enable_module3=False,
        lambda_geo=1.0,
        lambda_cycle_reg=1.0,
    ),
    'version5': RegistrationConfig(
        enable_module1=True,
        enable_module2=False,
        enable_module3=True,
        lambda_geo=1.0,
        lambda_reg_quality=1.0,
    ),
    'version6': RegistrationConfig(
        enable_module1=False,
        enable_module2=True,
        enable_module3=True,
        lambda_cycle_reg=1.0,
        lambda_reg_quality=1.0,
    ),
    'version7': RegistrationConfig(
        enable_module1=True,
        enable_module2=True,
        enable_module3=True,
        lambda_geo=1.0,
        lambda_cycle_reg=1.0,
        lambda_reg_quality=1.0,
    ),
}


def get_config(version: str) -> RegistrationConfig:
    """Get configuration for a specific version."""
    if version not in VERSION_CONFIGS:
        raise ValueError(f"Unknown version: {version}. Available: {list(VERSION_CONFIGS.keys())}")
    return VERSION_CONFIGS[version]

