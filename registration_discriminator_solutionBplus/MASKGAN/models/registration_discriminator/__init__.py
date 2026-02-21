"""
Registration Discriminator Module for Solution B (Block-level)
"""

from .deformation_complexity import (
    compute_global_complexity,
    compute_local_complexity,
    compute_block_quality_scores
)
from .registration_discriminator import RegistrationDiscriminator
from .registration_model_wrapper import RegistrationModelWrapper

__all__ = [
    'compute_global_complexity',
    'compute_local_complexity',
    'compute_block_quality_scores',
    'RegistrationDiscriminator',
    'RegistrationModelWrapper',
]

