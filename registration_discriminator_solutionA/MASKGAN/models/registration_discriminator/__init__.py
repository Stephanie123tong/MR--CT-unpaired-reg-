"""
Registration Discriminator Module for Solution A
"""

from .deformation_complexity import (
    compute_global_complexity,
    compute_local_complexity
)
from .registration_discriminator import RegistrationDiscriminator
from .registration_model_wrapper import RegistrationModelWrapper

__all__ = [
    'compute_global_complexity',
    'compute_local_complexity',
    'RegistrationDiscriminator',
    'RegistrationModelWrapper',
]

