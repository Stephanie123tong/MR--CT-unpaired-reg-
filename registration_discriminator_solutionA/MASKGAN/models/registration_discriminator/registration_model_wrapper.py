"""
Registration Model Wrapper for multigradICON
"""

import torch
import torch.nn as nn
import sys
import os

# Add multigradICON to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Path structure: registration_discriminator -> models -> MASKGAN -> registration_discriminator_solutionA -> multigradICON
# From: registration_discriminator_solutionA/MASKGAN/models/registration_discriminator/
# To: registration_discriminator_solutionA/multigradICON/
multigradicon_src_path = os.path.abspath(os.path.join(current_dir, '../../../multigradICON/src'))
multigradicon_root_path = os.path.abspath(os.path.join(current_dir, '../../../multigradICON'))
if os.path.exists(multigradicon_src_path):
    if multigradicon_src_path not in sys.path:
        sys.path.insert(0, multigradicon_src_path)
if os.path.exists(multigradicon_root_path):
    if multigradicon_root_path not in sys.path:
        sys.path.insert(0, multigradicon_root_path)


class RegistrationModelWrapper(nn.Module):
    """
    Wrapper for multigradICON registration model.
    Solution 2: Expands 2D images to 3D to use 3D multigradICON with pretrained weights.
    """
    
    def __init__(self, registration_model=None, input_shape=None, use_3d_expansion=True, device=None):
        super(RegistrationModelWrapper, self).__init__()
        self.use_3d_expansion = use_3d_expansion
        self.device = device
        
        if registration_model is None:
            # Try to load multigradICON
            try:
                from unigradicon import get_model_from_model_zoo
                import icon_registration as icon
                from icon_registration import config
                
                # Set device for multigradICON if provided
                if device is not None:
                    config.device = device
                
                if use_3d_expansion:
                    # Solution 2: Use 3D multigradICON with 2D->3D expansion
                    print("Loading multigradICON (3D) for 2D images using expansion method...")
                    print("2D images will be expanded to 3D (depth=1) to use pretrained 3D weights.")
                    
                    # Use default 3D input_shape for multigradICON
                    # The actual 2D images will be expanded in forward()
                    self.registration_model = get_model_from_model_zoo(
                        "multigradicon",
                        loss_fn=icon.LNCC(sigma=5),
                        apply_intensity_conservation_loss=False
                    )
                    # Move to device if specified
                    if device is not None:
                        self.registration_model = self.registration_model.to(device)
                    print("Successfully loaded multigradICON with pretrained weights.")
                else:
                    # Try to create 2D network (no pretrained weights)
                    print("Warning: 2D multigradICON mode selected but no pretrained weights available.")
                    print("Using dummy model. For best results, use use_3d_expansion=True.")
                    self.registration_model = DummyRegistrationModel(input_shape)
                
            except ImportError as e:
                print(f"Warning: multigradICON not found ({e}). Using dummy model.")
                self.registration_model = DummyRegistrationModel(input_shape)
            except Exception as e:
                print(f"Warning: Failed to load multigradICON ({e}). Using dummy model.")
                import traceback
                traceback.print_exc()
                self.registration_model = DummyRegistrationModel(input_shape)
        else:
            self.registration_model = registration_model
        
        # Freeze registration model
        if self.registration_model is not None and not isinstance(self.registration_model, DummyRegistrationModel):
            for param in self.registration_model.parameters():
                param.requires_grad = False
            self.registration_model.eval()
    
    def _expand_2d_to_3d(self, image_2d):
        """
        Expand 2D image to 3D by adding a depth dimension.
        
        Args:
            image_2d: [B, C, H, W]
        
        Returns:
            image_3d: [B, C, 1, H, W] (depth dimension = 1)
        """
        if image_2d.dim() == 4:  # [B, C, H, W]
            return image_2d.unsqueeze(2)  # [B, C, 1, H, W]
        elif image_2d.dim() == 5:  # Already 3D
            return image_2d
        else:
            raise ValueError(f"Unsupported image dimension: {image_2d.dim()}")
    
    def _extract_2d_from_3d_deformation(self, phi_3d):
        """
        Extract 2D deformation field from 3D deformation field.
        
        Args:
            phi_3d: [B, 3, D, H, W] - 3D deformation field (x, y, z directions)
        
        Returns:
            phi_2d: [B, 2, H, W] - 2D deformation field (x, y directions)
        """
        if phi_3d.dim() == 5:  # [B, 3, D, H, W]
            # Extract x and y components, remove depth dimension
            # phi_3d[:, 0, :, :, :] is x-direction
            # phi_3d[:, 1, :, :, :] is y-direction
            # phi_3d[:, 2, :, :, :] is z-direction (not needed for 2D)
            
            # Take the middle slice in depth dimension (or average if D > 1)
            if phi_3d.shape[2] == 1:
                # Single depth slice
                phi_2d = phi_3d[:, :2, 0, :, :]  # [B, 2, H, W]
            else:
                # Multiple depth slices, take middle one
                mid_depth = phi_3d.shape[2] // 2
                phi_2d = phi_3d[:, :2, mid_depth, :, :]  # [B, 2, H, W]
            
            return phi_2d
        elif phi_3d.dim() == 4:  # Already 2D [B, 2, H, W]
            return phi_3d
        else:
            raise ValueError(f"Unsupported deformation field dimension: {phi_3d.dim()}")
    
    def forward(self, image_A, image_B):
        """
        Register image_A to image_B.
        
        Args:
            image_A: [B, C, H, W] - Moving image (2D)
            image_B: [B, C, H, W] - Fixed image (2D)
        
        Returns:
            result: Object with phi_AB_vectorfield attribute [B, 2, H, W]
        """
        # Handle dummy model
        if isinstance(self.registration_model, DummyRegistrationModel):
            return DummyRegistrationModel.forward_static(image_A, image_B)
        
        # Solution 2: Expand 2D to 3D if needed
        if self.use_3d_expansion and image_A.dim() == 4:
            # Expand 2D images to 3D
            image_A_3d = self._expand_2d_to_3d(image_A)  # [B, C, 1, H, W]
            image_B_3d = self._expand_2d_to_3d(image_B)  # [B, C, 1, H, W]
            
            # Ensure images are on the same device as registration model
            if self.device is not None:
                image_A_3d = image_A_3d.to(self.device)
                image_B_3d = image_B_3d.to(self.device)
            
            # Register using 3D multigradICON
            with torch.no_grad():
                result_3d = self.registration_model(image_A_3d, image_B_3d)
            
            # Extract 2D deformation field from 3D result
            phi_3d = result_3d.phi_AB_vectorfield  # [B, 3, 1, H, W] or [B, 3, D, H, W]
            phi_2d = self._extract_2d_from_3d_deformation(phi_3d)  # [B, 2, H, W]
            
            # Ensure phi_2d is on the same device as input
            phi_2d = phi_2d.to(image_A.device)
            
            # Create result object with 2D deformation field
            class Result:
                def __init__(self):
                    self.phi_AB_vectorfield = phi_2d
            
            return Result()
        else:
            # Direct registration (already 3D or using 2D model)
            with torch.no_grad():
                result = self.registration_model(image_A, image_B)
            
            # Ensure phi_AB_vectorfield exists and is 2D
            if not hasattr(result, 'phi_AB_vectorfield'):
                class Result:
                    pass
                r = Result()
                if isinstance(result, dict):
                    phi = result.get('phi_AB_vectorfield', result.get('phi', None))
                elif isinstance(result, tuple):
                    phi = result[0] if len(result) > 0 else None
                else:
                    raise ValueError("Registration model must return phi_AB_vectorfield")
                
                # Convert to 2D if needed
                if phi is not None and phi.dim() == 5:
                    phi = self._extract_2d_from_3d_deformation(phi)
                r.phi_AB_vectorfield = phi
                return r
            
            # Convert phi_AB_vectorfield to 2D if it's 3D
            if result.phi_AB_vectorfield.dim() == 5:
                result.phi_AB_vectorfield = self._extract_2d_from_3d_deformation(result.phi_AB_vectorfield)
            
            return result


class DummyRegistrationModel(nn.Module):
    """Dummy registration model for testing (returns identity deformation)."""
    
    def __init__(self, input_shape=None):
        super(DummyRegistrationModel, self).__init__()
        self.input_shape = input_shape
    
    @staticmethod
    def forward_static(image_A, image_B):
        """Static method to create identity deformation field."""
        B, C, H, W = image_A.shape
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=image_A.device, dtype=image_A.dtype),
            torch.arange(W, device=image_A.device, dtype=image_A.dtype),
            indexing='ij'
        )
        
        phi = torch.stack([x_coords, y_coords], dim=0)
        phi = phi.unsqueeze(0).repeat(B, 1, 1, 1)
        
        class Result:
            def __init__(self):
                self.phi_AB_vectorfield = phi
        
        return Result()
    
    def forward(self, image_A, image_B):
        return self.forward_static(image_A, image_B)

