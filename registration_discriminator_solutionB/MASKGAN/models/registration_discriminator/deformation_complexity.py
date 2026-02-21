"""
Deformation Field Complexity Computation
"""

import torch
import torch.nn.functional as F


def compute_deformation_magnitude(phi):
    """Compute the magnitude of deformation field."""
    if phi.dim() == 4:  # 2D: [B, 2, H, W]
        magnitude = torch.sqrt(phi[:, 0:1, :, :]**2 + phi[:, 1:2, :, :]**2)
    elif phi.dim() == 5:  # 3D: [B, 3, H, W, D]
        magnitude = torch.sqrt(phi[:, 0:1, :, :, :]**2 + 
                              phi[:, 1:2, :, :, :]**2 + 
                              phi[:, 2:3, :, :, :]**2)
    else:
        raise ValueError(f"Unsupported phi dimension: {phi.dim()}")
    return magnitude


def compute_deformation_gradient(phi):
    """Compute the gradient (smoothness) of deformation field."""
    if phi.dim() == 4:  # 2D
        grad_x_x = phi[:, 0:1, :, 1:] - phi[:, 0:1, :, :-1]
        grad_x_y = phi[:, 0:1, 1:, :] - phi[:, 0:1, :-1, :]
        grad_y_x = phi[:, 1:2, :, 1:] - phi[:, 1:2, :, :-1]
        grad_y_y = phi[:, 1:2, 1:, :] - phi[:, 1:2, :-1, :]
        
        grad_x_x = F.pad(grad_x_x, (0, 1, 0, 0), mode='replicate')
        grad_x_y = F.pad(grad_x_y, (0, 0, 0, 1), mode='replicate')
        grad_y_x = F.pad(grad_y_x, (0, 1, 0, 0), mode='replicate')
        grad_y_y = F.pad(grad_y_y, (0, 0, 0, 1), mode='replicate')
        
        gradient = torch.sqrt(grad_x_x**2 + grad_x_y**2 + grad_y_x**2 + grad_y_y**2)
    else:
        raise NotImplementedError("3D gradient computation not implemented yet")
    return gradient


def compute_deformation_curvature(phi):
    """Compute the curvature (second derivative) of deformation field."""
    if phi.dim() == 4:  # 2D
        d2x_dx2 = phi[:, 0:1, :, 2:] - 2*phi[:, 0:1, :, 1:-1] + phi[:, 0:1, :, :-2]
        d2x_dy2 = phi[:, 0:1, 2:, :] - 2*phi[:, 0:1, 1:-1, :] + phi[:, 0:1, :-2, :]
        d2y_dx2 = phi[:, 1:2, :, 2:] - 2*phi[:, 1:2, :, 1:-1] + phi[:, 1:2, :, :-2]
        d2y_dy2 = phi[:, 1:2, 2:, :] - 2*phi[:, 1:2, 1:-1, :] + phi[:, 1:2, :-2, :]
        
        d2x_dx2 = F.pad(d2x_dx2, (1, 1, 0, 0), mode='replicate')
        d2x_dy2 = F.pad(d2x_dy2, (0, 0, 1, 1), mode='replicate')
        d2y_dx2 = F.pad(d2y_dx2, (1, 1, 0, 0), mode='replicate')
        d2y_dy2 = F.pad(d2y_dy2, (0, 0, 1, 1), mode='replicate')
        
        curvature = torch.sqrt(d2x_dx2**2 + d2x_dy2**2 + d2y_dx2**2 + d2y_dy2**2)
    else:
        raise NotImplementedError("3D curvature computation not implemented yet")
    return curvature


def compute_jacobian_determinant(phi):
    """Compute the Jacobian determinant of deformation field."""
    if phi.dim() == 4:  # 2D: [B, 2, H, W]
        B, _, H, W = phi.shape
        
        dphi_x_dx = phi[:, 0:1, :, 1:] - phi[:, 0:1, :, :-1]
        dphi_x_dx = F.pad(dphi_x_dx, (0, 1, 0, 0), mode='replicate')
        
        dphi_x_dy = phi[:, 0:1, 1:, :] - phi[:, 0:1, :-1, :]
        dphi_x_dy = F.pad(dphi_x_dy, (0, 0, 0, 1), mode='replicate')
        
        dphi_y_dx = phi[:, 1:2, :, 1:] - phi[:, 1:2, :, :-1]
        dphi_y_dx = F.pad(dphi_y_dx, (0, 1, 0, 0), mode='replicate')
        
        dphi_y_dy = phi[:, 1:2, 1:, :] - phi[:, 1:2, :-1, :]
        dphi_y_dy = F.pad(dphi_y_dy, (0, 0, 0, 1), mode='replicate')
        
        jacobian = (1 + dphi_x_dx) * (1 + dphi_y_dy) - dphi_x_dy * dphi_y_dx
    else:
        raise NotImplementedError("3D Jacobian computation not implemented yet")
    return jacobian


def compute_jacobian_anomaly(jacobian, threshold_low=0.5, threshold_high=2.0):
    """Compute the ratio of anomalous regions in Jacobian determinant."""
    anomaly_map = (jacobian < threshold_low) | (jacobian > threshold_high)
    anomaly_ratio = anomaly_map.float().mean(dim=[2, 3], keepdim=True)
    return anomaly_ratio, anomaly_map.float()


def compute_global_complexity(phi, alpha=1.0, beta=0.5, gamma=0.1, delta=0.5,
                             threshold_low=0.5, threshold_high=2.0):
    """Compute global complexity score from deformation field."""
    magnitude = compute_deformation_magnitude(phi)
    gradient = compute_deformation_gradient(phi)
    curvature = compute_deformation_curvature(phi)
    jacobian = compute_jacobian_determinant(phi)
    anomaly_ratio, _ = compute_jacobian_anomaly(jacobian, threshold_low, threshold_high)
    
    mean_magnitude = magnitude.mean(dim=[2, 3], keepdim=True)
    mean_gradient = gradient.mean(dim=[2, 3], keepdim=True)
    mean_curvature = curvature.mean(dim=[2, 3], keepdim=True)
    
    complexity = (alpha * mean_magnitude + 
                  beta * mean_gradient + 
                  gamma * mean_curvature + 
                  delta * anomaly_ratio)
    
    return complexity.squeeze(-1).squeeze(-1)  # [B]


def compute_local_complexity(phi, num_blocks=8, alpha=1.0, beta=0.5, gamma=0.5,
                            threshold_low=0.5, threshold_high=2.0):
    """Compute local complexity map by dividing image into blocks."""
    B, _, H, W = phi.shape
    
    magnitude = compute_deformation_magnitude(phi)
    gradient = compute_deformation_gradient(phi)
    jacobian = compute_jacobian_determinant(phi)
    _, anomaly_map = compute_jacobian_anomaly(jacobian, threshold_low, threshold_high)
    
    block_h = H // num_blocks
    block_w = W // num_blocks
    
    local_complexity = []
    
    for i in range(num_blocks):
        row_complexity = []
        for j in range(num_blocks):
            h_start = i * block_h
            h_end = (i + 1) * block_h if i < num_blocks - 1 else H
            w_start = j * block_w
            w_end = (j + 1) * block_w if j < num_blocks - 1 else W
            
            mag_block = magnitude[:, :, h_start:h_end, w_start:w_end]
            grad_block = gradient[:, :, h_start:h_end, w_start:w_end]
            anomaly_block = anomaly_map[:, :, h_start:h_end, w_start:w_end]
            
            block_complexity = (alpha * mag_block.mean() + 
                               beta * grad_block.mean() + 
                               gamma * anomaly_block.mean())
            row_complexity.append(block_complexity)
        
        local_complexity.append(torch.stack(row_complexity, dim=1))
    
    local_complexity = torch.stack(local_complexity, dim=2)
    return local_complexity


def compute_block_quality_scores(phi, warped_image, target_image, num_blocks=8,
                                 alpha=1.0, beta=0.5, gamma=0.1, delta=0.5,
                                 threshold_low=0.5, threshold_high=2.0):
    """
    Compute quality scores for each block based on registration results.
    
    Args:
        phi: Deformation field [B, 2, H, W]
        warped_image: Warped moving image [B, C, H, W]
        target_image: Target image [B, C, H, W]
        num_blocks: Number of blocks per dimension
        alpha, beta, gamma, delta: Weights for different complexity components
    
    Returns:
        block_features: [B, num_blocks*num_blocks, feature_dim] - Features for each block
        block_scores: [B, num_blocks*num_blocks] - Quality scores for each block
    """
    B, _, H, W = phi.shape
    _, C, _, _ = warped_image.shape
    
    # Compute deformation metrics
    magnitude = compute_deformation_magnitude(phi)  # [B, 1, H, W]
    gradient = compute_deformation_gradient(phi)     # [B, 1, H, W]
    jacobian = compute_jacobian_determinant(phi)     # [B, 1, H, W]
    _, anomaly_map = compute_jacobian_anomaly(jacobian, threshold_low, threshold_high)  # [B, 1, H, W]
    
    # Compute registration residual (difference between warped and target)
    registration_residual = torch.abs(warped_image - target_image)  # [B, C, H, W]
    registration_residual = registration_residual.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    
    block_h = H // num_blocks
    block_w = W // num_blocks
    
    block_features_list = []
    block_scores_list = []
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            h_start = i * block_h
            h_end = (i + 1) * block_h if i < num_blocks - 1 else H
            w_start = j * block_w
            w_end = (j + 1) * block_w if j < num_blocks - 1 else W
            
            # Extract block regions
            mag_block = magnitude[:, :, h_start:h_end, w_start:w_end]
            grad_block = gradient[:, :, h_start:h_end, w_start:w_end]
            anomaly_block = anomaly_map[:, :, h_start:h_end, w_start:w_end]
            residual_block = registration_residual[:, :, h_start:h_end, w_start:w_end]
            
            # Compute block-level metrics
            block_magnitude = mag_block.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1]
            block_gradient = grad_block.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1]
            block_anomaly = anomaly_block.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1]
            block_residual = residual_block.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1]
            
            # Combine into feature vector [B, 4]
            block_feature = torch.stack([
                block_residual.squeeze(-1).squeeze(-1),      # [B]
                block_magnitude.squeeze(-1).squeeze(-1),     # [B]
                block_gradient.squeeze(-1).squeeze(-1),      # [B]
                block_anomaly.squeeze(-1).squeeze(-1)        # [B]
            ], dim=1)  # [B, 4]
            
            # Compute quality score (higher is better)
            # Quality = - (weighted sum of complexity metrics)
            block_score = - (alpha * block_residual.squeeze(-1).squeeze(-1) +
                            beta * block_magnitude.squeeze(-1).squeeze(-1) +
                            gamma * block_gradient.squeeze(-1).squeeze(-1) +
                            delta * block_anomaly.squeeze(-1).squeeze(-1))
            
            block_features_list.append(block_feature)
            block_scores_list.append(block_score)
    
    # Stack: [B, num_blocks*num_blocks, 4]
    block_features = torch.stack(block_features_list, dim=1)
    # Stack: [B, num_blocks*num_blocks]
    block_scores = torch.stack(block_scores_list, dim=1)
    
    return block_features, block_scores