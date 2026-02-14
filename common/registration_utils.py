"""
Registration utility module for MultiGradICON integration.
This module provides a unified interface for registration operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add multigradICON to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multigradICON', 'src'))

try:
    from unigradicon import get_model_from_model_zoo, make_sim
    import icon_registration as icon
except ImportError as e:
    print(f"Warning: Could not import MultiGradICON modules: {e}")
    print("Make sure multigradICON is properly installed.")


class RegistrationWrapper:
    """
    Wrapper for MultiGradICON registration model.
    Handles tensor conversion and registration inference.
    """
    def __init__(self, model_name='multigradicon', device='cuda', freeze=True):
        """
        Initialize registration wrapper.
        
        Args:
            model_name: 'multigradicon' or 'unigradicon'
            device: 'cuda' or 'cpu'
            freeze: Whether to freeze the model (default: True)
        """
        self.device = device
        self.model_name = model_name
        
        # Load MultiGradICON model
        try:
            loss_fn = make_sim("lncc")
            self.reg_model = get_model_from_model_zoo(model_name, loss_fn)
            self.reg_model.to(device)
            self.reg_model.eval()
            
            if freeze:
                for param in self.reg_model.parameters():
                    param.requires_grad = False
                    
            print(f"Loaded {model_name} registration model on {device}")
        except Exception as e:
            print(f"Error loading registration model: {e}")
            self.reg_model = None
    
    def register_pair(self, moving_img, fixed_img):
        """
        Register moving image to fixed image.
        
        Args:
            moving_img: Tensor [B, C, H, W] or [B, C, D, H, W]
            fixed_img: Tensor [B, C, H, W] or [B, C, D, H, W]
            
        Returns:
            dict with keys:
                - 'phi': deformation field
                - 'warped': warped moving image
                - 'quality_score': registration quality score (similarity)
        """
    def register_pair(self, moving_img, fixed_img):
        """
        无梯度的配准接口，用于评估/打分（例如给判别器提供标签）。

        该接口在 no_grad 环境下调用，并使用 .item() 提取标量，不会将梯度传回生成网络。
        """
        if self.reg_model is None:
            return None

        with torch.no_grad():
            # Ensure images are on correct device
            moving_img = moving_img.to(self.device)
            fixed_img = fixed_img.to(self.device)

            # Forward pass through registration model (no grad)
            result = self.reg_model(moving_img, fixed_img)

            # Extract deformation field and warped image from the registration module
            phi_AB = getattr(self.reg_model, "phi_AB_vectorfield", None)
            warped = getattr(self.reg_model, "warped_image_A", None)

            # Compute quality score (scalar similarity)
            if hasattr(result, "similarity_loss"):
                # use negative similarity loss as a scalar score (higher is better)
                quality_score = -result.similarity_loss.item()
            else:
                if warped is not None:
                    quality_score = self._compute_similarity(warped, fixed_img)
                else:
                    quality_score = 0.0

            return {
                "phi": phi_AB,
                "warped": warped,
                "quality_score": quality_score,
            }

    def register_pair_with_grad(self, moving_img, fixed_img):
        """
        带梯度的配准接口，用于训练时直接构造形变场损失。

        与 register_pair 不同：
            - 不使用 no_grad，不调用 .item()
            - 返回的字段均为 tensor，梯度可从形变场 / warped 图像回传到 moving_img（即生成图）。
        """
        if self.reg_model is None:
            return None

        # 保证在正确的设备上，且保持计算图
        moving_img = moving_img.to(self.device)
        fixed_img = fixed_img.to(self.device)

        # MultiGradICON 当前预训练权重默认是 3D 模型（identity_map 为 5D: [1, C, D, H, W]）。
        # 我们这边的生成结果是 2D 切片 [B, C, H, W]，需要在深度维度上扩展成“伪 3D”：[B, C, 1, H, W]，
        # 这样才能和 identity_map 的维度对齐，避免 forward 里的 assert 报错。
        if moving_img.dim() == 4:
            moving_img = moving_img.unsqueeze(2)  # [B, C, 1, H, W]
        if fixed_img.dim() == 4:
            fixed_img = fixed_img.unsqueeze(2)    # [B, C, 1, H, W]

        # 如果 MultiGradICON 内部预先构建了 identity_map，则需要保证空间尺寸一致
        # identity_map.shape: [1, D, H, W] 或 [1, D, D, H, W]，只关心后面的空间维度
        if hasattr(self.reg_model, "identity_map"):
            id_shape = self.reg_model.identity_map.shape
            img_shape = moving_img.shape
            if id_shape[2:] != img_shape[2:]:
                target_spatial = id_shape[2:]
                if moving_img.dim() == 5:  # [B, C, D, H, W] 3D（包括上面从 2D 扩展出的伪 3D）
                    moving_img = F.interpolate(
                        moving_img, size=target_spatial[-3:], mode="trilinear", align_corners=False
                    )
                    fixed_img = F.interpolate(
                        fixed_img, size=target_spatial[-3:], mode="trilinear", align_corners=False
                    )
                # 其他维度情况暂不处理，直接让后续 assert 报错以暴露问题

        # 前向传播，保留梯度
        result = self.reg_model(moving_img, fixed_img)

        # 从注册模块中读取形变场和配准结果
        phi_AB = getattr(self.reg_model, "phi_AB_vectorfield", None)
        warped = getattr(self.reg_model, "warped_image_A", None)
        sim_loss = getattr(result, "similarity_loss", None)

        return {
            "phi": phi_AB,
            "warped": warped,
            "similarity_loss": sim_loss,
        }
    
    def _compute_similarity(self, img1, img2):
        """Compute simple similarity metric (normalized cross-correlation)."""
        img1_flat = img1.flatten(1)
        img2_flat = img2.flatten(1)
        
        img1_norm = img1_flat - img1_flat.mean(dim=1, keepdim=True)
        img2_norm = img2_flat - img2_flat.mean(dim=1, keepdim=True)
        
        numerator = (img1_norm * img2_norm).sum(dim=1)
        denominator = torch.sqrt((img1_norm ** 2).sum(dim=1) * (img2_norm ** 2).sum(dim=1)) + 1e-8
        
        ncc = (numerator / denominator).mean()
        return ncc.item()


def compute_deformation_field_similarity(phi1, phi2):
    """
    Compute similarity between two deformation fields.
    
    Args:
        phi1: Deformation field 1 [B, D, H, W] or [B, D, D, H, W]
        phi2: Deformation field 2 [B, D, H, W] or [B, D, D, H, W]
        
    Returns:
        Similarity score (L2 distance, lower is more similar)
    """
    if phi1 is None or phi2 is None:
        return torch.tensor(0.0)
    
    diff = phi1 - phi2
    similarity = torch.mean(diff ** 2)
    return similarity

