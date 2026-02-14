import torch
import torch.nn.functional as F
import itertools
import sys
import os
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# 尝试导入 apex 以使用混合精度；若不可用，则退回普通 FP32 训练
try:
    from apex import amp  # type: ignore
    USE_APEX = True
except ImportError:
    amp = None
    USE_APEX = False

# 将上级目录下的 common 模块加入路径，方便导入 RegistrationWrapper
common_path = os.path.join(os.path.dirname(__file__), "..", "..", "common")
if common_path not in sys.path:
    sys.path.insert(0, common_path)

try:
    from registration_utils import RegistrationWrapper
except ImportError:
    RegistrationWrapper = None


class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--n_attentions', type=int, default=5, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--argmax',  action='store_true', default=False, help='Only select max value in the attention')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_mask', type=float, default=0.5, help='attention mask')
            parser.add_argument('--lambda_shape', type=float, default=0, help='attention mask')
            parser.add_argument('--lambda_co_A', type=float, default=2, help='weight for correlation coefficient loss (A -> B)')
            parser.add_argument('--lambda_co_B', type=float, default=2,
                                help='weight for correlation coefficient loss (B -> A )')
            # 形变场配准正则相关参数
            parser.add_argument('--lambda_reg', type=float, default=0.0,
                                help='weight for deformation-field based registration loss')
            parser.add_argument('--reg_patch_size', type=int, default=16,
                                help='patch size for local deformation pooling')
            parser.add_argument('--reg_topk_ratio', type=float, default=0.25,
                                help='ratio of top-k patches used for local deformation loss')
            parser.add_argument('--reg_alpha', type=float, default=1.0,
                                help='weight for global deformation magnitude term')
            parser.add_argument('--reg_beta', type=float, default=1.0,
                                help='weight for local top-k deformation term')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # 额外增加 reg_global / reg_local，便于单独观察全局和局部形变正则的曲线
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A',
                           'D_B', 'G_B', 'cycle_B', 'idt_B',
                           'reg', 'reg_global', 'reg_local']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'mask_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'mask_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'real_B','fake_A']
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        self.lambda_shape = 0

        # 配准模块配置
        self.lambda_reg = getattr(opt, "lambda_reg", 0.0)
        self.reg_patch_size = getattr(opt, "reg_patch_size", 16)
        self.reg_topk_ratio = getattr(opt, "reg_topk_ratio", 0.25)
        self.reg_alpha = getattr(opt, "reg_alpha", 1.0)
        self.reg_beta = getattr(opt, "reg_beta", 1.0)
        self.loss_reg = torch.tensor(0.0, device=self.device)
        self.loss_reg_global = torch.tensor(0.0, device=self.device)
        self.loss_reg_local = torch.tensor(0.0, device=self.device)
        # 简单的计数器，用于控制形变可视化保存频率
        self._deform_step = 0

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        n_att=0)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'unet', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                        n_att=0)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMask = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if USE_APEX:
                models, optimizers = amp.initialize(
                    [self.netG_A, self.netG_B, self.netD_A, self.netD_B],
                    self.optimizers,
                    opt_level=opt.opt_level,
                    num_losses=2,
                )
                self.netG_A, self.netG_B, self.netD_A, self.netD_B = models

                self.netG_A = torch.nn.DataParallel(self.netG_A, self.gpu_ids)
                self.netG_B = torch.nn.DataParallel(self.netG_B, self.gpu_ids)
                self.netD_A = torch.nn.DataParallel(self.netD_A, self.gpu_ids)
                self.netD_B = torch.nn.DataParallel(self.netD_B, self.gpu_ids)
                self.optimizer_G, self.optimizer_D = optimizers
            else:
                # 不使用 apex 时，直接 DataParallel 包装，optimizers 保持原始的 Adam
                self.netG_A = torch.nn.DataParallel(self.netG_A, self.gpu_ids)
                self.netG_B = torch.nn.DataParallel(self.netG_B, self.gpu_ids)
                self.netD_A = torch.nn.DataParallel(self.netD_A, self.gpu_ids)
                self.netD_B = torch.nn.DataParallel(self.netD_B, self.gpu_ids)

            # 初始化 MultiGradICON 配准包装器（只在训练阶段使用）
            if self.lambda_reg > 0.0 and RegistrationWrapper is not None:
                self.reg_wrapper = RegistrationWrapper(model_name='multigradicon', device=self.device, freeze=True)
            else:
                self.reg_wrapper = None
        else:
            self.reg_wrapper = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.mask_A = input['A_mask'].to(self.device)
        self.mask_B = input['B_mask'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # def forward(self):
    #     """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #     self.fake_B, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
    #     self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
    #     self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG_A(self.real_A)  # G_A(A)
    #     self.rec_A, _, _, _, _, _, _, _, _, _, _, \
    #     _, _, _, _, _, _, _, _, _, _, \
    #     _, _, _, _, _, _, _, _, _ = self.netG_B(self.fake_B)   # G_B(G_A(A))
    #     self.fake_A, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
    #     self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
    #     self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netG_B(self.real_B)  # G_B(B)
    #     self.rec_B, _, _, _, _, _, _, _, _, _, _, \
    #     _, _, _, _, _, _, _, _, _, _, \
    #     _, _, _, _, _, _, _, _, _ = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A  = self.netG_B(self.fake_B)   # G_B(G_A(A))

        self.fake_A  = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_mask = self.opt.lambda_mask
        lambda_shape = self.lambda_shape
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # deformation-field based registration regularization (only MR->CT, i.e. G_A)
        self.loss_reg = torch.tensor(0.0, device=self.device)
        self.loss_reg_global = torch.tensor(0.0, device=self.device)
        self.loss_reg_local = torch.tensor(0.0, device=self.device)
        if self.isTrain and self.lambda_reg > 0.0 and self.reg_wrapper is not None:
            # 保存 fake_B 的原始尺寸（在 resize 之前）
            fake_B_original_shape = self.fake_B.shape  # [B, C, H, W] 或 [B, C, D, H, W]
            
            reg_outputs = self.reg_wrapper.register_pair_with_grad(self.fake_B, self.real_A)
            if reg_outputs is not None:
                phi = reg_outputs.get("phi", None)
                if phi is not None:
                    # 关键修复：将形变场从 MultiGradICON 的 175×175×175 插值回原始尺寸
                    # 这样 loss 计算就在生成器的原始输出尺寸上进行，位置对应更准确
                    if phi.dim() == 5:  # 3D: [B, 3, D_reg, H_reg, W_reg]
                        # 获取原始 fake_B 的空间尺寸（去掉 batch 和 channel 维度）
                        if len(fake_B_original_shape) == 4:  # 原始是 2D: [B, C, H, W]
                            # 需要插值回 [B, 3, 1, H_orig, W_orig]
                            target_spatial = (1, fake_B_original_shape[-2], fake_B_original_shape[-1])
                        else:  # 原始是 3D: [B, C, D, H, W]
                            target_spatial = fake_B_original_shape[-3:]
                        # 使用 trilinear 插值（对 3D 向量场，每个分量独立插值）
                        phi_resized = F.interpolate(
                            phi, size=target_spatial, mode="trilinear", align_corners=False
                        )
                    elif phi.dim() == 4:  # 2D: [B, 2, H_reg, W_reg]
                        # 获取原始 fake_B 的空间尺寸
                        target_spatial = fake_B_original_shape[-2:]  # (H, W)
                        # 使用 bilinear 插值
                        phi_resized = F.interpolate(
                            phi, size=target_spatial, mode="bilinear", align_corners=False
                        )
                    else:
                        phi_resized = phi
                    
                    # 在原始尺寸上计算形变场的平方范数
                    # phi_resized 形状: [B, C(=2/3), ...] 对应原始 fake_B 的空间尺寸
                    disp_sq = (phi_resized ** 2).sum(dim=1, keepdim=True)  # [B,1,...]
                    # 全局形变大小
                    global_term = disp_sq.mean()

                    # 局部 top-k patch（使用 avg_pool 作为 ROI 汇聚）
                    if disp_sq.dim() == 4:  # 2D: [B,1,H,W]
                        patch = max(1, min(self.reg_patch_size, disp_sq.shape[-2], disp_sq.shape[-1]))
                        pooled = F.avg_pool2d(disp_sq, kernel_size=patch, stride=patch)
                    elif disp_sq.dim() == 5:  # 3D: [B,1,D,H,W]
                        patch = max(1, min(self.reg_patch_size, disp_sq.shape[-3], disp_sq.shape[-2], disp_sq.shape[-1]))
                        pooled = F.avg_pool3d(disp_sq, kernel_size=patch, stride=patch)
                    else:
                        pooled = disp_sq

                    B = pooled.shape[0]
                    local_vals = pooled.view(B, -1)
                    k = max(1, int(local_vals.shape[1] * self.reg_topk_ratio))
                    topk_vals, _ = torch.topk(local_vals, k, dim=1)
                    local_term = topk_vals.mean()

                    # 记录未乘 lambda_reg 的原始 global/local，用于日志曲线
                    self.loss_reg_global = global_term.detach()
                    self.loss_reg_local = local_term.detach()

                    self.loss_reg = self.reg_alpha * global_term + self.reg_beta * local_term
                    self.loss_reg = self.loss_reg * self.lambda_reg

                    # 可选：周期性保存一次形变大小热力图（下采样后），用于离线可视化
                    self._deform_step += 1
                    if self._deform_step % 500 == 0:
                        try:
                            # 只取当前 batch 第一张，计算形变模长并下采样到 64x64 方便存储
                            mag = torch.sqrt(disp_sq[0:1])  # [1,1,...]
                            if mag.dim() == 4:
                                # 2D: [1,1,H,W] -> [1,1,64,64]
                                mag_ds = F.interpolate(mag, size=(64, 64), mode="area")
                                mag_np = mag_ds.squeeze().detach().cpu().numpy()
                                debug_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, "deform_debug")
                                os.makedirs(debug_dir, exist_ok=True)
                                fname = f"step_{self._deform_step:07d}_mag.npy"
                                np.save(os.path.join(debug_dir, fname), mag_np)
                        except Exception:
                            # 任何可视化相关错误都不影响训练
                            pass

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
            + self.loss_idt_A + self.loss_idt_B + self.loss_reg
        #self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        if USE_APEX:
            with amp.scale_loss(self.loss_G, self.optimizer_G, loss_id=0) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer_G), 1.0)
        else:
            self.loss_G.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        loss_D = self.loss_D_B+self.loss_D_A
        if USE_APEX:
            with amp.scale_loss(loss_D, self.optimizer_D, loss_id=1) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer_G), 1.0)
        else:
            loss_D.backward()
        self.optimizer_D.step()
