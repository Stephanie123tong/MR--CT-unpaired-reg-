import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from apex import amp
from .registration_discriminator import (
    compute_global_complexity,
    compute_local_complexity,
    RegistrationDiscriminator,
    RegistrationModelWrapper
)


class MaskGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--argmax',  action='store_true', default=False, help='Only select max value in the attention')
        parser.add_argument('--n_attentions', type=int, default=5, help='weight for cycle loss (A -> B -> A)')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_co_A', type=float, default=0.0, help='weight for correlation coefficient loss (A -> B)')
            parser.add_argument('--lambda_co_B', type=float, default=0.0,
                                help='weight for correlation coefficient loss (B -> A )')
            
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_mask', type=float, default=0.5, help='attention mask')
            parser.add_argument('--lambda_shape', type=float, default=0, help='attention mask')
            
            # Registration discriminator options (Solution A)
            parser.add_argument('--use_registration_discriminator', action='store_true', 
                              help='enable registration discriminator')
            parser.add_argument('--lambda_reg_global', type=float, default=0.05,
                              help='weight for global registration complexity loss')
            parser.add_argument('--lambda_reg_local', type=float, default=0.5,
                              help='weight for local registration complexity loss')
            parser.add_argument('--complexity_alpha', type=float, default=1.0,
                              help='weight for deformation magnitude in complexity')
            parser.add_argument('--complexity_beta', type=float, default=0.5,
                              help='weight for deformation gradient in complexity')
            parser.add_argument('--complexity_gamma', type=float, default=0.1,
                              help='weight for deformation curvature in complexity')
            parser.add_argument('--complexity_delta', type=float, default=0.5,
                              help='weight for jacobian anomaly in complexity')
            parser.add_argument('--num_blocks', type=int, default=8,
                              help='number of blocks per dimension for local complexity')
            parser.add_argument('--use_3d_expansion', action='store_true', default=True,
                              help='expand 2D images to 3D to use pretrained multigradICON weights (Solution 2)')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'maskA', 'maskB', 'shapeA', 'shapeB']
        if self.isTrain and hasattr(opt, 'use_registration_discriminator') and opt.use_registration_discriminator:
            self.loss_names.extend(['D_reg_A', 'D_reg_B', 'G_reg_A', 'G_reg_B'])
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'o1_b', 'o2_b', 'o3_b', 'o_bg_b',
            'a1_b', 'a2_b', 'a3_b', 'a_bg_b',  'i1_b', 'i2_b', 'i3_b', 'i_last_b', 'mask_A', 
             'rec_a1_b', 'rec_a2_b', 'rec_a3_b', 'rec_a_bg_b', 'rec_i1_b', 'rec_i2_b', 'rec_i3_b', 'rec_i_last_b', 
             'rec_o1_b', 'rec_o2_b', 'rec_o3_b', 'rec_o_bg_b']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'o1_a', 'o2_a', 'o_bg_a',
            'a1_a', 'a2_a', 'a_bg_a', 'i1_a', 'i2_a', 'i_last_a', 'mask_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'a10_b', 'real_B','fake_A', 'a10_a']
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            if hasattr(opt, 'use_registration_discriminator') and opt.use_registration_discriminator:
                self.model_names.extend(['D_reg_A', 'D_reg_B'])
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        self.lambda_shape = 0

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'att', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        n_att=opt.n_attentions, argmax=opt.argmax)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'att', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                        n_att=opt.n_attentions, argmax=opt.argmax)

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

            models, optimizers = amp.initialize([self.netG_A, self.netG_B, self.netD_A, self.netD_B], 
                                    self.optimizers,
                                        opt_level=opt.opt_level, num_losses=2)
            self.netG_A, self.netG_B, self.netD_A, self.netD_B = models

            self.netG_A = torch.nn.DataParallel(self.netG_A, self.gpu_ids)
            self.netG_B = torch.nn.DataParallel(self.netG_B, self.gpu_ids)
            self.netD_A = torch.nn.DataParallel(self.netD_A, self.gpu_ids)
            self.netD_B = torch.nn.DataParallel(self.netD_B, self.gpu_ids)
            self.optimizer_G, self.optimizer_D = optimizers
            
            # Initialize registration model and discriminator (Solution A)
            if hasattr(opt, 'use_registration_discriminator') and opt.use_registration_discriminator:
                # Initialize registration model wrapper
                # Solution 2: Use 3D expansion (use_3d_expansion=True) to use pretrained weights
                input_shape = [1, opt.input_nc, opt.load_size, opt.load_size]
                use_3d_expansion = getattr(opt, 'use_3d_expansion', True)  # Default: True (use 3D expansion)
                self.net_reg = RegistrationModelWrapper(
                    None, 
                    input_shape=input_shape,
                    use_3d_expansion=use_3d_expansion,
                    device=self.device
                ).to(self.device)
                
                # Initialize registration discriminators
                num_blocks = getattr(opt, 'num_blocks', 8)
                self.netD_reg_A = RegistrationDiscriminator(
                    num_blocks=num_blocks,
                    gan_mode=opt.gan_mode,
                    use_local=True
                ).to(self.device)
                
                self.netD_reg_B = RegistrationDiscriminator(
                    num_blocks=num_blocks,
                    gan_mode=opt.gan_mode,
                    use_local=True
                ).to(self.device)
                
                # Registration loss weights
                self.lambda_reg_global = getattr(opt, 'lambda_reg_global', 0.05)
                self.lambda_reg_local = getattr(opt, 'lambda_reg_local', 0.5)
                
                # Complexity computation parameters
                self.complexity_alpha = getattr(opt, 'complexity_alpha', 1.0)
                self.complexity_beta = getattr(opt, 'complexity_beta', 0.5)
                self.complexity_gamma = getattr(opt, 'complexity_gamma', 0.1)
                self.complexity_delta = getattr(opt, 'complexity_delta', 0.5)
                self.num_blocks = num_blocks
                
                # Create optimizer for registration discriminators
                self.optimizer_D_reg = torch.optim.Adam(
                    itertools.chain(self.netD_reg_A.parameters(), self.netD_reg_B.parameters()),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_D_reg)
                
                # Wrap registration discriminators with DataParallel
                self.netD_reg_A = torch.nn.DataParallel(self.netD_reg_A, self.gpu_ids)
                self.netD_reg_B = torch.nn.DataParallel(self.netD_reg_B, self.gpu_ids)
            else:
                self.net_reg = None
                self.netD_reg_A = None
                self.netD_reg_B = None
                self.lambda_reg_global = 0.0
                self.lambda_reg_local = 0.0


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
        self.fake_B, self.outputs_B, self.attentions_B, self.images_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A, output_rec_A, att_rec_A, image_rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.o1_b, self.o2_b, self.o3_b, self.o_bg_b = self.outputs_B[0], self.outputs_B[1], self.outputs_B[-2], self.outputs_B[-1]
        self.a1_b, self.a2_b, self.a3_b, self.a_bg_b = self.attentions_B[0], self.attentions_B[1], self.attentions_B[-2], self.attentions_B[-1]
        self.i1_b, self.i2_b, self.i3_b, self.i_last_b = self.images_B[0], self.images_B[1], self.images_B[-2], self.images_B[-1]
        #self.a_fg_b = 1 - self.a_bg_b
        self.att_rec_bg_a = att_rec_A[-1]
        self.rec_a1_b, self.rec_a2_b, self.rec_a3_b, self.rec_a_bg_b =  att_rec_A[0], att_rec_A[1], att_rec_A[-2], att_rec_A[-1] 
        self.rec_i1_b, self.rec_i2_b, self.rec_i3_b, self.rec_i_last_b = image_rec_A[0], image_rec_A[1], image_rec_A[-2], image_rec_A[-1]
        self.rec_o1_b, self.rec_o2_b, self.rec_o3_b, self.rec_o_bg_b = output_rec_A[0], output_rec_A[1], output_rec_A[-2], output_rec_A[-1]


        self.fake_A, self.outputs_A, self.attentions_A, self.images_A  = self.netG_B(self.real_B)  # G_B(B)
        # self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a
        # self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a
        # self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a
        self.rec_B, att_rec_B, _, _ = self.netG_A(self.fake_A)   # G_A(G_B(B))
        self.o1_a, self.o2_a, self.o_bg_a= self.outputs_A[0], self.outputs_A[-2], self.outputs_A[-1]
        self.a1_a, self.a2_a, self.a_bg_a= self.attentions_A[0], self.attentions_A[-2], self.attentions_A[-1]
        self.i1_a, self.i2_a, self.i_last_a= self.images_A[0], self.images_A[-2], self.images_A[-1]
        #self.a_fg_a = 1 - self.a_bg_a
        self.att_rec_bg_b = att_rec_B[-1]


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
    
    def backward_D_reg_A(self):
        """
        Calculate GAN loss for registration discriminator D_reg_A (Solution A).
        Every iteration computes both positive and negative pairs.
        """
        if self.net_reg is None:
            self.loss_D_reg_A = torch.tensor(0.0, device=self.device)
            return
        
        # ========== Positive Pair: (real_B, real_B) ==========
        reg_result_real = self.net_reg(self.real_B, self.real_B)
        phi_real = reg_result_real.phi_AB_vectorfield
        
        # Compute complexity
        global_complexity_real = compute_global_complexity(
            phi_real,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_gamma,
            delta=self.complexity_delta
        )
        
        local_complexity_real = compute_local_complexity(
            phi_real,
            num_blocks=self.num_blocks,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_delta
        )
        
        # ========== Negative Pair: (fake_B, real_B) ==========
        fake_B_sample = self.fake_B_pool.query(self.fake_B)
        reg_result_fake = self.net_reg(fake_B_sample.detach(), self.real_B)
        phi_fake = reg_result_fake.phi_AB_vectorfield
        
        global_complexity_fake = compute_global_complexity(
            phi_fake,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_gamma,
            delta=self.complexity_delta
        )
        
        local_complexity_fake = compute_local_complexity(
            phi_fake,
            num_blocks=self.num_blocks,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_delta
        )
        
        # ========== Discriminator Forward ==========
        pred_real = self.netD_reg_A(global_complexity_real, local_complexity_real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        pred_fake = self.netD_reg_A(global_complexity_fake, local_complexity_fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        self.loss_D_reg_A = (loss_D_real + loss_D_fake) * 0.5
    
    def backward_D_reg_B(self):
        """
        Calculate GAN loss for registration discriminator D_reg_B (Solution A).
        Same as backward_D_reg_A but for direction B -> A.
        """
        if self.net_reg is None:
            self.loss_D_reg_B = torch.tensor(0.0, device=self.device)
            return
        
        # Positive pair: (real_A, real_A)
        reg_result_real = self.net_reg(self.real_A, self.real_A)
        phi_real = reg_result_real.phi_AB_vectorfield
        
        global_complexity_real = compute_global_complexity(
            phi_real,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_gamma,
            delta=self.complexity_delta
        )
        
        local_complexity_real = compute_local_complexity(
            phi_real,
            num_blocks=self.num_blocks,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_delta
        )
        
        # Negative pair: (fake_A, real_A)
        fake_A_sample = self.fake_A_pool.query(self.fake_A)
        reg_result_fake = self.net_reg(fake_A_sample.detach(), self.real_A)
        phi_fake = reg_result_fake.phi_AB_vectorfield
        
        global_complexity_fake = compute_global_complexity(
            phi_fake,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_gamma,
            delta=self.complexity_delta
        )
        
        local_complexity_fake = compute_local_complexity(
            phi_fake,
            num_blocks=self.num_blocks,
            alpha=self.complexity_alpha,
            beta=self.complexity_beta,
            gamma=self.complexity_delta
        )
        
        pred_real = self.netD_reg_B(global_complexity_real, local_complexity_real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        pred_fake = self.netD_reg_B(global_complexity_fake, local_complexity_fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        self.loss_D_reg_B = (loss_D_real + loss_D_fake) * 0.5

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
            self.idt_A, _, _, _  = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _, _  = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        if lambda_mask > 0:
            self.loss_maskB = self.criterionMask(self.a_bg_b, self.mask_A)
            self.loss_maskA = self.criterionMask(self.a_bg_a, self.mask_B)
            loss_mask = lambda_mask*(self.loss_maskA + self.loss_maskB)
        else:
            self.loss_maskA = self.loss_maskB = 0
            loss_mask = 0
        
        self.loss_shapeB = lambda_shape *self.criterionMask(self.a_bg_b, self.att_rec_bg_a)
        self.loss_shapeA = lambda_shape * self.criterionMask(self.a_bg_a, self.att_rec_bg_a)
        loss_shape = self.loss_shapeA + self.loss_shapeB

        loss_cor_coe_GA = networks.Cor_CoeLoss(self.fake_B, self.real_A) * self.opt.lambda_co_A# fake ct & real mr; Evaluate the Generator of ct(G_A)
        loss_cor_coe_GB = networks.Cor_CoeLoss(self.fake_A, self.real_B) * self.opt.lambda_co_B # fake mr & real ct; Evaluate the Generator of mr(G_B)
        loss_cor = loss_cor_coe_GA  + loss_cor_coe_GB
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Registration losses (Solution A)
        if hasattr(self, 'net_reg') and self.net_reg is not None and \
           hasattr(self.opt, 'use_registration_discriminator') and self.opt.use_registration_discriminator:
            # For fake_B (A -> B direction)
            reg_result_B = self.net_reg(self.fake_B, self.real_B)
            phi_B = reg_result_B.phi_AB_vectorfield
            
            global_complexity_B = compute_global_complexity(
                phi_B,
                alpha=self.complexity_alpha,
                beta=self.complexity_beta,
                gamma=self.complexity_gamma,
                delta=self.complexity_delta
            )
            
            local_complexity_B = compute_local_complexity(
                phi_B,
                num_blocks=self.num_blocks,
                alpha=self.complexity_alpha,
                beta=self.complexity_beta,
                gamma=self.complexity_delta
            )
            
            # Generator wants to minimize complexity
            self.loss_G_reg_A = (
                self.lambda_reg_global * global_complexity_B.mean() +
                self.lambda_reg_local * local_complexity_B.mean()
            )
            
            # Also use discriminator loss (adversarial)
            pred_reg = self.netD_reg_A(global_complexity_B, local_complexity_B)
            self.loss_G_reg_A += self.criterionGAN(pred_reg, True)
            
            # For fake_A (B -> A direction)
            reg_result_A = self.net_reg(self.fake_A, self.real_A)
            phi_A = reg_result_A.phi_AB_vectorfield
            
            global_complexity_A = compute_global_complexity(
                phi_A,
                alpha=self.complexity_alpha,
                beta=self.complexity_beta,
                gamma=self.complexity_gamma,
                delta=self.complexity_delta
            )
            
            local_complexity_A = compute_local_complexity(
                phi_A,
                num_blocks=self.num_blocks,
                alpha=self.complexity_alpha,
                beta=self.complexity_beta,
                gamma=self.complexity_delta
            )
            
            self.loss_G_reg_B = (
                self.lambda_reg_global * global_complexity_A.mean() +
                self.lambda_reg_local * local_complexity_A.mean()
            )
            
            pred_reg = self.netD_reg_B(global_complexity_A, local_complexity_A)
            self.loss_G_reg_B += self.criterionGAN(pred_reg, True)
        else:
            self.loss_G_reg_A = torch.tensor(0.0, device=self.device)
            self.loss_G_reg_B = torch.tensor(0.0, device=self.device)
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
            + self.loss_idt_A + self.loss_idt_B + loss_mask + loss_shape + loss_cor \
            + self.loss_G_reg_A + self.loss_G_reg_B
        #self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        
        # G_A and G_B
        disc_list = [self.netD_A, self.netD_B]
        if hasattr(self, 'netD_reg_A') and self.netD_reg_A is not None:
            disc_list.extend([self.netD_reg_A, self.netD_reg_B])
        self.set_requires_grad(disc_list, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        with amp.scale_loss(self.loss_G, self.optimizer_G, loss_id=0) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer_G), 1.0)
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        loss_D = self.loss_D_B+self.loss_D_A
        with amp.scale_loss(loss_D, self.optimizer_D, loss_id=1) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer_G), 1.0)
        self.optimizer_D.step()
        
        # Registration discriminators D_reg_A and D_reg_B (Solution A)
        if hasattr(self, 'netD_reg_A') and self.netD_reg_A is not None and \
           hasattr(self, 'optimizer_D_reg') and self.optimizer_D_reg is not None:
            self.set_requires_grad([self.netD_reg_A, self.netD_reg_B], True)
            self.optimizer_D_reg.zero_grad()
            self.backward_D_reg_A()
            self.backward_D_reg_B()
            loss_D_reg = self.loss_D_reg_A + self.loss_D_reg_B
            loss_D_reg.backward()
            self.optimizer_D_reg.step()
