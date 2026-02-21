# 方案A实现总结

## 我具体做了什么

### 1. 创建了配准判别器模块 (`MASKGAN/models/registration_discriminator/`)

#### 1.1 `deformation_complexity.py` - 形变场复杂度计算
**功能：** 从配准模型的形变场中提取复杂度特征

**包含的函数：**
- `compute_deformation_magnitude(phi)`: 计算形变幅度
- `compute_deformation_gradient(phi)`: 计算形变梯度（平滑度）
- `compute_deformation_curvature(phi)`: 计算形变曲率
- `compute_jacobian_determinant(phi)`: 计算雅可比行列式
- `compute_jacobian_anomaly(jacobian)`: 计算雅可比异常区域
- `compute_global_complexity(phi)`: **计算全局复杂度（一个标量）**
- `compute_local_complexity(phi, num_blocks=8)`: **计算局部复杂度（8×8=64个块的值）**

**核心设计：**
- 全局复杂度 = α×形变幅度 + β×形变梯度 + γ×形变曲率 + δ×雅可比异常
- 局部复杂度：将图像切分成N×N块，每块计算复杂度值

#### 1.2 `registration_discriminator.py` - 判别器网络
**功能：** 使用形变场复杂度作为输入的判别器网络

**网络结构：**
- **全局分支**：MLP处理全局复杂度标量 → [B, 1]
- **局部分支**：轻量级CNN处理局部复杂度图 → [B, num_blocks^2]
- **融合层**：将全局和局部特征融合 → [B, 1]

**输出：** 判别结果（用于LSGAN或vanilla GAN）

#### 1.3 `registration_model_wrapper.py` - 配准模型包装器
**功能：** 包装multigradICON配准模型，提供统一接口

**包含：**
- `RegistrationModelWrapper`: 包装实际的multigradICON模型
- `DummyRegistrationModel`: 虚拟配准模型（用于测试，返回恒等形变场）

**关键特性：**
- 自动冻结配准模型权重（不参与训练）
- 自动设置eval模式
- 提供统一的forward接口，返回包含`phi_AB_vectorfield`的结果对象

---

### 2. 修改了MASKGAN模型 (`MASKGAN/models/mask_gan_model.py`)

#### 2.1 添加了命令行参数（在`modify_commandline_options`中）
```python
--use_registration_discriminator  # 启用配准判别器
--lambda_reg_global 0.05          # 全局复杂度损失权重
--lambda_reg_local 0.5            # 局部复杂度损失权重
--complexity_alpha 1.0            # 形变幅度权重
--complexity_beta 0.5             # 形变梯度权重
--complexity_gamma 0.1             # 形变曲率权重
--complexity_delta 0.5             # 雅可比异常权重
--num_blocks 8                     # 块数量（8×8=64块）
```

#### 2.2 在`__init__`中添加了配准模型和判别器初始化
**添加的内容：**
- `self.net_reg`: 配准模型包装器（RegistrationModelWrapper）
- `self.netD_reg_A`: A→B方向的配准判别器
- `self.netD_reg_B`: B→A方向的配准判别器
- `self.optimizer_D_reg`: 配准判别器的优化器
- 复杂度计算参数（alpha, beta, gamma, delta, num_blocks）
- 损失权重（lambda_reg_global, lambda_reg_local）

**添加到loss_names：**
- `D_reg_A`, `D_reg_B`, `G_reg_A`, `G_reg_B`

**添加到model_names：**
- `D_reg_A`, `D_reg_B`（用于保存/加载）

#### 2.3 添加了`backward_D_reg_A()`方法（方案A核心）
**功能：** 计算配准判别器D_reg_A的损失

**每轮迭代都执行：**
1. **真实对计算**：
   - 输入：(real_B, real_B)
   - 配准 → 得到形变场 phi_real
   - 计算：全局复杂度_real, 局部复杂度图_real
   - 目标：复杂度应该接近0

2. **伪对计算**：
   - 输入：(fake_B.detach(), real_B) - 从fake_B_pool采样
   - 配准 → 得到形变场 phi_fake
   - 计算：全局复杂度_fake, 局部复杂度图_fake
   - 目标：复杂度应该较大

3. **判别器损失**：
   - pred_real = discriminator(全局复杂度_real, 局部复杂度图_real)
   - pred_fake = discriminator(全局复杂度_fake, 局部复杂度图_fake)
   - loss_D_real = criterion(pred_real, True)
   - loss_D_fake = criterion(pred_fake, False)
   - loss_D_reg_A = (loss_D_real + loss_D_fake) * 0.5

#### 2.4 添加了`backward_D_reg_B()`方法
**功能：** 与`backward_D_reg_A()`相同，但用于B→A方向
- 真实对：(real_A, real_A)
- 伪对：(fake_A, real_A)

#### 2.5 修改了`backward_G()`方法
**添加了配准损失：**

对于fake_B（A→B方向）：
1. 配准fake_B和real_B → 得到形变场phi_B
2. 计算全局复杂度和局部复杂度
3. 生成器损失：
   - `loss_G_reg_A = lambda_reg_global × 全局复杂度 + lambda_reg_local × mean(局部复杂度)`
   - 加上判别器对抗损失：`criterionGAN(discriminator(复杂度), True)`

对于fake_A（B→A方向）：
- 同样的计算，得到`loss_G_reg_B`

**总生成器损失：**
```python
loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + 
         loss_idt_A + loss_idt_B + loss_mask + loss_shape + loss_cor +
         loss_G_reg_A + loss_G_reg_B  # 新增的配准损失
```

#### 2.6 修改了`optimize_parameters()`方法
**添加了配准判别器训练步骤：**

训练流程：
1. **生成器训练**：冻结所有判别器（包括配准判别器）
2. **标准判别器训练**：训练D_A和D_B
3. **配准判别器训练**（新增）：
   - 解冻D_reg_A和D_reg_B
   - 调用`backward_D_reg_A()`和`backward_D_reg_B()`
   - 计算总损失：`loss_D_reg = loss_D_reg_A + loss_D_reg_B`
   - 反向传播并更新

---

### 3. 方案A的核心特点

#### 3.1 每轮都计算正负对
- **真实对**：每轮都计算(real_B, real_B)的配准
- **伪对**：每轮都计算(fake_B, real_B)的配准
- **优点**：训练稳定，判别器能及时学习当前生成器状态
- **缺点**：计算开销大（每轮两次配准）

#### 3.2 全局+局部反馈
- **全局反馈**：一个标量复杂度值，反映整体配准难度
- **局部反馈**：8×8=64个块的复杂度图，提供空间分辨率的反馈

#### 3.3 形变场复杂度分析
从形变场中提取4个特征：
1. **形变幅度**：需要多大形变才能配准
2. **形变梯度**：形变场的平滑度
3. **形变曲率**：形变变化率
4. **雅可比异常**：折叠/过度拉伸的区域比例

---

## 文件修改清单

### 新增文件：
1. `MASKGAN/models/registration_discriminator/__init__.py`
2. `MASKGAN/models/registration_discriminator/deformation_complexity.py`
3. `MASKGAN/models/registration_discriminator/registration_discriminator.py`
4. `MASKGAN/models/registration_discriminator/registration_model_wrapper.py`

### 修改文件：
1. `MASKGAN/models/mask_gan_model.py`
   - 添加了配准判别器相关导入
   - 添加了命令行参数
   - 添加了配准模型和判别器初始化
   - 添加了`backward_D_reg_A()`和`backward_D_reg_B()`方法
   - 修改了`backward_G()`添加配准损失
   - 修改了`optimize_parameters()`添加配准判别器训练

---

## 使用方法

### 1. 训练时启用配准判别器
```bash
python train.py --use_registration_discriminator \
                --lambda_reg_global 0.05 \
                --lambda_reg_local 0.5 \
                --num_blocks 8 \
                ...其他参数
```

### 2. 加载multigradICON模型
需要在`registration_model_wrapper.py`中修改`RegistrationModelWrapper.__init__`，加载你的multigradICON模型：

```python
# 在RegistrationModelWrapper.__init__中
from unigradicon import get_model_from_model_zoo
multigradicon = get_model_from_model_zoo("multigradicon")
# 注意：需要确保multigradICON支持2D图像
```

### 3. 调整参数
根据训练效果调整：
- `lambda_reg_global`: 0.01-0.1（全局损失权重）
- `lambda_reg_local`: 0.1-1.0（局部损失权重）
- `complexity_alpha/beta/gamma/delta`: 复杂度计算权重
- `num_blocks`: 4-16（块数量，影响计算量和精度）

---

## 关键技术细节

### 1. 配准模型接口
- 配准模型必须返回包含`phi_AB_vectorfield`属性的对象
- `phi_AB_vectorfield`形状：[B, 2, H, W]（2D）或[B, 3, H, W, D]（3D）
- 配准模型权重被冻结，不参与训练

### 2. 形变场坐标系统
- 形变场phi表示每个像素的位移向量
- phi[:, 0, ...]是x方向位移
- phi[:, 1, ...]是y方向位移
- 恒等映射（无形变）时，phi应该接近坐标网格

### 3. 复杂度计算
- 全局复杂度：对空间维度求平均，得到标量
- 局部复杂度：对每个块求平均，得到N×N的复杂度图
- 权重组合：alpha×magnitude + beta×gradient + gamma×curvature + delta×anomaly

### 4. 训练流程（方案A）
每轮迭代：
1. Forward：生成fake_B和fake_A
2. Generator backward：计算所有损失（包括配准损失）
3. Standard discriminator backward：训练D_A和D_B
4. Registration discriminator backward：训练D_reg_A和D_reg_B（每轮都计算正负对）

---

## 注意事项

1. **multigradICON需要2D版本**：MASKGAN处理2D图像，需要确保multigradICON支持2D
2. **计算开销**：每轮需要计算两次配准，可能较慢
3. **内存占用**：需要存储两个配准结果
4. **参数调整**：需要仔细调整lambda_reg_global和lambda_reg_local

---

## 下一步工作

1. **加载实际的multigradICON模型**：修改`registration_model_wrapper.py`
2. **确保multigradICON支持2D**：可能需要修改multigradICON代码
3. **测试训练**：先用小数据集测试，确保代码运行正常
4. **参数调优**：根据训练效果调整各种权重参数

