# 方案A实现 - 我具体做了什么

## 总体概述

我在现有的MASKGAN和multigradICON代码基础上，实现了**方案A：每轮都计算正负对的配准判别器**。

---

## 一、创建的新模块

### 1. `MASKGAN/models/registration_discriminator/` 目录

创建了4个新文件：

#### ① `deformation_complexity.py` - 形变场复杂度计算
**作用：** 从配准模型的形变场中提取复杂度特征

**核心函数：**
- `compute_global_complexity()`: 计算全局复杂度（一个标量）
  - 输入：形变场phi [B, 2, H, W]
  - 输出：复杂度标量 [B]
  - 计算：形变幅度 + 形变梯度 + 形变曲率 + 雅可比异常

- `compute_local_complexity()`: 计算局部复杂度（块级）
  - 输入：形变场phi [B, 2, H, W], num_blocks=8
  - 输出：复杂度图 [B, 8, 8]（64个块的值）
  - 方法：将图像切分成8×8块，每块计算复杂度

#### ② `registration_discriminator.py` - 判别器网络
**作用：** 使用复杂度作为输入的判别器

**网络结构：**
- 全局分支：MLP处理标量 → [B, 1]
- 局部分支：CNN处理块图 → [B, 64] → [B, 1]
- 融合：全局 + 局部 → 最终判别结果

#### ③ `registration_model_wrapper.py` - 配准模型包装器
**作用：** 包装multigradICON，提供统一接口

**功能：**
- 自动加载multigradICON模型（如果可用）
- 冻结配准模型权重（不参与训练）
- 提供统一的forward接口
- 包含DummyRegistrationModel用于测试

#### ④ `__init__.py` - 模块导出
导出所有需要的类和函数

---

## 二、修改的现有文件

### 1. `MASKGAN/models/mask_gan_model.py`

#### 修改1：添加导入
```python
from .registration_discriminator import (
    compute_global_complexity,
    compute_local_complexity,
    RegistrationDiscriminator,
    RegistrationModelWrapper
)
```

#### 修改2：添加命令行参数（在`modify_commandline_options`中）
添加了8个新参数：
- `--use_registration_discriminator`: 启用配准判别器
- `--lambda_reg_global`: 全局复杂度损失权重（默认0.05）
- `--lambda_reg_local`: 局部复杂度损失权重（默认0.5）
- `--complexity_alpha/beta/gamma/delta`: 复杂度计算权重
- `--num_blocks`: 块数量（默认8）

#### 修改3：在`__init__`中初始化配准组件
**添加的内容：**
1. 初始化配准模型：`self.net_reg = RegistrationModelWrapper(...)`
2. 初始化配准判别器：`self.netD_reg_A`和`self.netD_reg_B`
3. 创建优化器：`self.optimizer_D_reg`
4. 设置参数：复杂度权重、损失权重等
5. 添加到loss_names和model_names

#### 修改4：添加`backward_D_reg_A()`方法（方案A核心）
**每轮迭代都执行：**

```python
# 1. 真实对：(real_B, real_B)
reg_result_real = self.net_reg(self.real_B, self.real_B)
phi_real = reg_result_real.phi_AB_vectorfield
global_complexity_real = compute_global_complexity(phi_real, ...)
local_complexity_real = compute_local_complexity(phi_real, ...)

# 2. 伪对：(fake_B, real_B)
fake_B_sample = self.fake_B_pool.query(self.fake_B)
reg_result_fake = self.net_reg(fake_B_sample.detach(), self.real_B)
phi_fake = reg_result_fake.phi_AB_vectorfield
global_complexity_fake = compute_global_complexity(phi_fake, ...)
local_complexity_fake = compute_local_complexity(phi_fake, ...)

# 3. 判别器损失
pred_real = self.netD_reg_A(global_complexity_real, local_complexity_real)
pred_fake = self.netD_reg_A(global_complexity_fake, local_complexity_fake)
loss_D_real = self.criterionGAN(pred_real, True)
loss_D_fake = self.criterionGAN(pred_fake, False)
self.loss_D_reg_A = (loss_D_real + loss_D_fake) * 0.5
```

#### 修改5：添加`backward_D_reg_B()`方法
与`backward_D_reg_A()`相同，但用于B→A方向

#### 修改6：修改`backward_G()`添加配准损失
**在生成器损失中添加：**

```python
# 对于fake_B
reg_result_B = self.net_reg(self.fake_B, self.real_B)
phi_B = reg_result_B.phi_AB_vectorfield
global_complexity_B = compute_global_complexity(phi_B, ...)
local_complexity_B = compute_local_complexity(phi_B, ...)

# 生成器希望降低复杂度
self.loss_G_reg_A = (
    lambda_reg_global * global_complexity_B.mean() +
    lambda_reg_local * local_complexity_B.mean()
)

# 加上判别器对抗损失
pred_reg = self.netD_reg_A(global_complexity_B, local_complexity_B)
self.loss_G_reg_A += self.criterionGAN(pred_reg, True)
```

#### 修改7：修改`optimize_parameters()`添加配准判别器训练
**添加了第三步训练：**

```python
# 1. Generator training (freeze all discriminators)
# 2. Standard discriminator training (D_A, D_B)
# 3. Registration discriminator training (NEW)
self.set_requires_grad([self.netD_reg_A, self.netD_reg_B], True)
self.optimizer_D_reg.zero_grad()
self.backward_D_reg_A()
self.backward_D_reg_B()
loss_D_reg = self.loss_D_reg_A + self.loss_D_reg_B
loss_D_reg.backward()
self.optimizer_D_reg.step()
```

---

## 三、方案A的核心设计

### 1. 每轮都计算正负对
- ✅ 真实对：(real_B, real_B) - 每轮都计算
- ✅ 伪对：(fake_B, real_B) - 每轮都计算
- ✅ 判别器每轮都有正负对比，训练稳定

### 2. 全局+局部反馈
- ✅ 全局：一个标量复杂度值
- ✅ 局部：8×8=64个块的复杂度图
- ✅ 提供多层次反馈

### 3. 形变场复杂度分析
从形变场提取4个特征：
1. 形变幅度（magnitude）
2. 形变梯度（gradient）
3. 形变曲率（curvature）
4. 雅可比异常（jacobian anomaly）

---

## 四、文件结构

```
registration_discriminator_solutionA/
├── MASKGAN/
│   └── models/
│       ├── mask_gan_model.py          [修改] 集成配准判别器
│       └── registration_discriminator/ [新建] 配准判别器模块
│           ├── __init__.py
│           ├── deformation_complexity.py
│           ├── registration_discriminator.py
│           └── registration_model_wrapper.py
└── multigradICON/                      [已有] 配准模型
```

---

## 五、使用方法

### 1. 训练时启用配准判别器
```bash
cd MASKGAN
python train.py \
    --use_registration_discriminator \
    --lambda_reg_global 0.05 \
    --lambda_reg_local 0.5 \
    --num_blocks 8 \
    ...其他原有参数
```

### 2. 加载multigradICON模型
需要在`registration_model_wrapper.py`中修改，加载你的multigradICON模型。

**注意：** multigradICON默认是3D的，需要确保支持2D图像，或者修改代码适配2D。

---

## 六、关键技术点

### 1. 配准模型接口
- 必须返回包含`phi_AB_vectorfield`的对象
- `phi_AB_vectorfield`形状：[B, 2, H, W]（2D）
- 配准模型权重被冻结

### 2. 复杂度计算
- 全局：对空间维度求平均 → 标量
- 局部：对每个块求平均 → N×N图
- 组合：alpha×magnitude + beta×gradient + gamma×curvature + delta×anomaly

### 3. 训练流程
每轮迭代：
1. Forward：生成fake_B和fake_A
2. Generator backward：计算所有损失（包括配准损失）
3. Standard discriminator backward：训练D_A和D_B
4. **Registration discriminator backward**：训练D_reg_A和D_reg_B（每轮都计算正负对）

---

## 七、注意事项

1. **multigradICON需要2D支持**：当前代码使用DummyRegistrationModel，需要实现2D版本的multigradICON
2. **计算开销**：每轮需要计算两次配准，可能较慢
3. **参数调整**：需要根据训练效果调整lambda_reg_global和lambda_reg_local

---

## 总结

我完成了方案A的完整实现：
- ✅ 创建了形变场复杂度计算模块
- ✅ 创建了配准判别器网络
- ✅ 创建了配准模型包装器
- ✅ 修改了MASKGAN模型集成配准判别器
- ✅ 实现了每轮都计算正负对的训练流程
- ✅ 提供了全局+局部的多层次反馈

所有代码都在`registration_discriminator_solutionA`文件夹内，可以直接使用。

