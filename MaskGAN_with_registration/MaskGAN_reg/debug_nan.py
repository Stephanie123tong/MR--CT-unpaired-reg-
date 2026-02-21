"""
NaN 排查脚本
用于诊断训练中出现 NaN 的原因
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data import create_dataset
from options.train_options import TrainOptions
from models import create_model

def check_data():
    """检查数据是否正常"""
    print("=" * 60)
    print("步骤 1: 检查数据")
    print("=" * 60)
    
    # 保存原始 argv
    original_argv = sys.argv.copy()
    
    # 设置模拟的命令行参数
    sys.argv = [
        'debug_nan.py',
        '--dataroot', '/root/cermep_processed',
        '--model', 'mask_gan',
        '--batch_size', '1',
        '--max_dataset_size', '5',
        '--serial_batches',
        '--wdb_disabled',
        '--no_html',
        '--gpu_ids', '0',
        '--name', 'debug_test',
        '--checkpoints_dir', './checkpoints',
        '--lambda_reg', '1.0',
    ]
    
    try:
        opt = TrainOptions().parse()
    finally:
        # 恢复原始 argv
        sys.argv = original_argv
    
    dataset = create_dataset(opt)
    print(f"✓ Dataset size: {len(dataset)}")
    
    for i, data in enumerate(dataset):
        print(f"\n--- Sample {i} ---")
        A = data['A']
        B = data['B']
        
        print(f"real_A: shape={A.shape}, range=[{A.min():.4f}, {A.max():.4f}], mean={A.mean():.4f}")
        print(f"  has_nan: {torch.isnan(A).any()}, has_inf: {torch.isinf(A).any()}")
        
        print(f"real_B: shape={B.shape}, range=[{B.min():.4f}, {B.max():.4f}], mean={B.mean():.4f}")
        print(f"  has_nan: {torch.isnan(B).any()}, has_inf: {torch.isinf(B).any()}")
        
        if 'A_mask' in data:
            mask_A = data['A_mask']
            print(f"mask_A: shape={mask_A.shape}, range=[{mask_A.min():.4f}, {mask_A.max():.4f}]")
            print(f"  has_nan: {torch.isnan(mask_A).any()}")
        
        if i >= 2:
            break
    
    print("\n✓ 数据检查完成\n")
    return opt, dataset

def check_model_forward(opt, dataset):
    """检查模型前向传播"""
    print("=" * 60)
    print("步骤 2: 检查模型前向传播")
    print("=" * 60)
    
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # 获取一个样本
    data_iter = iter(dataset)
    data = next(data_iter)
    model.set_input(data)
    
    print("执行 forward()...")
    try:
        with torch.no_grad():
            model.forward()
        
        print("✓ Forward 成功")
        print(f"fake_B: shape={model.fake_B.shape}, range=[{model.fake_B.min():.4f}, {model.fake_B.max():.4f}]")
        print(f"  has_nan: {torch.isnan(model.fake_B).any()}, has_inf: {torch.isinf(model.fake_B).any()}")
        
        print(f"fake_A: shape={model.fake_A.shape}, range=[{model.fake_A.min():.4f}, {model.fake_A.max():.4f}]")
        print(f"  has_nan: {torch.isnan(model.fake_A).any()}, has_inf: {torch.isinf(model.fake_A).any()}")
        
        # 检查 attention
        if hasattr(model, 'a_bg_b'):
            print(f"a_bg_b: shape={model.a_bg_b.shape}, range=[{model.a_bg_b.min():.4f}, {model.a_bg_b.max():.4f}]")
            print(f"  has_nan: {torch.isnan(model.a_bg_b).any()}")
        
        if hasattr(model, 'att_rec_bg_a'):
            print(f"att_rec_bg_a: shape={model.att_rec_bg_a.shape}, range=[{model.att_rec_bg_a.min():.4f}, {model.att_rec_bg_a.max():.4f}]")
            print(f"  has_nan: {torch.isnan(model.att_rec_bg_a).any()}")
        
    except Exception as e:
        print(f"✗ Forward 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n✓ 前向传播检查完成\n")
    return model

def check_losses(model, dataset):
    """检查各个 loss 的计算"""
    print("=" * 60)
    print("步骤 3: 检查 Loss 计算")
    print("=" * 60)
    
    model.train()
    
    try:
        # 重新获取数据
        data_iter = iter(dataset)
        data = next(data_iter)
        model.set_input(data)
        
        # 先执行 forward
        print("执行 forward()...")
        model.forward()
        
        # 然后计算 loss
        print("计算 backward_G()...")
        model.backward_G()
        
        print("\n--- Loss 值检查 ---")
        loss_names = ['G_A', 'G_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 
                     'maskA', 'maskB', 'shapeA', 'shapeB', 'reg', 'reg_global', 'reg_local']
        
        for name in loss_names:
            loss_val = getattr(model, f'loss_{name}', None)
            if loss_val is not None:
                if isinstance(loss_val, torch.Tensor):
                    if loss_val.numel() == 1:
                        val = loss_val.item()
                        is_nan = torch.isnan(loss_val).item()
                    else:
                        val = float('nan')
                        is_nan = torch.isnan(loss_val).any().item()
                else:
                    val = float(loss_val)
                    is_nan = (val != val)  # NaN check
                
                status = "✗ NaN" if is_nan else "✓ OK"
                print(f"{status} {name}: {val:.6f}" if not is_nan else f"{status} {name}: {val}")
            else:
                print(f"⚠ {name}: 不存在")
        
        print("\n--- 总 Loss ---")
        if hasattr(model, 'loss_G'):
            if isinstance(model.loss_G, torch.Tensor):
                if model.loss_G.numel() == 1:
                    total_loss = model.loss_G.item()
                    is_nan = torch.isnan(model.loss_G).item()
                else:
                    total_loss = float('nan')
                    is_nan = True
            else:
                total_loss = float(model.loss_G)
                is_nan = (total_loss != total_loss)
            status = "✗ NaN" if is_nan else "✓ OK"
            print(f"{status} loss_G: {total_loss:.6f}" if not is_nan else f"{status} loss_G: {total_loss}")
        
    except Exception as e:
        print(f"✗ Loss 计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Loss 检查完成\n")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("NaN 排查工具")
    print("=" * 60 + "\n")
    
    try:
        opt, dataset = check_data()
        model = check_model_forward(opt, dataset)
        if model:
            check_losses(model, dataset)
    except Exception as e:
        print(f"\n✗ 排查过程出错: {e}")
        import traceback
        traceback.print_exc()

