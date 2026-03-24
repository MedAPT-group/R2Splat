"""
分析当前代码的 FOV Clipping 界限并给出软 clipping 的参数建议
"""
import numpy as np

def analyze_fov_limits():
    """分析 FOV clipping 的界限"""
    
    print("="*80)
    print("当前代码的 FOV Clipping 界限分析")
    print("="*80)
    
    # 假设典型的相机参数
    scenarios = [
        {
            "name": "标准相机 (512×512, fx=fy=256)",
            "width": 512, "height": 512,
            "fx": 256, "fy": 256,
            "cx": 256, "cy": 256
        },
        {
            "name": "宽屏相机 (1920×1080, fx=fy=1000)",
            "width": 1920, "height": 1080,
            "fx": 1000, "fy": 1000,
            "cx": 960, "cy": 540
        },
        {
            "name": "正方形相机 (1024×1024, fx=fy=512)",
            "width": 1024, "height": 1024,
            "fx": 512, "fy": 512,
            "cx": 512, "cy": 512
        },
    ]
    
    print("\n代码中的界限公式:")
    print("  tan_fovx = 0.5 * width / fx")
    print("  tan_fovy = 0.5 * height / fy")
    print("  lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx")
    print("  lim_x_neg = cx / fx + 0.3 * tan_fovx")
    print("  lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy")
    print("  lim_y_neg = cy / fy + 0.3 * tan_fovy")
    print()
    print("裁剪条件: -lim_x_neg ≤ x/z ≤ lim_x_pos")
    print("="*80)
    
    for scenario in scenarios:
        W = scenario["width"]
        H = scenario["height"]
        fx = scenario["fx"]
        fy = scenario["fy"]
        cx = scenario["cx"]
        cy = scenario["cy"]
        
        print(f"\n{scenario['name']}")
        print("-"*80)
        
        # 计算 FOV
        fov_x_deg = 2 * np.arctan(W / (2 * fx)) * 180 / np.pi
        fov_y_deg = 2 * np.arctan(H / (2 * fy)) * 180 / np.pi
        
        print(f"相机参数: W={W}, H={H}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"视场角: FOVx = {fov_x_deg:.2f}°, FOVy = {fov_y_deg:.2f}°")
        
        # 计算界限
        tan_fovx = 0.5 * W / fx
        tan_fovy = 0.5 * H / fy
        
        lim_x_pos = (W - cx) / fx + 0.3 * tan_fovx
        lim_x_neg = cx / fx + 0.3 * tan_fovx
        lim_y_pos = (H - cy) / fy + 0.3 * tan_fovy
        lim_y_neg = cy / fy + 0.3 * tan_fovy
        
        print(f"\n计算的界限:")
        print(f"  tan(FOVx/2) = {tan_fovx:.4f}")
        print(f"  tan(FOVy/2) = {tan_fovy:.4f}")
        print(f"\n  x/z 的界限:")
        print(f"    lim_x_pos = {lim_x_pos:.4f}")
        print(f"    lim_x_neg = {lim_x_neg:.4f}")
        print(f"    范围: [{-lim_x_neg:.4f}, {lim_x_pos:.4f}]")
        
        print(f"\n  y/z 的界限:")
        print(f"    lim_y_pos = {lim_y_pos:.4f}")
        print(f"    lim_y_neg = {lim_y_neg:.4f}")
        print(f"    范围: [{-lim_y_neg:.4f}, {lim_y_pos:.4f}]")
        
        # 分析界限的物理意义
        # 基础视场范围（从图像边界）
        base_x_pos = (W - cx) / fx
        base_x_neg = cx / fx
        base_y_pos = (H - cy) / fy
        base_y_neg = cy / fy
        
        # 扩展量
        extend_x = 0.3 * tan_fovx
        extend_y = 0.3 * tan_fovy
        
        print(f"\n  界限分解:")
        print(f"    基础范围(图像边界): x/z ∈ [{-base_x_neg:.4f}, {base_x_pos:.4f}]")
        print(f"    扩展量: ±{extend_x:.4f} (即 0.3 × tan(FOVx/2))")
        print(f"    总范围: x/z ∈ [{-lim_x_neg:.4f}, {lim_x_pos:.4f}]")
        print(f"    扩展比例: {(extend_x/base_x_pos)*100:.1f}% (相对于基础范围)")
        
        # 对应的角度
        angle_pos = np.arctan(lim_x_pos) * 180 / np.pi
        angle_neg = np.arctan(lim_x_neg) * 180 / np.pi
        
        print(f"\n  对应角度:")
        print(f"    正向: arctan({lim_x_pos:.4f}) = {angle_pos:.2f}°")
        print(f"    负向: arctan({lim_x_neg:.4f}) = {angle_neg:.2f}°")
        print(f"    总角度范围: {angle_pos + angle_neg:.2f}° (≈ {fov_x_deg * 1.3:.2f}°, 是FOV的1.3倍)")

def recommend_soft_clipping_params():
    """推荐软 clipping 的参数"""
    
    print("\n\n" + "="*80)
    print("软 Clipping (Tanh) 参数推荐")
    print("="*80)
    
    print("\n理论分析:")
    print("-"*80)
    print("硬 clipping: t = z × clamp(x/z, -L_neg, L_pos)")
    print("软 clipping: t = z × s × tanh((x/z) / s)")
    print()
    print("其中 s 是软化尺度参数，控制:")
    print("  1. s 越大 → 过渡越平缓，越远离硬 clipping")
    print("  2. s 越小 → 过渡越陡峭，越接近硬 clipping")
    print("  3. tanh 在 ±1 处达到 ~76% 的饱和值")
    print()
    
    print("参数选择策略:")
    print("  s = α × L，其中 L 是硬 clipping 的界限")
    print()
    print("  α = 0.6: 非常接近硬 clipping (tanh(1.67) ≈ 0.93)")
    print("  α = 0.7: 较接近硬 clipping (tanh(1.43) ≈ 0.89)")
    print("  α = 0.8: 平衡选择 (tanh(1.25) ≈ 0.85) ← 推荐")
    print("  α = 0.9: 更平滑 (tanh(1.11) ≈ 0.80)")
    print("  α = 1.0: 最平滑 (tanh(1.0) ≈ 0.76)")
    print()
    
    print("="*80)
    print("推荐配置")
    print("="*80)
    
    scenarios = [
        {
            "name": "标准相机 (512×512)",
            "lim_x": 1.325,  # 典型值
            "lim_y": 1.325,
        },
        {
            "name": "宽屏相机 (1920×1080)",
            "lim_x": 1.248,
            "lim_y": 0.702,
        },
    ]
    
    alphas = [0.7, 0.8, 0.9]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-"*80)
        L_x = scenario["lim_x"]
        L_y = scenario["lim_y"]
        
        print(f"硬 clipping 界限: L_x ≈ {L_x:.3f}, L_y ≈ {L_y:.3f}")
        print()
        
        for alpha in alphas:
            s_x = alpha * L_x
            s_y = alpha * L_y
            
            # 在界限处的值
            t_at_limit = s_x * np.tanh(L_x / s_x)
            saturation = t_at_limit / L_x * 100
            
            print(f"  α = {alpha}:")
            print(f"    s_x = {s_x:.3f}, s_y = {s_y:.3f}")
            print(f"    在 x/z = L_x 处: t/z = {t_at_limit:.3f} ({saturation:.1f}% 饱和)")
            print(f"    导数: dt/dx|_{x/z=L} = {1/np.cosh(L_x/s_x)**2:.3f}")
            print()

def code_example():
    """给出代码示例"""
    
    print("\n" + "="*80)
    print("代码实现建议")
    print("="*80)
    
    print("""
在 proj.cuh 的 persp_proj 函数中:

原始代码:
```cuda
T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
T lim_x_neg = cx / fx + 0.3f * tan_fovx;
T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));
```

修改为软 clipping (推荐 α = 0.8):
```cuda
T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
T lim_x_neg = cx / fx + 0.3f * tan_fovx;
T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
T lim_y_neg = cy / fy + 0.3f * tan_fovy;

// 软化尺度 (0.8 是推荐值，可调整)
const T alpha = 0.8f;
T scale_x = alpha * lim_x_pos;  // 假设对称
T scale_y = alpha * lim_y_pos;

// 软 clipping
T tx, dtx_dx, dtx_dz;
soft_clip_fwd(x, z, scale_x, tx, dtx_dx, dtx_dz);

T ty, dty_dy, dty_dz;
soft_clip_fwd(y, z, scale_y, ty, dty_dy, dty_dz);
```

或者更简单的内联实现:
```cuda
T scale_x = 0.8f * lim_x_pos;
T scale_y = 0.8f * lim_y_pos;

T xi_x = x / (z * scale_x);
T xi_y = y / (z * scale_y);

T tx = z * scale_x * tanh(xi_x);
T ty = z * scale_y * tanh(xi_y);
```

注意事项:
1. 如果左右边界不对称，需要先判断 x 的符号，选择 lim_x_pos 或 lim_x_neg
2. 对于中心化相机 (cx ≈ W/2), 可以简化为对称处理
3. α = 0.8 是推荐值，可以通过实验调整 (0.7-0.9 范围)
""")

def summary():
    """总结"""
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    print("""
1. 当前代码的 FOV clipping 界限:
   - lim_x_pos = (width - cx) / fx + 0.3 × tan(FOVx/2)
   - lim_x_neg = cx / fx + 0.3 × tan(FOVy/2)
   - 典型值: x/z ∈ [-1.3, 1.3] (对于512×512相机)
   - 物理意义: 允许高斯中心在图像边界外延伸 30% 的视场角

2. 软 clipping 参数推荐:
   - 公式: s = α × L，其中 L 是硬界限
   - 推荐值: α = 0.8
   - 对于标准相机: s_x ≈ 1.06, s_y ≈ 1.06
   - 对于宽屏相机: s_x ≈ 1.00, s_y ≈ 0.56

3. 为什么选择 α = 0.8:
   - 在 x/z = L 处达到 85% 饱和，合理的过渡
   - 一阶导数在边界处 ≈ 0.39 (平滑但不过于平缓)
   - 二阶导数连续且有界
   - 平衡了保真度和平滑性

4. 如何使用:
   - 直接在投影函数中用 tanh 替换 clamp
   - 反向传播自动变平滑，无需修改梯度代码
   - 可通过调整 α ∈ [0.7, 0.9] 来控制平滑程度
""")
    print("="*80)

if __name__ == "__main__":
    analyze_fov_limits()
    recommend_soft_clipping_params()
    code_example()
    summary()
