import numpy as np
import gstools as gs
import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm

# =====================================================
# 1️⃣ 随机场生成函数
# =====================================================
def generate_random_field(model_type="Exponential", var=1.0, len_scale=50,
                          anis=[1.0, 0.8, 0.6],
                          nx=100, ny=100, nz=60, dx=5, dy=5, dz=3, seed=None):
    """
    生成三维各向异性随机场（基于 GSTools）
    """
    seed = seed or np.random.randint(0, 100000)
    if model_type == "Gaussian":
        model = gs.Gaussian(dim=3, var=var, len_scale=len_scale, anis=anis)
    elif model_type == "Spherical":
        model = gs.Spherical(dim=3, var=var, len_scale=len_scale, anis=anis)
    else:
        model = gs.Exponential(dim=3, var=var, len_scale=len_scale, anis=anis)

    srf = gs.SRF(model, seed=seed)
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dy, dy)
    z = np.arange(0, nz * dz, dz)
    field = srf.structured([x, y, z])
    return field, model, {"var": var, "len_scale": len_scale, "anis": anis, 
                          "seed": seed, "model_type": model_type}


# =====================================================
# 2️⃣ 模拟规则钻孔函数（NumPy 矢量化）
# =====================================================
def simulate_drillholes_from_field(field, dx=5, dy=5, dz=3,
                                   hole_spacing=100, sample_interval=3):
    """
    从三维随机场中抽取规则钻孔样品（NumPy矢量化实现）
    """
    nx, ny, nz = field.shape
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz

    # === 规则布孔坐标 ===
    x_holes = np.arange(0, Lx, hole_spacing)
    y_holes = np.arange(0, Ly, hole_spacing)
    z_samples = np.arange(0, Lz, sample_interval)

    gx, gy, gz = np.meshgrid(x_holes, y_holes, z_samples, indexing="ij")
    coords = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

    # === 坐标转网格索引 ===
    ix = np.clip((coords[:, 0] / dx).astype(int), 0, nx - 1)
    iy = np.clip((coords[:, 1] / dy).astype(int), 0, ny - 1)
    iz = np.clip((coords[:, 2] / dz).astype(int), 0, nz - 1)

    # === 直接矢量索引提取值 ===
    values = field[ix, iy, iz]

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "value": values
    })
    return df


# =====================================================
# 3️⃣ 纯数据保存（无可视化）
# =====================================================
def save_results(field, drill_df, params, base_output="synthetic_data"):
    """
    仅保存数据文件，不生成任何图像
    """
    folder_name = f"{params['seed']}_{params['len_scale']:.1f}_{params['var']:.2f}"
    output_dir = os.path.join(base_output, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # === 核心数据保存 ===
    np.save(f"{output_dir}/synthetic_field.npy", field)
    drill_df.to_csv(f"{output_dir}/drillholes.csv", index=False)
    
    # 保存完整参数（包括模型类型）
    params['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params['field_shape'] = list(field.shape)
    params['n_drillholes'] = len(drill_df)
    
    with open(f"{output_dir}/parameters.json", "w") as f:
        json.dump(params, f, indent=4)

    return output_dir


# =====================================================
# 4️⃣ 单个场生成函数（供并行调用）
# =====================================================
def generate_single_field(args):
    """
    生成单个随机场（并行工作函数）
    args: (i, params, base_output)
    """
    i, params, base_output = args
    
    # 生成随机场
    field, model, full_params = generate_random_field(
        model_type=params["model_type"],
        var=params["var"],
        len_scale=params["len_scale"],
        anis=params["anis"],
        nx=100, ny=100, nz=60, 
        dx=5, dy=5, dz=3
    )

    # 生成钻孔数据
    drill_df = simulate_drillholes_from_field(field)
    
    # 保存数据
    output_dir = save_results(field, drill_df, full_params, base_output=base_output)
    
    # 返回元数据
    return {
        "id": i,
        "folder": os.path.basename(output_dir),
        **full_params
    }


# =====================================================
# 5️⃣ 主流程（支持并行和串行）
# =====================================================
def main(n_fields=10, base_output="synthetic_data", parallel=True, n_workers=None):
    """
    批量生成随机场数据（支持并行加速）
    
    参数:
        n_fields: 生成场的数量
        base_output: 输出目录
        parallel: 是否使用并行（True/False）
        n_workers: 并行进程数（None则自动设为 CPU核心数-1）
    """
    import time
    from multiprocessing import Pool, cpu_count
    
    os.makedirs(base_output, exist_ok=True)
    
    # 自动确定工作进程数
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"\n{'='*60}")
    print(f"🚀 开始批量生成随机场数据（CNN训练专用）")
    print(f"{'='*60}")
    print(f"📊 数量: {n_fields} 个随机场")
    print(f"📁 输出目录: {base_output}")
    print(f"⚡ 模式: {'并行生成' if parallel else '串行生成'}")
    if parallel:
        print(f"🔧 并行进程数: {n_workers} (CPU核心数: {cpu_count()})")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # 预生成所有随机参数（确保可复现）
    np.random.seed(42)  # 可选：设置全局种子
    task_params = []
    for i in range(n_fields):
        params = {
            "model_type": np.random.choice(["Gaussian", "Exponential", "Spherical"]),
            "var": float(np.random.uniform(0.5, 2.0)),
            "len_scale": float(np.random.uniform(30, 80)),
            "anis": np.random.uniform(0.6, 1.0, size=3).tolist(),
        }
        task_params.append((i, params, base_output))
    
    # 执行生成
    if parallel and n_fields > 1:
        # 并行模式
        with Pool(processes=n_workers) as pool:
            metadata_list = list(tqdm(
                pool.imap(generate_single_field, task_params),
                total=n_fields,
                desc="生成进度",
                unit="场"
            ))
    else:
        # 串行模式
        metadata_list = []
        for args in tqdm(task_params, desc="生成进度", unit="场"):
            metadata_list.append(generate_single_field(args))
    
    elapsed_time = time.time() - start_time
    
    # 保存汇总元数据
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = os.path.join(base_output, "dataset_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ 全部完成！")
    print(f"{'='*60}")
    print(f"📊 生成数量: {n_fields} 个随机场")
    print(f"📁 数据目录: {base_output}")
    print(f"📋 元数据汇总: {metadata_path}")
    print(f"⏱️  总耗时: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print(f"⚡ 平均速度: {elapsed_time/n_fields:.1f} 秒/场")
    if parallel:
        print(f"🚀 加速比: ~{n_fields*9/elapsed_time:.1f}x (相比串行9秒/场)")
    print(f"{'='*60}\n")
    
    # 显示数据统计
    print("📈 数据集统计:")
    print(f"  - 模型类型分布: {metadata_df['model_type'].value_counts().to_dict()}")
    print(f"  - 方差范围: [{metadata_df['var'].min():.2f}, {metadata_df['var'].max():.2f}]")
    print(f"  - 相关长度范围: [{metadata_df['len_scale'].min():.1f}, {metadata_df['len_scale'].max():.1f}]")
    print(f"  - 平均钻孔样本数: {metadata_df['n_drillholes'].mean():.0f}")


# =====================================================
# 6️⃣ 执行入口
# =====================================================
if __name__ == "__main__":
    # ==================== 配置参数 ====================
    N_FIELDS = 2000         # 生成场的数量
    OUTPUT_DIR = "synthetic_data"  # 输出目录
    PARALLEL = True       # 是否使用并行（True/False）
    N_WORKERS = None      # 并行进程数（None=自动，或指定数字如4、8）
    
    # ==================== 执行生成 ====================
    main(n_fields=N_FIELDS, base_output=OUTPUT_DIR, 
         parallel=PARALLEL, n_workers=N_WORKERS)
    
    # ==================== 使用示例 ====================
    # 1. 串行生成（调试用）
    # main(n_fields=10, parallel=False)
    
    # 2. 并行生成（默认，自动检测CPU核心数）
    # main(n_fields=100, parallel=True)
    
    # 3. 指定并行进程数
    # main(n_fields=100, parallel=True, n_workers=4)
    
    # 4. 大批量生成
    # main(n_fields=1000, parallel=True)