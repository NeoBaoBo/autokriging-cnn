"""random_field_generator.py：等量缩小的各向异性随机场数据生成（64 方向 range 标签）

- 实际矿山 3750×3000×840m 等量缩小 3 倍 -> 场 450×450×282m
- 钻孔间距 32m（96m/3），垂向采样 1m（2-6m/3）
- 长程结构 major 50-100m（推理时乘回 3 还原真实 150-300m）
- 比例：minor/major 0.6-1.0、vertical/major 0.4-1.0（斑岩型近各向同性）
- dip ±90（覆盖陡倾主轴）
- 标签：64 个方向的 range（斐波那契球面采样）

用法：
  python random_field_generator.py --out_dir synthetic_data_v10 --n_samples 3000
"""
import argparse
import json
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

from scipy.spatial import cKDTree
from gstools.tools.geometric import matrix_rotate

try:
    import gstools as gs
except ImportError:
    gs = None

from AutoKrigingNN import fibonacci_sphere, N_DIR


# ============ 全局配置 ============
INPUT_SIZE = (32, 32, 32)
RAW_GRID_SPACING = (5.0, 5.0, 3.0)
BIG_GRID_SHAPE = (90, 90, 94)    # 450×450×282m（Z 保持 3 倍 range 容纳陡倾；XY 缩到 range 的 4.5 倍省算力）
HOLE_SPACING = 32.0              # 96m / 3

# 长程结构参数范围（缩小 3 倍，对应真实 150-300m -> 50-100m）
MAJOR_MIN, MAJOR_MAX = 50.0, 100.0
MINOR_RATIO_MIN, MINOR_RATIO_MAX = 0.6, 1.0   # 水平近各向同性（斑岩型矿床各向异性小）
VERTICAL_RATIO_MIN = 0.4                       # 垂向比例下限，上限 = minor/major（保证 major>=minor>=vertical）
SILL_MIN, SILL_MAX = 0.3, 1.0
NUGGET_RATIO = 0.08


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_valid(arr, mask, fill_value=0.0):
    """空值填充，不做 z-score（z-score 在训练/推理阶段做），保留原始品位量级"""
    arr = arr.astype(np.float32).copy()
    valid = mask > 0.5
    arr[~valid] = fill_value
    return arr.astype(np.float32)


# ============ 随机场生成 ============
def generate_sgs_field(params, grid_shape, spacing, seed=None):
    """用 GSTools 生成各向异性随机场（真实旋转主轴）"""
    if gs is None:
        raise ImportError("未安装 gstools，请先执行：pip install gstools")

    nx, ny, nz = grid_shape
    dx, dy, dz = spacing
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dy, dy)
    z = np.arange(0, nz * dz, dz)

    major = params["major"]
    minor = params["minor"]
    vertical = params["vertical"]
    anis_ratio = [minor / major, vertical / major]
    angles_rad = np.radians([params["azimuth"], params["dip"], params["plunge"]])

    model_type = params["model_type"]
    if model_type == "Gaussian":
        model = gs.Gaussian(dim=3, var=params["sill"], len_scale=major, anis=anis_ratio, angles=angles_rad)
    elif model_type == "Spherical":
        model = gs.Spherical(dim=3, var=params["sill"], len_scale=major, anis=anis_ratio, angles=angles_rad)
    else:
        model = gs.Exponential(dim=3, var=params["sill"], len_scale=major, anis=anis_ratio, angles=angles_rad)

    srf = gs.SRF(model, seed=seed)
    field = srf.structured([x, y, z]).astype(np.float32)

    nugget = params["nugget"]
    if nugget > 0:
        noise = np.random.normal(0, np.sqrt(nugget), size=field.shape).astype(np.float32)
        field = field + noise
    return field


# ============ 钻孔模拟 ============
def simulate_drillholes(field, spacing, hole_spacing_xy, sample_interval_z, missing_rate, depth_ratio, seed):
    rng = np.random.default_rng(seed)
    dx, dy, dz = spacing
    nx, ny, nz = field.shape
    lx, ly, lz = nx * dx, ny * dy, nz * dz

    xs = np.arange(0, lx, hole_spacing_xy)
    ys = np.arange(0, ly, hole_spacing_xy)

    records = []
    for xh in xs:
        for yh in ys:
            max_depth = lz * rng.uniform(0.70, depth_ratio)
            zs = np.arange(0, max_depth, sample_interval_z)
            for zc in zs:
                if rng.random() < missing_rate:
                    continue
                ix = min(int(xh / dx), nx - 1)
                iy = min(int(yh / dy), ny - 1)
                iz = min(int(zc / dz), nz - 1)
                records.append([xh, yh, zc, float(field[ix, iy, iz])])

    df = pd.DataFrame(records, columns=["x", "y", "z", "value"])
    meta = {
        "hole_spacing_xy": hole_spacing_xy,
        "sample_interval_z": sample_interval_z,
        "missing_rate": missing_rate,
        "depth_ratio": depth_ratio,
        "n_samples": int(len(df)),
    }
    return df, meta


# ============ IDW 体素化 ============
def idw_interpolation_3d(drill_df, target_grid_shape, k=12, power=2.0):
    if len(drill_df) == 0:
        raise ValueError("钻孔数据为空，无法进行 IDW 重建")

    coords = drill_df[["x", "y", "z"]].values.astype(np.float32)
    values = drill_df["value"].values.astype(np.float32)

    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)

    gx = np.linspace(xmin, xmax, target_grid_shape[0], dtype=np.float32)
    gy = np.linspace(ymin, ymax, target_grid_shape[1], dtype=np.float32)
    gz = np.linspace(zmin, zmax, target_grid_shape[2], dtype=np.float32)

    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    target_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    tree = cKDTree(coords)
    dists, idxs = tree.query(target_points, k=min(k, len(coords)))

    if len(coords) == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    dists = np.maximum(dists, 1e-6)
    weights = 1.0 / (dists ** power)
    neighbor_vals = values[idxs]
    pred = np.sum(weights * neighbor_vals, axis=1) / np.sum(weights, axis=1)

    value_grid = pred.reshape(target_grid_shape).astype(np.float32)

    nearest_dist = dists[:, 0]
    threshold = np.percentile(nearest_dist, 70)
    mask_grid = (nearest_dist.reshape(target_grid_shape) <= threshold).astype(np.float32)

    return value_grid, mask_grid


# ============ 64 方向 range 标签 ============
def directional_ranges(azimuth, dip, plunge, major, minor, vertical, directions):
    """按各向异性椭球计算 64 个方向上的 range"""
    angles_rad = np.radians([azimuth, dip, plunge])
    R = matrix_rotate(3, angles_rad)
    e = R.T
    a = np.array([major, minor, vertical], dtype=np.float64)
    ranges = []
    for u in directions:
        inv_r2 = sum(((u @ e[k]) ** 2) / (a[k] ** 2) for k in range(3))
        ranges.append(1.0 / np.sqrt(inv_r2))
    return np.array(ranges, dtype=np.float32)


# ============ 单样本生成 ============
def gen_case(args):
    group_dir, case_id, base_seed = args
    seed = base_seed + case_id
    set_seed(seed)
    rng = np.random.default_rng(seed)

    major = float(rng.uniform(MAJOR_MIN, MAJOR_MAX))
    minor_ratio = float(rng.uniform(MINOR_RATIO_MIN, MINOR_RATIO_MAX))
    minor = float(major * minor_ratio)
    vertical = float(major * rng.uniform(VERTICAL_RATIO_MIN, minor_ratio))
    sill = float(rng.uniform(SILL_MIN, SILL_MAX))
    nugget = float(sill * NUGGET_RATIO)
    azimuth = float(rng.uniform(0, 360))
    dip = float(rng.uniform(-90, 90))
    plunge = float(rng.uniform(-30, 30))

    params = {
        "model_type": rng.choice(["Gaussian", "Exponential", "Spherical"]),
        "sill": sill, "nugget": nugget,
        "range": major, "major": major, "minor": minor, "vertical": vertical,
        "azimuth": azimuth, "dip": dip, "plunge": plunge,
    }

    field_raw = generate_sgs_field(params, BIG_GRID_SHAPE, RAW_GRID_SPACING, seed=base_seed + 10000 + case_id)
    drill_df, drill_meta = simulate_drillholes(
        field_raw, spacing=RAW_GRID_SPACING, hole_spacing_xy=HOLE_SPACING,
        sample_interval_z=1.0, missing_rate=0.05, depth_ratio=0.8, seed=base_seed + 20000 + case_id
    )
    idw_grid, mask_grid = idw_interpolation_3d(drill_df, INPUT_SIZE, k=12, power=2.0)
    idw_grid = normalize_valid(idw_grid, mask_grid, fill_value=0.0)

    directions = fibonacci_sphere(N_DIR)
    ranges = directional_ranges(azimuth, dip, plunge, major, minor, vertical, directions)

    case_dir = Path(group_dir) / f"case_{case_id:06d}"
    ensure_dir(case_dir)
    np.save(case_dir / "idw_grid.npy", idw_grid)
    np.save(case_dir / "mask_grid.npy", mask_grid)
    np.save(case_dir / "ranges.npy", ranges)
    drill_df.to_csv(case_dir / "drillholes.csv", index=False)
    with open(case_dir / "parameters.json", "w", encoding="utf-8") as f:
        json.dump({**params, **drill_meta}, f, ensure_ascii=False, indent=2)
    return case_id


# ============ 并行生成 ============
def gen_data(group_dir, n, base_seed, n_workers):
    ensure_dir(group_dir)
    tasks = [(group_dir, i, base_seed) for i in range(n)]
    log(f"生成 {group_dir}: {n} 样本, {n_workers} 进程")
    with Pool(processes=n_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(gen_case, tasks), 1):
            if i % 200 == 0 or i == n:
                log(f"  {group_dir}: {i}/{n}")
    log(f"  {group_dir}: 完成")


def main():
    ap = argparse.ArgumentParser(description="生成等量缩小(K=3)各向异性随机场训练数据（64 方向 range 标签）")
    ap.add_argument("--out_dir", default="synthetic_data_v10", help="输出目录")
    ap.add_argument("--n_samples", type=int, default=3000, help="样本数量")
    ap.add_argument("--n_workers", type=int, default=None, help="并行进程数，默认 CPU 核数-1")
    args = ap.parse_args()

    n_workers = args.n_workers or max(1, cpu_count() - 1)
    log(f"生成等量缩小(K=3)数据: {args.n_samples} 样本 -> {args.out_dir}, "
        f"场 {BIG_GRID_SHAPE}, 间距 {HOLE_SPACING}m, {n_workers} 进程")
    gen_data(args.out_dir, args.n_samples, 4000, n_workers)
    log("全部完成")


if __name__ == "__main__":
    main()
