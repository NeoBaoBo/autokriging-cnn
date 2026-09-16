"""CNN_train.py：训练 AutoKrigingNN 模型（z-score 输入，预测 64 方向 range）

用法：
  python CNN_train.py --data_dir synthetic_data_v10 --save_dir trained_models --epochs 300
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from AutoKrigingNN import (
    AutoKrigingNN, N_DIR, DEVICE, fibonacci_sphere,
    fit_tensor_psd, tensor_to_axes_angles, angle_between,
)

DATA_ROOT = "synthetic_data_v10"
SAVE_DIR = "trained_models"
MAX_EPOCHS = 300
BATCH_SIZE = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RangeDataset(Dataset):
    """带 z-score 的数据集：加载时对 idw_grid 有效体素做 z-score（与推理一致）"""

    def __init__(self, data_dir):
        self.samples = sorted(Path(data_dir).glob("case_*"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        idw = np.load(d / "idw_grid.npy").astype(np.float32)
        mask = np.load(d / "mask_grid.npy").astype(np.float32)
        valid = mask > 0.5
        if np.any(valid):
            mean = idw[valid].mean()
            std = idw[valid].std()
            if std < 1e-6:
                std = 1.0
            idw[valid] = (idw[valid] - mean) / std
        idw[~valid] = 0.0
        x = np.stack([idw, mask], axis=0)
        y = np.load(d / "ranges.npy").astype(np.float32)
        with open(d / "parameters.json") as f:
            p = json.load(f)
        return torch.from_numpy(x), torch.from_numpy(y), (p["azimuth"], p["dip"], p["plunge"])


def evaluate_loss(model, loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n


def main():
    ap = argparse.ArgumentParser(description="训练 AutoKrigingNN（64 方向 range 预测）")
    ap.add_argument("--data_dir", default=DATA_ROOT)
    ap.add_argument("--save_dir", default=SAVE_DIR)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()
    data_root = args.data_dir
    save_dir = args.save_dir
    max_epochs = args.epochs
    batch_size = args.batch_size

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log(f"训练 AutoKrigingNN: epochs={max_epochs}, batch={batch_size}, data={data_root}")

    ds = RangeDataset(data_root)
    n = len(ds)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    log(f"训练 {n_train}, 验证 {n_val}, 测试 {n_test}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = AutoKrigingNN(out_dim=N_DIR).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_epoch = 0
    history = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        avg_train_loss = total_loss / len(train_loader)

        record = {"epoch": epoch, "train_loss": avg_train_loss, "lr": opt.param_groups[0]["lr"]}
        if epoch % 20 == 0 or epoch == max_epochs:
            val_loss = evaluate_loss(model, val_loader)
            record["val_loss"] = val_loss
            log(f"  epoch {epoch:3d}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
        history.append(record)

    pd.DataFrame(history).to_csv(f"{save_dir}/training_history.csv", index=False)
    log(f"最佳 epoch: {best_epoch}, best_val_loss={best_val_loss:.4f}")

    # 方向恢复评估
    log("方向恢复评估 ...")
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pth", weights_only=True))
    model.eval()
    directions = fibonacci_sphere(N_DIR)
    axis_errs = []
    angle_errs = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            pred = model(x.to(DEVICE)).cpu().numpy()
            true = y.numpy()
            for i in range(len(pred)):
                D_pred = fit_tensor_psd(directions, pred[i])
                a_pred, v_pred = tensor_to_axes_angles(D_pred)
                D_true = fit_tensor_psd(directions, true[i])
                a_true, v_true = tensor_to_axes_angles(D_true)
                axis_errs.append(np.abs(a_pred - a_true) / a_true)
                angle_errs.append(angle_between(v_pred[:, 0], v_true[:, 0]))

    axis_errs = np.array(axis_errs)
    angle_errs = np.array(angle_errs)
    log("=" * 60)
    log(f"轴长相对误差: major={axis_errs[:,0].mean():.1%}, minor={axis_errs[:,1].mean():.1%}, vertical={axis_errs[:,2].mean():.1%}")
    log(f"主轴方向夹角误差: 均值={angle_errs.mean():.1f}°, 中位数={np.median(angle_errs):.1f}°")

    eval_result = {
        "axis_err_major": float(axis_errs[:, 0].mean()),
        "axis_err_minor": float(axis_errs[:, 1].mean()),
        "axis_err_vertical": float(axis_errs[:, 2].mean()),
        "angle_err_mean": float(angle_errs.mean()),
        "angle_err_median": float(np.median(angle_errs)),
        "best_epoch": best_epoch,
    }
    with open(f"{save_dir}/eval_result.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    elapsed = (time.time() - t0) / 60
    log(f"训练完成，总耗时 {elapsed:.1f} 分钟")


if __name__ == "__main__":
    main()
