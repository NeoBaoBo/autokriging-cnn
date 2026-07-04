import os
import json
import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# =============================
# 全局配置
# =============================
DATA_DIR = "synthetic_data"
MODEL_SAVE_DIR = "trained_models"
LOG_DIR = "training_logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_SEED = 42
INPUT_SIZE = (32, 32, 32)
USE_DATA_AUGMENTATION = True
EARLY_STOPPING_PATIENCE = 20

# =============================
# 参数范围（3+6输出）
# Variogram: [nugget, sill, range]
# Ellipsoid: [major, minor, vertical, azimuth, dip, plunge]
# =============================
NUGGET_MIN, NUGGET_MAX = 0.0001, 0.3
SILL_MIN, SILL_MAX = 0.001, 1.8
RANGE_MIN, RANGE_MAX = 5.0, 150.0

MAJOR_MIN, MAJOR_MAX = 10.0, 500.0
MINOR_MIN, MINOR_MAX = 5.0, 300.0
VERTICAL_MIN, VERTICAL_MAX = 2.0, 150.0

AZIMUTH_MIN, AZIMUTH_MAX = 0.0, 360.0
DIP_MIN, DIP_MAX = -90.0, 90.0
PLUNGE_MIN, PLUNGE_MAX = -90.0, 90.0

# 控制 sill 的批内波动目标
TARGET_SILL_STD = 0.08

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_float32_matmul_precision("medium")


# =============================
# 随机种子
# =============================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_SEED)


# =============================
# 日志配置
# =============================
log_file = os.path.join(LOG_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
writer = SummaryWriter(os.path.join(LOG_DIR, "tensorboard"))


# =============================
# 数据增强
# 只对输入体素做增强，不改标签
# =============================
class DataAugmentation:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, noise_prob=0.15, noise_std=0.01):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.flip_prob:
            dim = np.random.choice([1, 2, 3])
            x = torch.flip(x, dims=[dim])

        if torch.rand(1).item() < self.rotate_prob:
            k = np.random.randint(1, 4)
            x = torch.rot90(x, k=k, dims=[2, 3])

        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x


# =============================
# 数据集定义：读取 synthetic_data_v3
# 输入:
#   voxel_value.npy
# 标签:
#   parameters.json 中的 3+6 参数
# 预留:
#   voxel_mask.npy （后续双通道可直接接）
# =============================
class GeologicalDatasetV3(Dataset):
    def __init__(self, data_dir, transform=None, use_mask=False):
        self.data_dir = data_dir
        self.transform = transform
        self.use_mask = use_mask
        self.samples = []
        self._load_metadata()

    def _load_metadata(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            voxel_value_path = os.path.join(folder_path, "voxel_value.npy")
            voxel_mask_path = os.path.join(folder_path, "voxel_mask.npy")
            param_path = os.path.join(folder_path, "parameters.json")

            if os.path.exists(voxel_value_path) and os.path.exists(param_path):
                sample = {
                    "voxel_value_path": voxel_value_path,
                    "param_path": param_path
                }
                if os.path.exists(voxel_mask_path):
                    sample["voxel_mask_path"] = voxel_mask_path
                self.samples.append(sample)

        if len(self.samples) == 0:
            raise RuntimeError("未找到有效样本，请检查 synthetic_data_v3 目录结构。")

        logger.info(f"成功加载样本: {len(self.samples)} 个")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        voxel_value = np.load(sample["voxel_value_path"]).astype(np.float32)

        # 单通道版本：只用 voxel_value
        if self.use_mask and "voxel_mask_path" in sample:
            voxel_mask = np.load(sample["voxel_mask_path"]).astype(np.float32)
            x = np.stack([voxel_value, voxel_mask], axis=0)   # [2, D, H, W]
        else:
            x = np.expand_dims(voxel_value, axis=0)           # [1, D, H, W]

        with open(sample["param_path"], "r", encoding="utf-8") as f:
            params = json.load(f)

        # Variogram 标签
        nugget = float(params.get("nugget", 0.05))
        sill = float(params.get("sill", params.get("var", 1.0)))
        rng = float(params.get("range", params.get("len_scale", 30.0)))

        nugget = np.clip(nugget, NUGGET_MIN, NUGGET_MAX)
        sill = np.clip(sill, SILL_MIN, SILL_MAX)
        rng = np.clip(rng, RANGE_MIN, RANGE_MAX)

        variogram_params = np.array([nugget, sill, rng], dtype=np.float32)

        # Ellipsoid 标签
        major = float(params.get("major", rng))
        minor = float(params.get("minor", major * 0.7))
        vertical = float(params.get("vertical", major * 0.4))
        azimuth = float(params.get("azimuth", 0.0))
        dip = float(params.get("dip", 0.0))
        plunge = float(params.get("plunge", 0.0))

        major = np.clip(major, MAJOR_MIN, MAJOR_MAX)
        minor = np.clip(minor, MINOR_MIN, min(MINOR_MAX, major))
        vertical = np.clip(vertical, VERTICAL_MIN, min(VERTICAL_MAX, minor))
        azimuth = np.clip(azimuth, AZIMUTH_MIN, AZIMUTH_MAX)
        dip = np.clip(dip, DIP_MIN, DIP_MAX)
        plunge = np.clip(plunge, PLUNGE_MIN, PLUNGE_MAX)

        ellipsoid_params = np.array(
            [major, minor, vertical, azimuth, dip, plunge],
            dtype=np.float32
        )

        x_tensor = torch.from_numpy(x).float()
        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        return (
            x_tensor,
            torch.tensor(variogram_params),
            torch.tensor(ellipsoid_params)
        )


# =============================
# 模型：保留旧版 3+6 输出
# 支持单通道/双通道
# =============================
class AutoKrigingCNN_V3(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Variogram: 3
        self.fc_v = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Ellipsoid: 6
        self.fc_e = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    @staticmethod
    def scale_to_range(x, min_val, max_val):
        return min_val + (max_val - min_val) * torch.sigmoid(x)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        v_raw = self.fc_v(x)
        e_raw = self.fc_e(x)

        # Variogram: [nugget, sill, range]
        pred_nugget = self.scale_to_range(v_raw[:, 0], NUGGET_MIN, NUGGET_MAX)
        pred_sill = self.scale_to_range(v_raw[:, 1], SILL_MIN, SILL_MAX)
        pred_range = self.scale_to_range(v_raw[:, 2], RANGE_MIN, RANGE_MAX)

        # 物理约束：nugget <= sill
        pred_nugget = torch.minimum(pred_nugget, pred_sill)

        v_out = torch.stack([pred_nugget, pred_sill, pred_range], dim=1)

        # Ellipsoid: [major, minor, vertical, azimuth, dip, plunge]
        pred_major = self.scale_to_range(e_raw[:, 0], MAJOR_MIN, MAJOR_MAX)
        pred_minor = self.scale_to_range(e_raw[:, 1], MINOR_MIN, MINOR_MAX)
        pred_vertical = self.scale_to_range(e_raw[:, 2], VERTICAL_MIN, VERTICAL_MAX)

        # 物理约束：major >= minor >= vertical
        pred_minor = torch.minimum(pred_minor, pred_major)
        pred_vertical = torch.minimum(pred_vertical, pred_minor)

        pred_azimuth = self.scale_to_range(e_raw[:, 3], AZIMUTH_MIN, AZIMUTH_MAX)
        pred_dip = self.scale_to_range(e_raw[:, 4], DIP_MIN, DIP_MAX)
        pred_plunge = self.scale_to_range(e_raw[:, 5], PLUNGE_MIN, PLUNGE_MAX)

        e_out = torch.stack([
            pred_major,
            pred_minor,
            pred_vertical,
            pred_azimuth,
            pred_dip,
            pred_plunge
        ], dim=1)

        return v_out, e_out


# =============================
# 损失函数：旧框架 + 新约束思想
# =============================
class MultiTaskLossV3(nn.Module):
    def __init__(
        self,
        w_v=1.0,
        w_e=0.3,
        w_std=0.8,
        w_nugget=0.2,
        w_axis=0.2,
        target_sill_std=0.08
    ):
        super().__init__()
        self.w_v = w_v
        self.w_e = w_e
        self.w_std = w_std
        self.w_nugget = w_nugget
        self.w_axis = w_axis
        self.target_sill_std = target_sill_std
        self.reg_loss = nn.SmoothL1Loss(beta=0.1)

    def forward(self, pv, pe, tv, te):
        loss_v = self.reg_loss(pv, tv)
        loss_e = self.reg_loss(pe, te)

        # sill 批内波动约束
        sill_pred = pv[:, 1]
        sill_std = torch.std(sill_pred, unbiased=False)
        loss_std = (sill_std - self.target_sill_std) ** 2

        # nugget 不应大于 sill
        nugget_pred = pv[:, 0]
        nugget_over_penalty = torch.mean(F.relu(nugget_pred - sill_pred) ** 2)

        # major >= minor >= vertical
        major = pe[:, 0]
        minor = pe[:, 1]
        vertical = pe[:, 2]
        axis_penalty = torch.mean(
            F.relu(minor - major) ** 2 + F.relu(vertical - minor) ** 2
        )

        total_loss = (
            self.w_v * loss_v +
            self.w_e * loss_e +
            self.w_std * loss_std +
            self.w_nugget * nugget_over_penalty +
            self.w_axis * axis_penalty
        )

        metrics = {
            "loss_v": loss_v.item(),
            "loss_e": loss_e.item(),
            "loss_std": loss_std.item(),
            "nugget_penalty": nugget_over_penalty.item(),
            "axis_penalty": axis_penalty.item(),
            "sill_std": sill_std.item()
        }

        return total_loss, metrics


# =============================
# 训练函数
# =============================
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()

    total_loss = 0.0
    total_metrics = {
        "loss_v": 0.0,
        "loss_e": 0.0,
        "loss_std": 0.0,
        "nugget_penalty": 0.0,
        "axis_penalty": 0.0,
        "sill_std": 0.0
    }

    loop = tqdm(dataloader, desc="Training", leave=False)

    for x, v, e in loop:
        x = x.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pv, pe = model(x)
                loss, metrics = criterion(pv, pe, v, e)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pv, pe = model(x)
            loss, metrics = criterion(pv, pe, v, e)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    n = len(dataloader)
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return total_loss / n, avg_metrics


# =============================
# 验证函数
# =============================
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_metrics = {
        "loss_v": 0.0,
        "loss_e": 0.0,
        "loss_std": 0.0,
        "nugget_penalty": 0.0,
        "axis_penalty": 0.0,
        "sill_std": 0.0
    }

    for x, v, e in dataloader:
        x = x.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)

        pv, pe = model(x)
        loss, metrics = criterion(pv, pe, v, e)

        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    n = len(dataloader)
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return total_loss / n, avg_metrics


# =============================
# 保存配置
# =============================
def save_training_config(use_mask):
    config = {
        "DATA_DIR": DATA_DIR,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "VAL_SIZE": VAL_SIZE,
        "TEST_SIZE": TEST_SIZE,
        "RANDOM_SEED": RANDOM_SEED,
        "INPUT_SIZE": INPUT_SIZE,
        "TARGET_SILL_STD": TARGET_SILL_STD,
        "USE_MASK": use_mask
    }
    path = os.path.join(LOG_DIR, "training_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存训练配置: {path}")


# =============================
# 主函数
# =============================
def main():
    # 这里先保持单通道；后续你要双通道时改成 True
    USE_MASK = False

    logger.info(f"使用设备: {DEVICE}")
    save_training_config(USE_MASK)

    transform = DataAugmentation() if USE_DATA_AUGMENTATION else None
    full_dataset = GeologicalDatasetV3(
        DATA_DIR,
        transform=transform,
        use_mask=USE_MASK
    )

    total_len = len(full_dataset)
    val_len = int(total_len * VAL_SIZE)
    test_len = int(total_len * TEST_SIZE)
    train_len = total_len - val_len - test_len

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    num_workers = max(2, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    logger.info(f"训练集: {train_len} | 验证集: {val_len} | 测试集: {test_len}")

    in_channels = 2 if USE_MASK else 1
    model = AutoKrigingCNN_V3(in_channels=in_channels).to(DEVICE)
    logger.info(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = MultiTaskLossV3(target_sill_std=TARGET_SILL_STD)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=8,
        factor=0.5
    )

    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_loss)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        for k, v in train_metrics.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        logger.info(
            f"Epoch [{epoch+1:03d}/{EPOCHS}] "
            f"Train={train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.6f} | "
            f"Val_loss_v={val_metrics['loss_v']:.4f}, "
            f"Val_loss_e={val_metrics['loss_e']:.4f}, "
            f"Val_sill_std={val_metrics['sill_std']:.4f}, "
            f"Val_axis_penalty={val_metrics['axis_penalty']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"保存最佳模型: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("触发 Early Stopping，提前结束训练。")
                break

    # =============================
    # 测试阶段
    # =============================
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    test_loss, test_metrics = validate(model, test_loader, criterion, DEVICE)

    logger.info(
        f"测试集结果 | "
        f"Loss={test_loss:.4f}, "
        f"loss_v={test_metrics['loss_v']:.4f}, "
        f"loss_e={test_metrics['loss_e']:.4f}, "
        f"loss_std={test_metrics['loss_std']:.4f}, "
        f"sill_std={test_metrics['sill_std']:.4f}"
    )

    test_result = {
        "test_loss": test_loss,
        **test_metrics
    }
    with open(os.path.join(LOG_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)

    writer.close()
    logger.info("训练完成。")


if __name__ == "__main__":
    main()