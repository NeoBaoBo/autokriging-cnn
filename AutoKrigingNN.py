"""
AutoKriging-CNN: Inference script for automated Kriging parameter prediction.
Loads the trained V3 model and predicts 9 Kriging parameters from drillhole data.

Usage
-----
CLI (recommended for batch processing):
    python AutoKrigingNN.py --model best_model.pth --data drillholes.xlsx --out results/
    python AutoKrigingNN.py --model best_model.pth --data-dir data/ --out results/
GUI:
    python AutoKrigingNN.py

Model Architecture
------------------
Single-channel 3D CNN:
  Input: 32x32x32 voxel grid (1 channel)
  Conv3d blocks: 16 -> 32 -> 64 -> 128, each with BN + ReLU
  Pool: MaxPool3d(2) after first 3 blocks; AdaptiveAvgPool3d(2,2,2) at end
  FC: 1024 -> 256 -> Dropout(0.25)
  Heads: Variogram (256->64->3) + Ellipsoid (256->64->6)
  Output: all 9 parameters via sigmoid scale-to-range mapping
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

# ============================================================
# Parameter bounds (V3 training ranges)
# ============================================================
NUGGET_MIN, NUGGET_MAX = 0.0001, 0.3
SILL_MIN, SILL_MAX = 0.001, 1.8
RANGE_MIN, RANGE_MAX = 5.0, 150.0
MAJOR_MIN, MAJOR_MAX = 10.0, 500.0
MINOR_MIN, MINOR_MAX = 5.0, 300.0
VERTICAL_MIN, VERTICAL_MAX = 2.0, 150.0
AZIMUTH_MIN, AZIMUTH_MAX = 0.0, 360.0
DIP_MIN, DIP_MAX = -90.0, 90.0
PLUNGE_MIN, PLUNGE_MAX = -90.0, 90.0


# ============================================================
# Model (V3: 1-channel, 9-output, all sigmoid)
# ============================================================
class AutoKrigingCNN_V3(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        self.fc = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.25))
        self.fc_v = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 3))
        self.fc_e = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 6))

    @staticmethod
    def scale_to_range(x, min_val, max_val):
        return min_val + (max_val - min_val) * torch.sigmoid(x)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Variogram
        pred_nugget = self.scale_to_range(self.fc_v(x)[:, 0], NUGGET_MIN, NUGGET_MAX)
        pred_sill   = self.scale_to_range(self.fc_v(x)[:, 1], SILL_MIN, SILL_MAX)
        pred_range  = self.scale_to_range(self.fc_v(x)[:, 2], RANGE_MIN, RANGE_MAX)
        pred_nugget = torch.minimum(pred_nugget, pred_sill)
        v_out = torch.stack([pred_nugget, pred_sill, pred_range], dim=1)
        # Ellipsoid axes: sort descending after sigmoid
        e = self.fc_e(x)
        a1 = self.scale_to_range(e[:, 0], MAJOR_MIN, MAJOR_MAX)
        a2 = self.scale_to_range(e[:, 1], MINOR_MIN, MINOR_MAX)
        a3 = self.scale_to_range(e[:, 2], VERTICAL_MIN, VERTICAL_MAX)
        axes, _ = torch.sort(torch.stack([a1, a2, a3], dim=1), dim=1, descending=True)
        az  = self.scale_to_range(e[:, 3], AZIMUTH_MIN, AZIMUTH_MAX)
        dip = self.scale_to_range(e[:, 4], DIP_MIN, DIP_MAX)
        plg = self.scale_to_range(e[:, 5], PLUNGE_MIN, PLUNGE_MAX)
        e_out = torch.stack([axes[:, 0], axes[:, 1], axes[:, 2], az, dip, plg], dim=1)
        return v_out, e_out


def load_model(model_path, device="cpu"):
    """Load with strict=True to guarantee architecture match."""
    model = AutoKrigingCNN_V3(in_channels=1)
    sd = torch.load(model_path, map_location=device, weights_only=True)
    result = model.load_state_dict(sd, strict=True)
    assert len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0, \
        f"Model mismatch! missing={result.missing_keys}, unexpected={result.unexpected_keys}"
    model.to(device).eval()
    return model


# ============================================================
# Voxelization (IDW with cKDTree acceleration)
# ============================================================
def voxelize_idw(x, y, z, val, target_shape=(32, 32, 32), k=8, power=2.0):
    x, y, z, val = map(np.asarray, [x, y, z, val], [np.float32]*4)
    mask = ~np.isnan(val)
    x, y, z, val = x[mask], y[mask], z[mask], val[mask]
    if len(val) == 0:
        raise ValueError("No valid data points")
    # Normalize coordinates to [0, target-1]
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9) * (target_shape[2] - 1)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-9) * (target_shape[1] - 1)
    zn = (z - z.min()) / (z.max() - z.min() + 1e-9) * (target_shape[0] - 1)
    pts = np.stack([zn, yn, xn], axis=1)
    gz, gy, gx = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]),
                             np.arange(target_shape[2]), indexing='ij')
    grid_pts = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(pts)
    dist, idx = tree.query(grid_pts, k=k)
    if k == 1:
        dist = dist.reshape(-1, 1); idx = idx.reshape(-1, 1)
    d_safe = np.where(dist < 1e-6, 1e-6, dist)
    weights = 1.0 / (d_safe ** power)
    vals_k = val[idx]
    weighted = (weights * vals_k).sum(axis=1) / weights.sum(axis=1)
    return weighted.reshape(target_shape).astype(np.float32)


def normalize_field(field):
    """Z-score normalization on non-zero voxels."""
    nonzero = field[field != 0]
    if len(nonzero) == 0:
        return field
    mean, std = nonzero.mean(), nonzero.std()
    out = (field - mean) / std if std > 1e-8 else field - mean
    out[field == 0] = 0.0
    return out.astype(np.float32)


# ============================================================
# Inference
# ============================================================
def predict(model, data_path, device="cpu"):
    df = pd.read_excel(data_path)
    x, y, z = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values
    element_cols = df.columns[3:].tolist()
    results = {}
    for elem in element_cols:
        val = pd.to_numeric(df[elem], errors='coerce').values
        field = voxelize_idw(x, y, z, val, target_shape=(32, 32, 32))
        field = normalize_field(field)
        x_tensor = torch.from_numpy(field).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            v, e = model(x_tensor.to(device))
        v, e = v[0].cpu().numpy(), e[0].cpu().numpy()
        results[elem] = {
            'nugget': float(v[0]), 'sill': float(v[1]), 'range': float(v[2]),
            'major': float(e[0]), 'semi_major': float(e[1]), 'minor': float(e[2]),
            'azimuth': float(e[3]), 'dip': float(e[4]), 'plunge': float(e[5]),
        }
    return results


# ============================================================
# CLI
# ============================================================
def main_cli():
    ap = argparse.ArgumentParser(
        description="AutoKriging-CNN V3: Kriging parameter prediction from drillhole data")
    ap.add_argument('--model', required=True, help='Path to trained .pth model')
    ap.add_argument('--data', help='Single drillhole .xlsx file')
    ap.add_argument('--data-dir', help='Directory of .xlsx files (batch)')
    ap.add_argument('--out', default='results', help='Output directory')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] Loading model: {args.model}")
    model = load_model(args.model, args.device)
    print(f"[OK] Model loaded (strict=True) on {args.device}")

    files = []
    if args.data:
        files.append(args.data)
    if args.data_dir:
        files.extend(os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
                     if f.lower().endswith(('.xlsx', '.xls')))
    if not files:
        print("[ERROR] Use --data or --data-dir to specify input files")
        return

    all_records = []
    for fp in files:
        basename = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n[INFO] {fp}")
        try:
            res = predict(model, fp, device=args.device)
        except Exception as e:
            print(f"  [FAIL] {e}")
            continue
        for elem, p in res.items():
            tag = f"{basename}_{elem}"
            print(f"  {tag}:  nugget={p['nugget']:.4f}  sill={p['sill']:.4f}  "
                  f"range={p['range']:.1f}  az={p['azimuth']:.1f}")
            all_records.append({'dataset': basename, 'element': elem, **p})
            pd.DataFrame([p]).to_excel(os.path.join(args.out, f"{tag}.xlsx"), index=False)

    if all_records:
        summary = os.path.join(args.out, "summary.xlsx")
        pd.DataFrame(all_records).to_excel(summary, index=False)
        with open(os.path.join(args.out, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Summary saved to {summary}")


# ============================================================
# GUI (optional, launched when no CLI arguments)
# ============================================================
def main_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    class App:
        def __init__(self, root):
            self.root = root
            self.root.title("AutoKriging-CNN V3")
            self.root.geometry("850x620")
            self.model_path = tk.StringVar()
            self.data_path = tk.StringVar()
            self._build()

        def _build(self):
            tk.Label(self.root, text="AutoKriging-CNN V3", font=("Arial", 14, "bold")).pack(pady=10)
            f = tk.Frame(self.root); f.pack(pady=10)
            tk.Button(f, text="Load Model", width=14, command=self._load_model).grid(row=0, column=0, padx=5)
            tk.Entry(f, textvariable=self.model_path, width=60).grid(row=0, column=1, padx=5)
            tk.Button(f, text="Load Data", width=14, command=self._load_data).grid(row=1, column=0, padx=5, pady=5)
            tk.Entry(f, textvariable=self.data_path, width=60).grid(row=1, column=1, padx=5, pady=5)
            tk.Button(self.root, text="Predict", width=20, height=2, bg="#4CAF50", fg="white",
                     command=self._run).pack(pady=5)
            self.output = scrolledtext.ScrolledText(self.root, font=("Consolas", 10), wrap=tk.WORD)
            self.output.pack(fill="both", expand=True, padx=15, pady=10)

        def _load_model(self):
            p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth *.pt")])
            if p: self.model_path.set(p)

        def _load_data(self):
            p = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
            if p: self.data_path.set(p)

        def _run(self):
            if not self.model_path.get() or not self.data_path.get():
                messagebox.showerror("Error", "Please load model and data files")
                return
            try:
                self.output.delete("1.0", tk.END)
                model = load_model(self.model_path.get(), "cpu")
                results = predict(model, self.data_path.get(), device="cpu")
                for elem, p in results.items():
                    self.output.insert(tk.END,
                        f"Element: {elem}\n"
                        f"  Variogram:  Nugget={p['nugget']:.4f}  Sill={p['sill']:.4f}  Range={p['range']:.1f} m\n"
                        f"  Ellipsoid:  Major={p['major']:.1f}  S-maj={p['semi_major']:.1f}  "
                        f"Minor={p['minor']:.1f} m\n"
                        f"              Azimuth={p['azimuth']:.1f}°  Dip={p['dip']:.1f}°  "
                        f"Plunge={p['plunge']:.1f}°\n"
                        f"{'─'*55}\n")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    tk.Tk()
    App(tk.Tk())
    tk.mainloop()  # Note: tk.Tk() is called twice; this keeps compatibility


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        main_gui()
