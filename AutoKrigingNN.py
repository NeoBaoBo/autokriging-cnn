# 模型使用 - UI优化版
# 功能：
# 1. 结果不再弹窗显示，而是在主界面内显示
# 2. 增加“保存结果”功能，可保存为 txt 或 xlsx
# 3. 增加轴长度比值计算：
#    - 主轴 / 半主轴
#    - 主轴 / 次轴
#    - 半主轴 / 次轴
# 4. 显示结果和保存结果中均包含这些比值

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CNN模型（9参数）
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1),
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

        self.shared = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        # Variogram
        self.fc_v = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # 椭球
        self.fc_e = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared(x)

        v_raw = self.fc_v(x)
        e_raw = self.fc_e(x)

        # Variogram
        nugget = torch.sigmoid(v_raw[:, 0]) * 0.2
        sill = F.softplus(v_raw[:, 1]) * 2
        rng = F.softplus(v_raw[:, 2]) * 200

        v = torch.stack([nugget, sill, rng], dim=1)

        # 搜索椭球
        major = F.softplus(e_raw[:, 0]) * 500
        minor = F.softplus(e_raw[:, 1]) * 300
        vertical = F.softplus(e_raw[:, 2]) * 150

        azimuth = (torch.tanh(e_raw[:, 3]) + 1) * 180
        dip = torch.tanh(e_raw[:, 4]) * 90
        plunge = torch.tanh(e_raw[:, 5]) * 90

        e = torch.stack([
            major,
            minor,
            vertical,
            azimuth,
            dip,
            plunge
        ], dim=1)

        return v, e


# =========================
# 构建3D网格
# =========================
def create_grid(x, y, z, val, cell_size=10):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    val = np.array(val)

    clean_x = []
    clean_y = []
    clean_z = []
    clean_v = []

    for i in range(len(val)):
        try:
            v = float(val[i])
            if np.isnan(v):
                continue

            clean_x.append(float(x[i]))
            clean_y.append(float(y[i]))
            clean_z.append(float(z[i]))
            clean_v.append(v)
        except Exception:
            continue

    x = np.array(clean_x)
    y = np.array(clean_y)
    z = np.array(clean_z)
    val = np.array(clean_v)

    if len(val) == 0:
        raise ValueError("有效数值为空，无法构建网格。")

    xmin = x.min()
    ymin = y.min()
    zmin = z.min()

    xi = ((x - xmin) / cell_size).astype(int)
    yi = ((y - ymin) / cell_size).astype(int)
    zi = ((z - zmin) / cell_size).astype(int)

    nx = xi.max() + 1
    ny = yi.max() + 1
    nz = zi.max() + 1

    print("网格尺寸:", nx, ny, nz)

    if nx * ny * nz > 50000000:
        raise ValueError("网格太大，请增大 cell_size")

    grid = np.zeros((nx, ny, nz), dtype=np.float32)

    for i in range(len(val)):
        grid[xi[i], yi[i], zi[i]] = val[i]

    return grid


# =========================
# GUI程序
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Kriging Neural Network System")
        self.root.geometry("920x720")

        self.data_path = ""
        self.model_path = ""
        self.result_text = ""
        self.result_records = []

        self.build_ui()

    def build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=15)

        tk.Label(
            top_frame,
            text="地球化学智能反演系统",
            font=("Arial", 16, "bold")
        ).grid(row=0, column=0, columnspan=3, pady=10)

        tk.Button(
            top_frame,
            text="选择钻孔数据",
            width=18,
            command=self.load_data
        ).grid(row=1, column=0, padx=10, pady=8)

        tk.Button(
            top_frame,
            text="选择模型文件",
            width=18,
            command=self.load_model
        ).grid(row=1, column=1, padx=10, pady=8)

        tk.Button(
            top_frame,
            text="开始预测",
            width=18,
            height=2,
            bg="#4CAF50",
            fg="white",
            command=self.run_analysis
        ).grid(row=1, column=2, padx=10, pady=8)

        tool_frame = tk.Frame(self.root)
        tool_frame.pack(pady=5)

        tk.Button(
            tool_frame,
            text="保存结果",
            width=16,
            command=self.save_results
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            tool_frame,
            text="清空结果",
            width=16,
            command=self.clear_results
        ).grid(row=0, column=1, padx=10)

        self.status = tk.Label(self.root, text="系统就绪", fg="blue")
        self.status.pack(pady=5)

        info_frame = tk.Frame(self.root)
        info_frame.pack(fill="x", padx=15, pady=5)

        self.data_label = tk.Label(info_frame, text="钻孔数据：未选择", anchor="w", fg="gray")
        self.data_label.pack(fill="x")

        self.model_label = tk.Label(info_frame, text="模型文件：未选择", anchor="w", fg="gray")
        self.model_label.pack(fill="x")

        result_frame = tk.LabelFrame(self.root, text="预测结果", padx=10, pady=10)
        result_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.result_box = scrolledtext.ScrolledText(
            result_frame,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.result_box.pack(fill="both", expand=True)

    def append_result(self, text):
        self.result_box.insert(tk.END, text + "\n")
        self.result_box.see(tk.END)
        self.root.update()

    def clear_results(self):
        self.result_box.delete("1.0", tk.END)
        self.result_text = ""
        self.result_records = []
        self.status.config(text="结果已清空", fg="blue")

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])

        if path:
            self.data_path = path
            self.data_label.config(text=f"钻孔数据：{path}", fg="black")
            self.status.config(text="已加载钻孔数据", fg="blue")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth *.pt")])

        if path:
            self.model_path = path
            self.model_label.config(text=f"模型文件：{path}", fg="black")
            self.status.config(text="已加载模型", fg="blue")

    # =========================
    # 保存结果
    # =========================
    def save_results(self):
        if not self.result_text.strip():
            messagebox.showwarning("提示", "当前没有可保存的结果。")
            return

        save_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".txt",
            filetypes=[
                ("Text File", "*.txt"),
                ("Excel File", "*.xlsx")
            ]
        )

        if not save_path:
            return

        try:
            ext = os.path.splitext(save_path)[1].lower()

            if ext == ".txt":
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.result_text)

            elif ext == ".xlsx":
                if not self.result_records:
                    raise ValueError("没有可写入 Excel 的结构化结果。")
                df = pd.DataFrame(self.result_records)
                df.to_excel(save_path, index=False)

            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.result_text)

            self.status.config(text="结果保存成功", fg="green")
            messagebox.showinfo("完成", "结果保存成功。")

        except Exception as e:
            self.status.config(text="结果保存失败", fg="red")
            messagebox.showerror("错误", f"保存失败：{e}")

    # =========================
    # 运行分析（支持多元素）
    # =========================
    def run_analysis(self):
        if self.data_path == "" or self.model_path == "":
            messagebox.showerror("错误", "请先加载数据和模型")
            return

        try:
            self.status.config(text="正在分析...", fg="orange")
            self.root.update()

            self.result_box.delete("1.0", tk.END)
            self.result_text = ""
            self.result_records = []

            df = pd.read_excel(self.data_path)

            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            z = df.iloc[:, 2]

            element_cols = df.columns[3:]

            if len(element_cols) == 0:
                messagebox.showerror("错误", "未检测到元素列")
                return

            model = CNNModel()

            state_dict = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True
            )

            model.load_state_dict(state_dict, strict=False)
            model.eval()

            self.append_result("预测完成")
            self.append_result("")

            for elem in element_cols:
                val = df[elem]

                grid1 = create_grid(x, y, z, val)

                # 单元素复制为2通道
                grid = np.stack([grid1, grid1])

                grid = torch.tensor(grid).unsqueeze(0)

                with torch.no_grad():
                    v, e = model(grid)

                v = v.numpy()[0]
                e = e.numpy()[0]

                major_axis = float(e[0])
                semimajor_axis = float(e[1])   # 当前代码中的第二轴
                minor_axis = float(e[2])       # 当前代码中的第三轴

                # 比值计算
                major_to_semimajor = major_axis / semimajor_axis if semimajor_axis != 0 else np.nan
                major_to_minor = major_axis / minor_axis if minor_axis != 0 else np.nan
                semimajor_to_minor = semimajor_axis / minor_axis if minor_axis != 0 else np.nan

                block_text = f"""元素: {elem}

Variogram:
Nugget = {v[0]:.4f}
Sill   = {v[1]:.4f}
Range  = {v[2]:.2f}

Search Ellipsoid:
Major Axis      = {major_axis:.2f}
Semi-major Axis = {semimajor_axis:.2f}
Minor Axis      = {minor_axis:.2f}
Azimuth         = {e[3]:.2f}°
Dip             = {e[4]:.2f}°
Plunge          = {e[5]:.2f}°

Axis Ratios:
Major / Semi-major = {major_to_semimajor:.4f}
Major / Minor      = {major_to_minor:.4f}
Semi-major / Minor = {semimajor_to_minor:.4f}

--------------------------------
"""
                self.append_result(block_text)
                self.result_text += block_text + "\n"

                self.result_records.append({
                    "Element": elem,
                    "Nugget": float(v[0]),
                    "Sill": float(v[1]),
                    "Range": float(v[2]),
                    "Major Axis": major_axis,
                    "Semi-major Axis": semimajor_axis,
                    "Minor Axis": minor_axis,
                    "Azimuth": float(e[3]),
                    "Dip": float(e[4]),
                    "Plunge": float(e[5]),
                    "Major / Semi-major": major_to_semimajor,
                    "Major / Minor": major_to_minor,
                    "Semi-major / Minor": semimajor_to_minor
                })

            self.status.config(text="预测完成", fg="green")

        except Exception as e:
            self.status.config(text="运行失败", fg="red")
            messagebox.showerror("错误", f"运行失败：{e}")


# =========================
# 关闭程序
# =========================
def on_closing():
    root.destroy()
    sys.exit()


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()