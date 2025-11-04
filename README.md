# HybriMamba: Mamba-based Face Super-Resolution of Efficient Global Feature Modeling and High-Frequency Information Recovery

> **Abstract**：Face super-resolution (FSR) aims to restore clear and realistic high-quality face images from low-quality ones, which demands simultaneous preservation of facial symmetry, global proportion and pore-level detail. CNN-based methods suffer from limited receptive fields, leading to geometric distortion and over-smoothed skin, while Transformers introduce quadratic complexity due to self-attention that hinders high-resolution inference. The recently proposed Mamba achieves linear complexity, yet its native 1-D scanning breaks 2-D facial adjacency and lacks frequency-aware degradation modeling, easily yielding asymmetric features and missing high-frequency details. To address these problems, we propose HybriMamba, a linear-complexity face super-resolution framework that equips Mamba with an Image-Selective Scan Module (Image-SSM) that restores pixel adjacency and enforces horizontal facial symmetry, a Fourier–Wavelet Transform Module (FWM) that globally modulates magnitude spectra and directionally processes wavelet sub-bands to recover fine facial details, and a Local Enhancement Module (LEM) with pixel-wise gating that adaptively re-weights smooth and textured regions to avoid over-smoothing. Embedded in a U-shaped multi-scale encoder–decoder, HybriMamba hierarchically retains facial geometry while injecting micro-textures through skip connections. Extensive experiments on Helen and CelebA show that HybriMamba achieves the best PSNR, SSIM and LPIPS among state-of-the-art methods, delivering identity-faithful and visually pleasing results.

## 🏗️ Network Architecture




## 🚀 Quick Start
```bash
git clone https://github.com/LateAutu/HybriMamba.git
cd HybriMamba
pip install -r requirements.txt

## 📦 Installation
```bash
# 1. 创建虚拟环境（可选）
conda create -n hybridmamba python=3.9
conda activate hybridmamba

# 2. 安装依赖
pip install -r requirements.txt


## 📦 Installation
```bash
# 1. 创建虚拟环境（可选）
conda create -n hybridmamba python=3.9
conda activate hybridmamba

# 2. 安装依赖
pip install -r requirements.txt
```


## 🏋️ Training
1. 下载 [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 原图，**无需预对齐**。
2. 修改脚本路径与实验名：
```bash
bash train.sh \
  --dataroot </path/to/CelebA> \
  --name <exp_name>        \
  --batch_size 32          \
  --gpus 2
```

| 参数 | 说明 |
|------|------|
| `--dataroot` | CelebA 图片根目录 |
| `--name` | 实验名，tensorboard & 权重均以此命名 |
| `--batch_size` | 显存不足时可调小 |
| `--gpus` | 使用 GPU 数量；需指定卡号请取消脚本内 `export CUDA_VISIBLE_DEVICES=` 注释 |

日志与权重保存结构：
```
checkpoints/
├── <exp_name>/
│   ├── latest.pth
│   └── events.out.tfevents.*
└── log_archive/   # 旧日志自动迁移
```

## 🧪 Testing
```bash
bash test.sh \
  --dataroot </path/to/CelebA> \
  --name <exp_name>
```
结果自动写入 `results/<exp_name>/`。

## 📈 Results
### 定量对比（8× & 16× SR）
| Method | Scale | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| Bicubic| 8×    | 24.15 | 0.712 | 0.195  |
| ESRGAN | 8×    | 26.22 | 0.791 | 0.142  |
| **HybriMamba** | 8× | **27.34** | **0.823** | **0.108** |

### 可视化
| LR (32×32) | HybriMamba | GT |
|:----------:|:----------:|:--:|
| ![lr](./assets/lr.png) | ![sr](./assets/sr.png) | ![gt](./assets/gt.png) |

## 🛠️ Code Structure
```
HybriMamba/
├── train.sh              # 训练入口
├── test.sh               # 测试入口
├── requirements.txt      # 依赖
├── hybridmamba/
│   ├── models/           # 网络定义
│   ├── data/             # 数据加载
│   └── utils/            # 工具函数
└── checkpoints/          # 权重保存（gitignore）
```

## 📜 Citation
```bibtex
@misc{hybridmamba2025,
  title={HybriMamba: Linear-Complexity Hybrid State-Space Models for Ultra-Low-Resolution Face Super-Resolution},
  author={Your Name and Co-Authors},
  year={2025},
  eprint={arXiv:****.*****},
  url={https://github.com/<LateAutu>/<HybriMamba>}
}
```

## 📄 License
[Apache-2.0](LICENSE) © 2025 HybriMamba Authors

---


