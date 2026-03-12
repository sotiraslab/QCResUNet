# QCResUNet

This repository implements **QCResUNet: Joint Subject-level and Voxel-level Prediction of Segmentation Quality**, a deep learning framework for quality control in medical image segmentation [[MICCAI]](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_17) [[MedIA]](https://www.sciencedirect.com/science/article/pii/S1361841525002658).

## 📖 Overview
<img src="images\network.jpg">

QCResUNet is designed to predict segmentation quality at both subject-level and voxel-level, providing comprehensive quality assessment for medical image segmentation tasks. The model combines ResNet-based feature extraction with U-Net architecture for simultaneous quality prediction and segmentation refinement.

## 🏗️ Architecture

### Network Design
QCResUNet features a sophisticated architecture that includes:

- **ResNet Encoder**: Extracts hierarchical features using ResNet34 or ResNet50 backbones
- **U-Net Decoder**: Reconstructs spatial information through skip connections
- **Quality Prediction Head**: Predicts segmentation quality scores
- **Segmentation Head**: Refines segmentation masks
- **ECA Attention**: Enhanced Channel Attention modules for feature refinement

### Key Components

1. **Input Processing**: Accepts 5-channel input (4 modalities + segmentation mask)
2. **Feature Extraction**: Multi-scale feature extraction via ResNet backbone
3. **Quality Assessment**: Joint prediction of subject-level and voxel-level quality
4. **Attention Mechanisms**: ECA modules for improved feature representation

## 📊 Dataset

The model is designed for medical image segmentation quality assessment, particularly tested on:
- **BraTS dataset** (Brain Tumor Segmentation)
- Multi-modal MRI data (T1, T1ce, T2, FLAIR)
- Ground truth and predicted segmentation masks

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)
- Required packages: `torch`, `numpy`, `matplotlib`, `scikit-learn`, `batchgenerators`

### Setup

```bash
# Clone the repository
git clone https://github.com/sotiraslab/QCResUNet.git
cd QCResUNet

# Install dependencies
pip install -r requirements.txt
```

## 🏋️ Training
```bash
python main.py --dataroot /path/to/dataset.csv \
               --save_dir /path/to/output \
               --arch qcresunet50 \
               --loss_func MAE \
               --optimizer Adam \
               --lr 2e-4 \
               --weight_decay 1e-4 \
               --batch_size 8 \
               --max_epochs 150 \
               --fold 1 \
               --lmd 2.0 \
               --use_pearson_loss \
               --pearson_loss_weight 1.0 \
               --fp16 \
               --compile_model
```

### Training Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataroot` | Path to dataset CSV file | Required |
| `--save_dir` | Output directory for models/logs | Required |
| `--arch` | Network architecture | `qcresunet34` |
| `--batch_size` | Training batch size | 4 |
| `--max_epochs` | Maximum training epochs | 100 |
| `--fold` | Cross-validation fold | 0 |
| `--lr` | Learning rate | 2e-4 |
| `--optimizer` | Optimizer type | `Adam` |
| `--loss_func` | Loss function | `MAE` |
| `--lmd` | Segmentation loss weight | 2.0 |
| `--fp16` | Enable mixed precision training | False |
| `--compile_model` | Compile model for speed | False |

### Supported Architectures

- `qcresunet34`: QCResUNet with ResNet34 backbone
- `qcresunet50`: QCResUNet with ResNet50 backbone
- `resunet34`, `resunet50`: Standard ResUNet variants
- `resunet34_deeplabv3`, `resunet50_deeplabv3`: DeepLabV3 variants
- `resunet34_attn`, `resunet50_attn`: Attention variants

## 🔬 Model Architecture Details

### QCResUNet34/QCResUNet50

```python
class QCResUNet(nn.Module):
    def __init__(self, 
                 network_depth,      # 34 or 50
                 num_input_channels, # 5 (4 modalities + seg mask)
                 n_classes,          # 1 (quality score)
                 out_channels,       # 1 (segmentation output)
                 upsample=UpsampleBilinear,
                 skip=True,
                 drop_path=0.0,
                 dropout=0.0,
                 do_reg=True,        # Quality prediction
                 do_ds=True):        # Segmentation
```


## 📄 Citation

If you use this code in your research, please cite the associated papers:

```
@inproceedings{han2024non,
  title={Non-adversarial learning: vector-quantized common latent space for multi-sequence MRI},
  author={Han, Luyi and Tan, Tao and Zhang, Tianyu and Wang, Xin and Gao, Yuan and Lu, Chunyao and Liang, Xinglong and Dou, Haoran and Huang, Yunzhi and Mann, Ritse},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={481--491},
  year={2024},
  organization={Springer}
}
@article{qiu2025qcresunet,
  title={QCResUNet: Joint subject-level and voxel-level segmentation quality prediction},
  author={Qiu, Peijie and Chakrabarty, Satrajit and Nguyen, Phuc and Ghosh, Soumyendu Sekhar and Sotiras, Aristeidis},
  journal={Medical Image Analysis},
  pages={103718},
  year={2025},
  publisher={Elsevier}
}
```
