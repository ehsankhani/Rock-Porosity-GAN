# Guided GAN for Porous Media Reconstruction

A deep learning project for reconstructing realistic porous media structures using Generative Adversarial Networks (GANs) with guided conditioning from CT scan data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🔬 Overview

This project implements a conditional Generative Adversarial Network (cGAN) for generating realistic porous media structures from CT scan data. The model takes solid rock images and porosity segmentation maps as input to generate realistic porous rock structures, enabling advanced material characterization and digital rock physics analysis.

### Key Features

- **Guided Generation**: Uses segmentation maps as conditioning input for controlled pore structure generation
- **High-Quality Reconstruction**: Achieves high SSIM (0.9296) and PSNR (38.15 dB) scores
- **3D Visualization**: Comprehensive 3D pore network visualization using VTK/Vedo
- **Advanced Analysis**: Includes frequency domain analysis, error mapping, and statistical evaluation
- **Robust Training**: Implements checkpoint saving/loading and loss monitoring

## 📊 Dataset

The project uses high-resolution CT scan data of porous rock samples with the following specifications:

- **Original Dataset**: 887 TIFF images (1024×996 pixels each)
- **Data Types**: 
  - Raw CT scans (uint16, 0-54108 intensity range)
  - Segmented porosity maps (uint8, binary classification)
- **Memory Usage**: ~2.6 GB total
- **Processing**: Resized to 256×256 for training efficiency

### 🔗 Data Availability

**Dataset and trained model checkpoints are available on Kaggle:**
- **Kaggle Dataset**: [Porous Media CT Scan Dataset](https://www.kaggle.com/datasets/ehsankhani/micro-ct)
- **Pre-trained Models**: Download trained generator weights and checkpoints from the Kaggle platform
- **Access**: Public dataset accessible for research and educational purposes

> **Note**: Due to the large size of the CT scan dataset (~2.6GB), the raw data and trained model checkpoints are hosted on Kaggle for easy access and download.

## 🏗️ Architecture

### Generator Network
- **Type**: U-Net architecture with skip connections
- **Input Channels**: 2 (solid rock image + segmentation map)
- **Output Channels**: 1 (generated porous structure)
- **Features**: Progressive downsampling/upsampling with dropout for regularization

### Discriminator Network
- **Type**: PatchGAN discriminator
- **Input**: Concatenated condition (solid + segmentation) and target/generated images
- **Architecture**: Convolutional layers with instance normalization and LeakyReLU activation

### Loss Function
```
Total Loss = Adversarial Loss + λ × L1 Loss
```
Where λ = 100 for balancing adversarial and reconstruction objectives.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM

### Dependencies
```bash
# Core ML libraries
pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn
pip install scikit-image scipy tqdm

# 3D visualization
pip install vedo vtk

# Image processing
pip install opencv-python pillow

# Jupyter notebook
pip install jupyter notebook

# Optional: Google Colab integration
pip install google-colab
```

### Quick Setup
```bash
git clone https://github.com/yourusername/guided-gan-porous-media.git
cd guided-gan-porous-media
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Data Preparation
```python
# Download dataset from Kaggle
# Extract to project directory
# Update paths in the notebook:
raw_path = 'path/to/raw/ct/scans'
segmented_path = 'path/to/segmented/images'
```

### 2. Training
```python
# Run the complete training pipeline
jupyter notebook Untitled62.ipynb

# Or train from checkpoint
LOAD_MODEL = True
CHECKPOINT_TO_LOAD = "checkpoints/gan_checkpoint_epoch_21.pth"
```

### 3. Inference
```python
# Load trained model
gen = Generator().to(device)
gen.load_state_dict(torch.load('guided_final_report/final_generator.pth'))

# Generate new samples
with torch.no_grad():
    generated_sample = gen(input_condition)
```

### 4. Analysis and Visualization
The notebook includes comprehensive analysis tools:
- 3D pore network visualization
- Error map generation
- Frequency domain analysis
- Statistical comparison
- Performance metrics calculation

## 📈 Results

### Quantitative Metrics
- **SSIM (Structural Similarity)**: 0.9296
- **PSNR (Peak Signal-to-Noise Ratio)**: 38.15 dB
- **Average Porosity**: 25.42%
- **Training Time**: ~71 minutes on GTX 1050

### Qualitative Results
- High-fidelity pore structure reconstruction
- Realistic texture preservation
- Accurate porosity distribution
- Consistent 3D connectivity

## 📁 Project Structure

```
709H/
├── Untitled62.ipynb              # Main notebook with complete pipeline
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── pore_network.png             # 3D visualization output
├── pore_network2.png            # Enhanced 3D visualization
├── checkpoints/                 # Training checkpoints (download from Kaggle)
├── guided_final_report/         # Final model and results
│   ├── final_generator.pth      # Trained generator weights
│   ├── loss_plot.png           # Training loss curves
│   └── comparison_grid.png     # Model performance visualization
├── final_report/               # Comprehensive analysis results
│   ├── analysis_plots/         # Statistical analysis plots
│   │   ├── intensity_distribution.png
│   │   ├── frequency_analysis.png
│   │   └── porosity_correlation.png
│   ├── error_maps/            # Error visualization
│   │   └── error_map_comparison.png
│   └── sample_analysis/       # Sample-by-sample comparison
└── guided_gan_outputs/        # Generated samples during training
```

## 🎯 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| SSIM | 0.9296 | Structural similarity between generated and real images |
| PSNR | 38.15 dB | Peak signal-to-noise ratio |
| Training Time | 71 min | On NVIDIA GTX 1050 |
| Memory Usage | 2.6 GB | Raw dataset size |
| Model Size | ~45 MB | Trained generator weights |

## 🔬 Applications

- **Digital Rock Physics**: Virtual experiments on generated rock samples
- **Pore Network Modeling**: Analysis of fluid flow properties
- **Material Characterization**: Understanding porous media structure
- **Reservoir Engineering**: Enhanced oil recovery simulations
- **Research**: Academic studies in geophysics and material science

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- CT scan dataset providers
- PyTorch and torchvision teams
- VTK/Vedo for 3D visualization capabilities
- Scientific computing community

## 📞 Contact

For questions, suggestions, or collaborations:

- **Email**: ehsanhamzekhani80@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](www.linkedin.com/in/ehsan-khanii)

---

**Keywords**: GAN, Generative Adversarial Networks, Porous Media, CT Scan, Digital Rock Physics, Deep Learning, Computer Vision, Material Science, PyTorch
