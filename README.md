# Medical Image Segmentation with Multimodal Learning
## Technical Documentation

![WhatsApp Image 2025-05-07 at 21 13 34](https://github.com/user-attachments/assets/55544941-d2d1-468d-8ad9-7fc7d4bcfbcc)

![WhatsApp Image 2025-05-07 at 21 22 21](https://github.com/user-attachments/assets/2ff5fd95-bbef-4670-9dec-5b0c75d64b80)
![Screenshot 2025-04-25 at 11 18 23 AM](https://github.com/user-attachments/assets/a30d9ae2-def7-4ae5-910a-3e5616be34eb)

Data Efficiency Analysis:
  Maintained 83% performance with 90% of the data
  Effective even in low-data scenarios
Robustness Analysis:
  Handles anatomical variability
  Adapts to complex structures




![Screenshot 2025-04-25 at 11 14 51 AM](https://github.com/user-attachments/assets/adf593f5-b17e-4023-8a4d-d0709a57e1fe)

![Screenshot 2025-04-11 at 1 43 15 PM](https://github.com/user-attachments/assets/f9526a71-992a-4edf-a19e-d4b8d3b1d7b1)


### 1. Implementation Overview

#### 1.1 Core Technologies
- **Deep Learning Framework**: PyTorch 2.0.1
- **Programming Language**: Python 3.9
- **Primary Libraries**:
  - MONAI (Medical Open Network for AI)
  - Transformers (Hugging Face)
  - Llama-2 (Meta AI)
  - TensorBoardX (for visualization)


<img width="521" alt="pipeline" src="https://github.com/user-attachments/assets/2e3be146-f62a-4a38-8311-8514deddec68" />



#### 1.2 System Requirements
- **Operating System**: Linux (Ubuntu recommended) / Windows with WSL
- **GPU**: NVIDIA GPU with CUDA support (45GB+ VRAM recommended)

- **RAM**: 64GB+ recommended
- **Storage**: 100GB+ free space

### 2. Installation Guide

#### 2.1 System Dependencies
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.9 python3.9-distutils git-lfs
```

#### 2.2 Python Environment Setup
```bash
# Install pip for Python 3.9
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Create and activate virtual environment (recommended)
python3.9 -m venv venv
source venv/bin/activate  # Linux
# or
.\venv\Scripts\activate  # Windows
```

#### 2.3 Package Installation
```bash
# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
```

### 3. Project Structure

```
project/
├── main.py              # Entry point for training and inference
├── trainer.py           # Training and evaluation logic
├── model/              
│   ├── llama2/         # Llama-2 model integration
│   └── segmentation/   # Segmentation model architecture
├── utils/              
│   ├── data_utils.py   # Data processing utilities
│   └── metrics.py      # Evaluation metrics
├── dataset/            
│   └── external1/      # Dataset storage
├── optimizers/         # Custom optimizer implementations
├── ckpt/               # Model checkpoints
└── requirements.txt    # Project dependencies
```

### 4. Dataset Preparation

#### 4.1 Data Format
- Input: 3D medical images in NPZ format
- Labels: Segmentation masks in NPZ format
- Text Reports: Excel/CSV format

#### 4.2 Directory Structure
```
dataset/
└── external1/
    └── [case_id]/
        ├── data.npz    # 3D medical image
        └── label.npz   # Segmentation mask
```

### 5. Model Architecture

#### 5.1 Components
1. **Image Encoder**: 3D UNet-based architecture
2. **Text Encoder**: Llama-2 7B model
3. **Multimodal Fusion**: Custom attention mechanism

#### 5.2 Key Features
- 3D medical image processing
- Text report integration
- Multi-scale feature extraction
- Attention-based fusion

### 6. Training Process

#### 6.1 Training Command
```bash
python main.py \
    --pretrained_dir ./ckpt/multimodal \
    --context True \
    --n_prompts 2 \
    --context_length 8 \
    --batch_size 1 \
    --roi_x 64 \
    --roi_y 352 \
    --roi_z 352
```

#### 6.2 Training Parameters
- Batch Size: 1 (adjustable based on GPU memory)
- Learning Rate: 1e-4
- Optimizer: AdamW
- Loss Function: DiceCELoss
- Mixed Precision: Enabled

### 7. Inference

#### 7.1 Inference Command
```bash
python main.py \
    --pretrained_dir ./ckpt/multimodal \
    --context True \
    --n_prompts 2 \
    --context_length 8 \
    --test_mode 2
```

#### 7.2 Output Format
- Segmentation masks in NIfTI format
- Evaluation metrics in CSV format
- TensorBoard logs for visualization

### 8. Performance Considerations

#### 8.1 Memory Requirements
- GPU Memory: 45GB+ recommended
- RAM: 64GB+ recommended
- Storage: 100GB+ for dataset and models

#### 8.2 Optimization Tips
1. Adjust batch size based on available GPU memory
2. Use gradient accumulation for larger effective batch sizes
3. Enable mixed precision training
4. Utilize data prefetching

### 9. Troubleshooting

#### 9.1 Common Issues
1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Installation Errors**
   - Ensure correct Python version (3.9)
   - Install CUDA toolkit matching PyTorch version
   - Use virtual environment

3. **Data Loading Issues**
   - Verify NPZ file format
   - Check file permissions
   - Validate data directory structure





