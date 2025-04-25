## Overview

Welcome to **CFP-SAM**, a PyTorch implementation of **Frequency-Inspired Contrastive Feature Perturbation for Remote Sensing Semantic Segmentation**. This repository provides a clean, modular codebase for training and evaluating remote sensing segmentation models built on top of the Segmentation Anything Model (SAM 2) foundation, enhanced via low-rank LoRA adapters and an adversarial style-perturbation branch. CFP-SAM encourages learning of **high-frequency**, domain-invariant features through a dual-branch parameter-sharing framework and a learnable style perturbation module, leading to state-of-the-art performance across multiple RSI benchmarks.

- **Frequency Principle**: Deep networks exhibit a bias to learn low-frequency content first, hindering fine-grained segmentation; CFP-SAM counteracts this by explicitly perturbing feature statistics to emphasize high-frequency components.
- **Prompt Learning in SAM**: The prompt-based training in SAM naturally promotes high-frequency feature learning, laying the groundwork for robust transfer to remote sensing domains.
- **LoRA Adapters**: We employ Low-Rank Adaptation (LoRA) to efficiently fine-tune only low-rank updates within the Transformer attention layers, drastically reducing trainable parameters while preserving pre-trained weights.
- **AdaIN-Inspired Perturbation**: Our LearnableStylePerturbation module draws on Adaptive Instance Normalization (AdaIN) to adjust feature mean and variance in an adversarial manner, expanding the feature space while maintaining semantic consistency.
- **Gram-Matrix Discrepancy**: We measure style shifts via the Frobenius norm between Gram matrices of clean vs. perturbed features, guiding adversarial maximization of domain discrepancy.


## Repository Structure

```
CFP-SAM/
├── config.py            # Hyperparameters & paths
├── dataset.py           # SegmentationDataset, data transforms
├── heads.py             # SegmentationHead & AuxiliaryClassifier
├── ldp.py               # LearnableStylePerturbation module
├── losses.py            # compute_loss, consistency & discrepancy losses
├── metrics.py           # accumulate_metrics, initialize_metrics
├── lora2.py             # LoRA_qkv_hiera & LoRA_sam2 adapter
├── train.py             # Main training loop (entry point)
└── README.md            # This document
```
## Prerequisites
**Clone and install SAM 2**  
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .                              # Install the SAM 2 package
```
**Download SAM 2 checkpoints**  
   - Place `sam2.1_hiera_large.pt` (and other desired variants) in `weights/`  
   - Ensure `sam2/configs/sam2.1/` contains the corresponding YAML files
   
## Installation

```bash
git clone https://github.com/xxx/CFP-SAM.git
cd CFP-SAM
conda create -n CFP-SAM python==3.8.16
conda activate CFP-SAM
```

**Requirements**  
- Python 3.10+  
- PyTorch 2.5+  
- torchvision 0.21.0+  
- safetensors  
- tqdm  
- Pillow  

## Usage

### Configuration

Edit `config.py` to set:
- `dataset_name`, dataset paths  
- `checkpoint`, `model_cfg` for SAM2  ([SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2))  
- Learning rates, batch size, number of epochs  
- LoRA rank & α for adapter strength
- Loss weights for consistency, discrepancy, auxiliary  

### Training

```bash
python train.py
```

This will:
1. Build SAM2 with LoRA adapters.  
2. Load datasets & create DataLoaders.  
3. Train dual-branch CFP-SAM with adversarial perturbation.  
4. Validate, compute IoU/F1/precision/recall, and checkpoint best model.

### Evaluation

After training, run:

```bash
python eval.py
```

*(Note: `eval.py` can be adapted from validation loop in `train.py`.)*

## Results

CFP-SAM outperforms baselines on RSI benchmarks, demonstrating superior high-frequency feature transfer:
- **ISPRS Vaihingen/Potsdam**: +1.2% mIoU vs. SAM-Base and SOTA methods.  
- **Building Dataset**: +2.3% mIoU margin.  
- **Xi’an Dataset**: +6.9% boost over UV-SAM.

## Citation


## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
