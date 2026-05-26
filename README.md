# CFP-SAM

PyTorch implementation of **Frequency-Inspired Contrastive Feature Perturbation for Remote Sensing Semantic Segmentation**.

CFP-SAM adapts SAM 2.1 Hiera to remote sensing semantic segmentation using parameter-efficient fine-tuning and training-time low-frequency feature perturbation. During inference, the perturbation branch is disabled; only the SAM2 image encoder, LoRA weights, and segmentation head are used.

## Updates

The repository currently provides the main training and evaluation code. Pretrained weights, trained checkpoints, and detailed reproduction logs will be updated progressively.

## Repository Structure

```text
CFP-SAM/
├── config.py
├── datasets.py
├── train.py
├── eval.py
├── losses.py
├── lora2.py
├── utils.py
├── models/
│   ├── heads.py
│   ├── hooks.py
│   └── ldp.py
└── sam2/
```

## Environment

Recommended environment:

```text
Python >= 3.10
PyTorch >= 2.5
torchvision >= 0.20
```

Install common dependencies:

```bash
pip install torch torchvision
pip install numpy opencv-python pillow tqdm safetensors hydra-core omegaconf
```


## SAM2 Checkpoint

The default configuration uses **SAM 2.1 Hiera-Large**.

Place the SAM2 checkpoint at:

```text
weights/sam2.1_hiera_large.pt
```

The corresponding SAM2 config path is set in `config.py`, for example:

```python
checkpoint = "weights/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

## Dataset Format

Prepare each dataset as:

```text
datasets/
└── <dataset_name>/
    ├── images/
    │   ├── xxx.png
    │   └── ...
    ├── masks/
    │   ├── xxx.png
    │   └── ...
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each split file contains one sample name per line. The image and mask files should share the same stem.

Example:

```text
top_potsdam_2_10
top_potsdam_2_11
top_potsdam_2_12
```

## Label Format

For binary urban-village datasets:

```python
num_classes = 2
ignore_index = 255
```

For ISPRS Vaihingen and Potsdam, clutter/background pixels are ignored during evaluation:

```python
num_classes = 5
ignore_index = 255
```

Make sure the mask encoding is consistent with `num_classes` and `ignore_index`.

## Configuration

Before training or evaluation, edit `config.py`.

Key fields include:

```python
dataset_name
train_split
val_split
num_classes
ignore_index
image_size
checkpoint
model_cfg
lora_rank
lora_alpha
ldp_stage_ids
ldp_p
```

## Training

Run:

```bash
python train.py
```

Outputs are saved to:

```text
logs/<dataset_name>_<timestamp>/
├── logs/
├── weights/
└── config_snapshot.py
```

The main saved files are:

```text
best_model_lora.safetensors
best_model_lora.pth
best_model_lora_LDP.safetensors
best_model_lora_LDP.pth
```

The `_LDP` suffix is used when the low-frequency perturbation module is enabled.

## Evaluation

Evaluate a specific run:

```bash
python eval.py --run_name <run_name> --test_list test.txt
```

Available arguments:

```text
--run_name     training run directory under logs/
--test_list    split file, e.g., test.txt
--use_amp      enable AMP during evaluation
```

Evaluation results are written to:

```text
logs/<run_name>/logs/eval_metrics.log
logs/<run_name>/logs/eval_results.json
```

## Metrics

For ISPRS Vaihingen and Potsdam, the evaluation reports:

```text
per-class IoU / Precision / Recall / F1
mIoU
mF1
OA
```

For Beijing and Xi'an urban-village datasets, the paper reports foreground-class:

```text
IoU
F1
Recall
Precision
```

Use the foreground class line in the evaluation output for binary datasets.

## Citation

```bibtex
@misc{li2026cfpsam,
  title={Frequency-Inspired Contrastive Feature Perturbation for Remote Sensing Semantic Segmentation},
  author={Li, Dongsheng and Hao, Pingting and Guo, Xinyuan and Zhang, Huijie and Lin, Yiming and Li, Jiaxin and Gao, Tong},
  year={2026},
  note={Manuscript under review}
}
```

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
