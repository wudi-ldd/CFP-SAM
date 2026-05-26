# CFP-SAM

PyTorch implementation of **Frequency-Inspired Contrastive Feature Perturbation for Remote Sensing Semantic Segmentation**.

This repository accompanies the corresponding manuscript currently under review.

This repository adapts **SAM 2.1 Hiera** to remote sensing semantic segmentation with:

- **LoRA** adapters on the image encoder attention layers
- a **learnable low-frequency perturbation** module applied to encoder features during training
- a **dual-branch optimization scheme** with
  - style-discrepancy maximization in the perturbation branch
  - segmentation supervision and semantic consistency in the segmentation branch

At inference time, the perturbation branch is disabled and the model runs with the SAM2 encoder, LoRA weights, and the segmentation head only.

## Method Summary

CFP-SAM is motivated by the frequency bias of deep networks. During fine-tuning, models can overfit low-frequency appearance statistics that are less transferable across remote sensing domains. CFP-SAM addresses this by perturbing low-frequency encoder feature amplitudes during training while enforcing prediction consistency between clean and perturbed branches.

The current codebase includes:

- SAM 2.1 Hiera backbone adaptation with LoRA
- multi-stage low-frequency perturbation (`MultiStageLDP`)
- segmentation head with auxiliary classifiers
- train / validation loop
- standalone evaluation script

## Repository Structure

```text
CFP-SAM/
├── config.py
├── datasets.py
├── eval.py
├── lora2.py
├── losses.py
├── train.py
├── utils.py
├── models/
│   ├── heads.py
│   ├── hooks.py
│   └── ldp.py
└── sam2/
```

## Environment

Tested with:

- Python 3.10
- PyTorch 2.5+
- torchvision 0.21+

Required packages:

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `Pillow`
- `tqdm`
- `safetensors`
- `hydra-core`
- `omegaconf`

Optional:

- `ttach` for test-time augmentation in `eval.py`

Example setup:

```bash
conda create -n cfp-sam python=3.10
conda activate cfp-sam
pip install torch torchvision
pip install numpy opencv-python pillow tqdm safetensors hydra-core omegaconf
pip install ttach
```

## Checkpoints and SAM2

This repository includes a local `sam2/` package. The default configuration uses **SAM 2.1 Hiera-Large**:

- checkpoint: `weights/sam2.1_hiera_large.pt`
- config: `configs/sam2.1/sam2.1_hiera_l.yaml`

Before training or evaluation, place the SAM2 checkpoint in:

```text
CFP-SAM/weights/sam2.1_hiera_large.pt
```

If you use another SAM2 variant, update `checkpoint` and `model_cfg` in `config.py`.

## Dataset Layout

The code expects datasets in the following layout:

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

Each split file should contain one file stem per line, for example:

```text
top_potsdam_2_10
top_potsdam_2_11
top_potsdam_2_12
```

The loader matches these entries against `images/*.png` and `masks/*.png`.

## Label Conventions

- For **binary datasets** such as Beijing / Xi'an:
  - set `num_classes = 2`
- For **ISPRS Vaihingen / Potsdam**:
  - set `num_classes = 5`
  - clutter/background pixels should be encoded as `255` so they are ignored during metric computation

Please ensure the mask encoding matches `num_classes` and `ignore_index` in `config.py`.

## Configuration

Edit `config.py` before running experiments. The most important fields are:

- `dataset_name`
- `train_split`
- `val_split`
- `num_classes`
- `ignore_index`
- `checkpoint`
- `model_cfg`
- `lora_rank`
- `lora_alpha`
- `ldp_stage_ids`
- `ldp_p`

The current file is a runnable example configuration rather than a universal default.

## Training

Run training from the `CFP-SAM` directory:

```bash
python train.py
```

Training outputs are written to:

```text
logs/<dataset_name>_<timestamp>/
├── logs/
│   └── best_model_metrics.log
├── weights/
│   ├── best_model_lora.safetensors
│   ├── best_model_lora.pth
│   ├── best_model_lora_LDP.safetensors
│   └── best_model_lora_LDP.pth
└── config_snapshot.py
```

The best checkpoint is selected according to validation IoU inside `train.py`.

## Evaluation

### Validation

By default, `eval.py` reads `CONFIG["val_split"]` unless `--test_list` is provided:

```bash
python eval.py --run_name <run_name>
```

### Final test evaluation

For final reporting, explicitly evaluate the test split:

```bash
python eval.py --run_name <run_name> --test_list test.txt
```

You can also evaluate the most recent run automatically by omitting `--run_name`.

### Optional arguments

```bash
python eval.py \
  --run_name <run_name> \
  --test_list test.txt \
  --tta_mode lr \
  --use_amp
```

Available options:

- `--run_name`: use a specific training run directory
- `--test_list`: specify the split file to evaluate
- `--tta_mode`: `lr`, `d4`, or `none`
- `--use_amp`: enable AMP during evaluation
- `--no_overlay`: disable prediction visualization export

## Evaluation Outputs

`eval.py` reports:

- per-class `IoU / Precision / Recall / F1`
- averaged summary metrics
- optional prediction visualizations

Prediction files are saved under:

```text
logs/<run_name>/predictions/
```

## Note on Binary Datasets

For **Beijing** and **Xi'an**, the manuscript reports **foreground** `IoU / F1 / Recall / Precision`.

The current evaluation script prints per-class metrics first and then prints an averaged summary. To match the paper protocol on binary datasets, use the **foreground class line** from the per-class output when recording the final paper numbers.

## Reproducibility Notes

The code fixes random seeds and uses deterministic CuDNN settings in both training and evaluation.

To avoid protocol mismatch when reproducing the manuscript results:

1. check that the split files are correct and disjoint
2. use the correct `num_classes` for the dataset
3. use explicit `--test_list test.txt` for final test reporting
4. record foreground metrics for binary datasets

## Citation

If you find this repository useful, please cite the manuscript below:

```bibtex
@misc{li2026cfpsam,
  title={Frequency-Inspired Contrastive Feature Perturbation for Remote Sensing Semantic Segmentation},
  author={Li, Dongsheng and Hao, Pingting and Guo, Xinyuan and Zhang, Huijie and Lin, Yiming and Li, Jiaxin and Gao, Tong},
  year={2026},
  note={Manuscript under review}
}
```

## License

Please add the license of your choice before public release.
