import torch


def get_device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = {
    # ------------------------------------------------------------------
    # Dataset & Paths
    # ------------------------------------------------------------------
    "dataset_name": "xian",                # folder name under datasets/
    "base_dir": "logs",                 # where logs & weights will be saved
    "log_file": "best_model_metrics.log",

    # ------------------------------------------------------------------
    # Training Hyper‑parameters
    # ------------------------------------------------------------------
    "num_epochs": 100,
    "batch_size": 2,
    "learning_rate_backbone": 1e-4,
    "learning_rate_head": 5e-5,
    "learning_rate_style": 1e-3,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-4,

    # ------------------------------------------------------------------
    # Model & Optimisation settings
    # ------------------------------------------------------------------
    "num_classes": 2,          # 0: background, 1: foreground
    "ignore_index": 255,
    "image_size": (1024, 1024),
    "lora_rank": 256,
    "lora_alpha": 256,
    "checkpoint": "weights/sam2.1_hiera_large.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",

    # ------------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------------
    "metric_weights": {"iou": 0.25, "f1": 0.25, "precision": 0.25, "recall": 0.25},
    "aux_weight": 1.0,
    "consistency_weight": 0.0,
    "domain_discrepancy_weight": 1.0,

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    "use_ldp": True,          # set True to enable LearnableDomainPerturbation
    "save_prefix": "best_model",
    "device": get_device(),
}
