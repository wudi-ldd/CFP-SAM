import torch

CONFIG = {

    'dataset_name': 'xian',
    'base_dir': 'logs',
    'log_file': 'best_model_metrics.log',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    'num_epochs': 100,
    'learning_rate_backbone': 1e-4,
    'learning_rate_head': 1e-4,
    'learning_rate_style': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 1e-2,

    'metric_weights': {
        'iou': 0.25,
        'f1': 0.25,
        'precision': 0.25,
        'recall': 0.25
    },

    'aux_weight': 1,
    'consistency_weight': 1,
    'domain_discrepancy_weight': 1,

    'use_ldp': True,
    'num_classes': 2,
    'batch_size': 2,
    'ignore_index': 255,

    'save_prefix': 'best_model',
    'image_size': (1024, 1024),

    'lora_rank':32,
    'lora_alpha':32,

    'checkpoint': "weights/sam2.1_hiera_large.pt",
    'model_cfg': "configs/sam2.1/sam2.1_hiera_l.yaml",

    'train_split': 'train.txt',
    'val_split': 'val.txt',


    'ldp_stage_ids': [0, 1, 2, 3],
    'ldp_p': 0.5,


    'use_checkpoint': True,
    'checkpoint_use_reentrant': False,
    'checkpoint_layers': None,


    'tta_mode': None,
    'use_amp_eval': True,

    'save_overlay': True,
    'overlay_mode': 'save_separate_masks',
    'overlay_alpha': 0.2,
    'overlay_dir': 'predictions',

    'eval_batch_size': 4,
    'eval_num_workers': 8,

    'eval_classes': None,
}

SEED = 42
