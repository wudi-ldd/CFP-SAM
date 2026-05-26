import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import safe_open, save_file


class LoRA_qkv_hiera(nn.Module):

    def __init__(self, qkv, dim, dim_out, num_heads, rank, alpha=1.0, dropout_p=0.1):
        super().__init__()
        self.qkv = qkv
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        self.lora_A_q = nn.Linear(dim, rank, bias=False)
        self.lora_A_v = nn.Linear(dim, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, dim_out, bias=False)
        self.lora_B_v = nn.Linear(rank, dim_out, bias=False)

        self.dropout_q = nn.Dropout(p=dropout_p)
        self.dropout_v = nn.Dropout(p=dropout_p)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):

        qkv = self.qkv(x)


        lora_q = self.lora_B_q(self.dropout_q(self.lora_A_q(x))) * self.scaling
        lora_v = self.lora_B_v(self.dropout_v(self.lora_A_v(x))) * self.scaling


        orig_shape = qkv.shape
        qkv_ = qkv.view(*orig_shape[:-1], 3, self.dim_out)

        qkv_[..., 0, :] = qkv_[..., 0, :] + lora_q
        qkv_[..., 2, :] = qkv_[..., 2, :] + lora_v

        return qkv_.view(*orig_shape)


class LoRA_sam2(nn.Module):
    def __init__(self, sam_model, rank: int, lora_layer=None, alpha=1.0):
        super(LoRA_sam2, self).__init__()
        self.rank = rank
        assert rank > 0

        self.sam_model = sam_model
        self.backbone = sam_model.image_encoder.trunk
        self.blocks = self.backbone.blocks

        if lora_layer:
            self.lora_layer = lora_layer
            print(f"LoRA applied to layers: {lora_layer}")
        else:
            self.lora_layer = list(range(len(self.blocks)))

        self.A_weights_q = nn.ModuleList()
        self.B_weights_q = nn.ModuleList()
        self.A_weights_v = nn.ModuleList()
        self.B_weights_v = nn.ModuleList()


        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            attn = blk.attn
            qkv_linear = attn.qkv

            dim = attn.dim
            dim_out = attn.dim_out
            num_heads = attn.num_heads

            lora_qkv = LoRA_qkv_hiera(
                qkv=qkv_linear,
                dim=dim,
                dim_out=dim_out,
                num_heads=num_heads,
                rank=rank,
                alpha=alpha,
            )

            attn.qkv = lora_qkv

            self.A_weights_q.append(lora_qkv.lora_A_q)
            self.B_weights_q.append(lora_qkv.lora_B_q)
            self.A_weights_v.append(lora_qkv.lora_A_v)
            self.B_weights_v.append(lora_qkv.lora_B_v)

    def save_lora_parameters(self, filename: str):
        a_q_tensors = {f"w_a_q_{i:03d}": self.A_weights_q[i].weight.detach().cpu() for i in range(len(self.A_weights_q))}
        b_q_tensors = {f"w_b_q_{i:03d}": self.B_weights_q[i].weight.detach().cpu() for i in range(len(self.B_weights_q))}
        a_v_tensors = {f"w_a_v_{i:03d}": self.A_weights_v[i].weight.detach().cpu() for i in range(len(self.A_weights_v))}
        b_v_tensors = {f"w_b_v_{i:03d}": self.B_weights_v[i].weight.detach().cpu() for i in range(len(self.B_weights_v))}

        merged_dict = {**a_q_tensors, **b_q_tensors, **a_v_tensors, **b_v_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str):

        with safe_open(filename, framework="pt") as f:
            with torch.no_grad():
                for i, w_A_q in enumerate(self.A_weights_q):
                    key = f"w_a_q_{i:03d}"
                    if key not in f.keys():
                        raise KeyError(f"Key {key} not found in the saved LoRA weights.")
                    saved = f.get_tensor(key).to(device=w_A_q.weight.device, dtype=w_A_q.weight.dtype)
                    if saved.shape != w_A_q.weight.shape:
                        raise ValueError(f"{key} shape mismatch: saved {saved.shape} vs model {w_A_q.weight.shape}")
                    w_A_q.weight.copy_(saved)

                for i, w_B_q in enumerate(self.B_weights_q):
                    key = f"w_b_q_{i:03d}"
                    if key not in f.keys():
                        raise KeyError(f"Key {key} not found in the saved LoRA weights.")
                    saved = f.get_tensor(key).to(device=w_B_q.weight.device, dtype=w_B_q.weight.dtype)
                    if saved.shape != w_B_q.weight.shape:
                        raise ValueError(f"{key} shape mismatch: saved {saved.shape} vs model {w_B_q.weight.shape}")
                    w_B_q.weight.copy_(saved)

                for i, w_A_v in enumerate(self.A_weights_v):
                    key = f"w_a_v_{i:03d}"
                    if key not in f.keys():
                        raise KeyError(f"Key {key} not found in the saved LoRA weights.")
                    saved = f.get_tensor(key).to(device=w_A_v.weight.device, dtype=w_A_v.weight.dtype)
                    if saved.shape != w_A_v.weight.shape:
                        raise ValueError(f"{key} shape mismatch: saved {saved.shape} vs model {w_A_v.weight.shape}")
                    w_A_v.weight.copy_(saved)

                for i, w_B_v in enumerate(self.B_weights_v):
                    key = f"w_b_v_{i:03d}"
                    if key not in f.keys():
                        raise KeyError(f"Key {key} not found in the saved LoRA weights.")
                    saved = f.get_tensor(key).to(device=w_B_v.weight.device, dtype=w_B_v.weight.dtype)
                    if saved.shape != w_B_v.weight.shape:
                        raise ValueError(f"{key} shape mismatch: saved {saved.shape} vs model {w_B_v.weight.shape}")
                    w_B_v.weight.copy_(saved)
