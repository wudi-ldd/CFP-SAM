import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import safe_open, save_file

class LoRA_qkv_hiera(nn.Module):
    """
    A LoRA adapter for the MultiScaleAttention module in the second-generation SAM model.
    Implements Low-Rank Adaptation (LoRA) to efficiently fine-tune the QKV transformations in the attention mechanism.
    
    Parameters:
        qkv: The original qkv linear layer
        dim: Input feature dimension
        dim_out: Output feature dimension
        num_heads: Number of attention heads
        rank: Rank of the LoRA adaptation (dimension of the low-rank decomposition)
        alpha: Scaling factor for LoRA
        dropout_p: Dropout probability to prevent overfitting
    
    Returns:
        qkv: Adjusted qkv tensor after applying LoRA
    """
    def __init__(self, qkv, dim, dim_out, num_heads, rank, alpha=1.0, dropout_p=0.5):
        super().__init__()
        self.qkv = qkv  # Original qkv transformation layer
        self.dim = dim  # Input dimension
        self.dim_out = dim_out  # Output dimension
        self.num_heads = num_heads  # Number of attention heads
        self.rank = rank  # LoRA rank
        self.alpha = alpha  # Scaling factor
        self.scaling = self.alpha / self.rank  # LoRA scaling coefficient
        self.head_dim = dim_out // num_heads  # Dimension per attention head

        # Define LoRA weights for Q and V
        # A matrices project high-dimensional input to low-dimensional space
        self.lora_A_q = nn.Linear(dim, rank, bias=False)
        self.lora_A_v = nn.Linear(dim, rank, bias=False)
        # B matrices project low-dimensional representation back to original dimension
        self.lora_B_q = nn.Linear(rank, dim_out, bias=False)
        self.lora_B_v = nn.Linear(rank, dim_out, bias=False)
        
        # Add dropout layers to prevent overfitting
        self.dropout_q = nn.Dropout(p=dropout_p)
        self.dropout_v = nn.Dropout(p=dropout_p)

        # Initialize LoRA weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize LoRA weight parameters:
        - A matrices are initialized using Kaiming uniform initialization
        - B matrices are initialized to zero to ensure no impact at the start of training
        """
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        """
        Forward pass function
        
        Parameters:
            x: Input tensor of shape (B, H, W, dim)
            
        Returns:
            qkv: Adjusted qkv tensor of shape (B, H*W, 3, num_heads, head_dim)
        """
        # Get input tensor shape
        B, H, W, _ = x.shape

        # 1. Compute the original qkv transformation
        qkv = self.qkv(x)  # (B, H, W, 3 * dim_out)

        # 2. Compute LoRA contributions and apply dropout
        # LoRA path for Q
        lora_q = self.lora_A_q(x)  # Dimension reduction
        lora_q = self.dropout_q(lora_q)  # Apply dropout
        lora_q = self.lora_B_q(lora_q)  # Dimension expansion
        lora_q = lora_q * self.scaling  # Apply scaling

        # LoRA path for V
        lora_v = self.lora_A_v(x)
        lora_v = self.dropout_v(lora_v)
        lora_v = self.lora_B_v(lora_v)
        lora_v = lora_v * self.scaling

        # 3. Reshape tensors to fit multi-head attention mechanism
        # Reshape qkv to (B, H*W, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, self.head_dim)
        # Reshape LoRA contributions to (B, H*W, num_heads, head_dim)
        lora_q = lora_q.reshape(B, H * W, self.num_heads, self.head_dim)
        lora_v = lora_v.reshape(B, H * W, self.num_heads, self.head_dim)

        # 4. Separate original q, k, v
        q, k, v = qkv.unbind(dim=2)

        # 5. Add LoRA contributions to original q and v
        q = q + lora_q
        v = v + lora_v

        # 6. Recombine qkv
        qkv = torch.stack([q, k, v], dim=2)

        return qkv

class LoRA_sam2(nn.Module):
    """
    Adds LoRA weights to the attention modules of the second-generation SAM's image encoder.

    Parameters:
        sam_model: An instance of the second-generation SAM model
        rank: Rank of the LoRA adaptation
        lora_layer: List of layers to apply LoRA to (default is all layers)
        alpha: Scaling factor for LoRA

    Returns:
        None
    """
    def __init__(self, sam_model, rank: int, lora_layer=None, alpha=1.0):
        super(LoRA_sam2, self).__init__()
        self.rank = rank
        assert rank > 0

        self.sam_model = sam_model

        # Get the Hiera backbone
        self.backbone = sam_model.image_encoder.trunk

        # Get all blocks
        self.blocks = self.backbone.blocks

        if lora_layer:
            self.lora_layer = lora_layer
            print(f"LoRA applied to layers: {lora_layer}")
        else:
            # Default to applying LoRA to all layers
            self.lora_layer = list(range(len(self.blocks)))

        # Use separate ModuleLists to manage LoRA weights for q and v
        self.A_weights_q = nn.ModuleList()
        self.B_weights_q = nn.ModuleList()
        self.A_weights_v = nn.ModuleList()
        self.B_weights_v = nn.ModuleList()

        # Freeze the image encoder's parameters
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            # Get the attention module
            attn = blk.attn

            # Get the qkv linear layer
            qkv_linear = attn.qkv

            dim = attn.dim  # Input dimension
            dim_out = attn.dim_out  # Output dimension
            num_heads = attn.num_heads

            # Replace the qkv linear layer with the LoRA_qkv_hiera module
            lora_qkv = LoRA_qkv_hiera(qkv_linear, dim, dim_out, num_heads, rank, alpha=alpha)

            # Replace the qkv linear layer in the attention module
            attn.qkv = lora_qkv

            # Store LoRA weights
            self.A_weights_q.append(lora_qkv.lora_A_q)
            self.B_weights_q.append(lora_qkv.lora_B_q)
            self.A_weights_v.append(lora_qkv.lora_A_v)
            self.B_weights_v.append(lora_qkv.lora_B_v)

    def save_lora_parameters(self, filename: str):
        """
        Saves the LoRA weights applied to the attention modules as a safetensors file.

        Parameters:
            filename: Name of the file to save

        Returns:
            None
        """
        # Save LoRA weights for q
        a_q_tensors = {f"w_a_q_{i:03d}": self.A_weights_q[i].weight.cpu() for i in range(len(self.A_weights_q))}
        b_q_tensors = {f"w_b_q_{i:03d}": self.B_weights_q[i].weight.cpu() for i in range(len(self.B_weights_q))}

        # Save LoRA weights for v
        a_v_tensors = {f"w_a_v_{i:03d}": self.A_weights_v[i].weight.cpu() for i in range(len(self.A_weights_v))}
        b_v_tensors = {f"w_b_v_{i:03d}": self.B_weights_v[i].weight.cpu() for i in range(len(self.B_weights_v))}

        # Merge all weight dictionaries
        merged_dict = {**a_q_tensors, **b_q_tensors, **a_v_tensors, **b_v_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str):
        """
        Loads LoRA weights from a safetensors file and moves them to the appropriate device.

        Parameters:
            filename: Name of the file containing the saved weights

        Returns:
            None
        """
        with safe_open(filename, framework="pt") as f:
            # Load LoRA weights for q
            for i, w_A_q in enumerate(self.A_weights_q):
                saved_key = f"w_a_q_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key).clone().detach().to(w_A_q.weight.device)
                    w_A_q.weight = nn.Parameter(saved_tensor)
                else:
                    raise KeyError(f"Key {saved_key} not found in the saved LoRA weights.")

            for i, w_B_q in enumerate(self.B_weights_q):
                saved_key = f"w_b_q_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key).clone().detach().to(w_B_q.weight.device)
                    w_B_q.weight = nn.Parameter(saved_tensor)
                else:
                    raise KeyError(f"Key {saved_key} not found in the saved LoRA weights.")

            # Load LoRA weights for v
            for i, w_A_v in enumerate(self.A_weights_v):
                saved_key = f"w_a_v_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key).clone().detach().to(w_A_v.weight.device)
                    w_A_v.weight = nn.Parameter(saved_tensor)
                else:
                    raise KeyError(f"Key {saved_key} not found in the saved LoRA weights.")

            for i, w_B_v in enumerate(self.B_weights_v):
                saved_key = f"w_b_v_{i:03d}"
                if saved_key in f.keys():
                    saved_tensor = f.get_tensor(saved_key).clone().detach().to(w_B_v.weight.device)
                    w_B_v.weight = nn.Parameter(saved_tensor)
                else:
                    raise KeyError(f"Key {saved_key} not found in the saved LoRA weights.")