import torch
from typing import List, Optional
from torch.utils.checkpoint import checkpoint


try:
    from torch.utils.checkpoint import _StopRecomputationError
except Exception:
    _StopRecomputationError = None


def forward_all(self, sample: torch.Tensor, style_intervener: Optional[torch.nn.Module] = None):

    trunk_features = self.trunk.forward_inter(sample)


    stages_nhwc = [t.permute(0, 2, 3, 1).contiguous() for t in trunk_features]

    if style_intervener is not None:
        stages_nhwc = style_intervener(stages_nhwc)

    trunk_features_mod = [t.permute(0, 3, 1, 2).contiguous() for t in stages_nhwc]

    features, pos = self.neck(trunk_features_mod)
    src = features[-1]

    output = {
        "vision_features": src,
        "vision_pos_enc": pos,
        "backbone_fpn": features,
        "trunk_features": trunk_features_mod,
        "stages_nhwc": stages_nhwc,
    }
    return output


def _ckpt_call(fn, x, use_reentrant: bool):

    autocast_enabled = torch.is_autocast_enabled()
    device_type = x.device.type

    gpu_dtype = None
    cpu_dtype = None
    try:
        gpu_dtype = torch.get_autocast_gpu_dtype()
    except Exception:
        gpu_dtype = None
    try:
        cpu_dtype = torch.get_autocast_cpu_dtype()
    except Exception:
        cpu_dtype = None

    def wrapped(inp: torch.Tensor):
        if not autocast_enabled:
            return fn(inp)

        try:
            if device_type == "cuda":
                dtype = gpu_dtype
                if dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                        return fn(inp)
                else:
                    with torch.autocast(device_type="cuda", enabled=True):
                        return fn(inp)
            else:
                dtype = cpu_dtype
                if dtype is not None:
                    with torch.autocast(device_type="cpu", dtype=dtype, enabled=True):
                        return fn(inp)
                else:
                    with torch.autocast(device_type="cpu", enabled=True):
                        return fn(inp)

        except Exception as e:
            if _StopRecomputationError is not None and isinstance(e, _StopRecomputationError):
                raise

            if device_type == "cuda":
                from torch.cuda.amp import autocast as cuda_autocast
                with cuda_autocast(enabled=True):
                    return fn(inp)
            return fn(inp)

    try:
        return checkpoint(wrapped, x, use_reentrant=use_reentrant, preserve_rng_state=True)
    except TypeError:
        return checkpoint(wrapped, x)


def forward_inter(self, x: torch.Tensor) -> List[torch.Tensor]:

    x = self.patch_embed(x)
    x = x + self._get_pos_embed(x.shape[1:3])

    use_ckpt = bool(getattr(self, "use_checkpoint", False)) and bool(self.training)
    ckpt_layers = getattr(self, "checkpoint_layers", None)
    use_reentrant = bool(getattr(self, "checkpoint_use_reentrant", False))

    trunk_features = []

    for i, blk in enumerate(self.blocks):
        if use_ckpt and (ckpt_layers is None or i in ckpt_layers):
            x = _ckpt_call(blk, x, use_reentrant=use_reentrant)
        else:
            x = blk(x)

        if i in self.stage_ends:
            feats = x.permute(0, 3, 1, 2).contiguous()
            trunk_features.append(feats)

    assert len(trunk_features) >= 4, "forward_inter expects features from at least four stages."
    trunk_features = trunk_features[:4]
    return trunk_features
