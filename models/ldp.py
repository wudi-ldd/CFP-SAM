import math
import torch
import torch.nn as nn

class LearnableDomainPerturbation(nn.Module):

    def __init__(
        self,
        num_features: int,
        p: float = 0.5,
        eps: float = 1e-6,
        r_min: float = 0.05,
        r_max: float = 0.45,
        per_channel: bool = True,
        rc_alpha: float = 0.25,
        rc_w_floor_px: float = 4.0,
        gamma_max: float = 1.0,
        beta_max: float = 1.0,
    ):
        super().__init__()
        self.eps = float(eps)
        self.p = float(p)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.per_channel = bool(per_channel)
        self.rc_alpha = float(rc_alpha)
        self.rc_w_floor_px = float(rc_w_floor_px)

        self.gamma_max = float(gamma_max)
        self.beta_max = float(beta_max)

        self.r_logits = nn.Parameter(torch.zeros(num_features), requires_grad=True)

        if self.per_channel:
            self.gamma_raw = nn.Parameter(torch.zeros(num_features), requires_grad=True)
            self.beta_raw  = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.gamma_raw = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.beta_raw  = nn.Parameter(torch.zeros(1), requires_grad=True)

    @staticmethod
    def _fft2(x):
        return torch.fft.fft2(x, norm='ortho')

    @staticmethod
    def _ifft2(X):
        return torch.fft.ifft2(X, norm='ortho')

    @staticmethod
    def _fftshift(X):
        return torch.fft.fftshift(X, dim=(-2, -1))

    @staticmethod
    def _ifftshift(X):
        return torch.fft.ifftshift(X, dim=(-2, -1))

    def _build_rc_mask(self, H: int, W: int, r_pix: torch.Tensor, device, dtype):

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        cy, cx = H // 2, W // 2
        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).view(1, 1, H, W)

        w = torch.clamp(self.rc_alpha * r_pix, min=self.rc_w_floor_px)
        t = ((dist - r_pix) / w).clamp(0.0, 1.0)
        M = 0.5 * (1.0 + torch.cos(math.pi * t))
        return M

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.p < 1.0 and torch.rand(1, device=x_nhwc.device).item() > self.p):
            return x_nhwc

        orig_dtype = x_nhwc.dtype

        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        device = x.device
        min_hw = float(min(H, W))

        x32 = x.to(torch.float32)
        F_complex = self._fft2(x32)
        F_complex = self._fftshift(F_complex)
        A = torch.abs(F_complex)
        P = torch.angle(F_complex)

        r_c = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(self.r_logits)
        r_pix = (r_c * min_hw).view(1, C, 1, 1)
        M = self._build_rc_mask(H, W, r_pix, device=device, dtype=torch.float32)
        M = M.expand(B, C, H, W)

        denom = M.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)

        mu = (A * M).sum(dim=(-2, -1), keepdim=True) / denom

        var = ((A - mu) ** 2 * M).sum(dim=(-2, -1), keepdim=True) / denom
        std = (var + self.eps).sqrt()

        if self.per_channel:
            gamma = (self.gamma_max * torch.tanh(self.gamma_raw)).view(1, C, 1, 1)
            beta  = (self.beta_max  * torch.tanh(self.beta_raw)).view(1, C, 1, 1)
        else:
            gamma = (self.gamma_max * torch.tanh(self.gamma_raw)).view(1, 1, 1, 1)
            beta  = (self.beta_max  * torch.tanh(self.beta_raw)).view(1, 1, 1, 1)

        scale = torch.exp(gamma)
        std_t = std * scale
        mean_t = mu + beta * std

        A_norm = (A - mu) / (std + self.eps)
        A_lf_new = (A_norm * std_t + mean_t) * M

        A_new = A * (1.0 - M) + A_lf_new
        A_new = A_new.clamp_min(0.0)

        F_new = torch.polar(A_new, P)
        F_new = self._ifftshift(F_new)
        x_new = self._ifft2(F_new).real

        return x_new.permute(0, 2, 3, 1).contiguous().to(orig_dtype)





class MultiStageLDP(nn.Module):

    STAGE_CHANNELS = [144, 288, 576, 1152]

    def __init__(self, stage_ids=(0, 1, 2, 3), p: float = 0.5):
        super().__init__()
        stage_ids = sorted(list(set(int(i) for i in stage_ids if i in [0, 1, 2, 3])))
        self.active_indices = stage_ids
        modules = []
        raw_list = []
        for i in range(4):
            if i in stage_ids:
                ldp = LearnableDomainPerturbation(num_features=self.STAGE_CHANNELS[i], p=p)
                modules.append(ldp)
                raw_list.append(ldp)
            else:
                ident = nn.Identity()
                modules.append(ident)
                raw_list.append(None)

        self.modules_per_stage = nn.ModuleList(modules)
        self._raw_ldps = raw_list

    def module_for(self, idx):
        if idx < 0 or idx >= 4:
            return None
        return self._raw_ldps[idx]

    def forward(self, stages_nhwc_list):

        assert isinstance(stages_nhwc_list, (list, tuple)) and len(stages_nhwc_list) >= 4,\
            "MultiStageLDP expects an NHWC feature list with at least four stages."
        out = []
        for i, x in enumerate(stages_nhwc_list):
            mod = self.modules_per_stage[i]

            if not isinstance(mod, nn.Identity):
                x = mod(x)
            out.append(x)
        return out
