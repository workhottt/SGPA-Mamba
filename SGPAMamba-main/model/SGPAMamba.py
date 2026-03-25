import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class UniversalEncoder(nn.Module):
    def __init__(self, backbone, in_ch, backbone_dim, out_dim, extract_fn=None):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.input_proj = nn.Conv2d(in_ch, 3, kernel_size=1) if in_ch != 3 else nn.Identity()
        self.extract_fn = extract_fn
        self.feature_adapter = nn.Sequential(
            nn.Linear(backbone_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.input_proj(x)
        with torch.no_grad():
            if self.extract_fn is not None:
                features = self.extract_fn(self.backbone, x)
            else:
                features = self.backbone(x)
                if len(features.shape) == 4:
                    features = rearrange(features, 'b c h w -> b (h w) c')
        return self.feature_adapter(features)

class CTMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.x_proj = nn.Linear(dim, dim // 4)
        self.dt_proj = nn.Linear(dim // 4, dim)
        self.out_proj = nn.Linear(dim, dim)

    def selective_scan(self, x):
        res = x
        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv(x)
        x = rearrange(x, 'b c n -> b n c')
        dt = F.softplus(self.dt_proj(self.x_proj(x)))
        return x * torch.sigmoid(dt) + res

    def forward(self, th, tl):
        gh, gl = torch.sigmoid(th) * tl, torch.sigmoid(tl) * th
        th_in, tl_in = th + gh, tl + gl
        h_fwd, l_fwd = self.selective_scan(th_in), self.selective_scan(tl_in)
        h_bwd = torch.flip(self.selective_scan(torch.flip(th_in, [1])), [1])
        l_bwd = torch.flip(self.selective_scan(torch.flip(tl_in, [1])), [1])
        return self.out_proj(h_fwd + h_bwd), self.out_proj(l_fwd + l_bwd)

class CMU_Structural_Stage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spectral_unc = nn.Conv2d(dim, 1, 1)
        self.elev_unc = nn.Conv2d(dim, 1, 1)
        
        lap = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("laplacian", lap)
        
        self.csa_gate = nn.Sequential(
            nn.Conv2d(1, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.spm = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, fh, fl, lidar_orig):
        Us, Ue = F.softplus(self.spectral_unc(fh)), F.softplus(self.elev_unc(fl))
        ws, we = torch.exp(-Us), 1.0 / (1.0 + Ue + 1e-6)
        fh_mod, fl_mod = fh * ws, fl * we
        
        with torch.no_grad():
            edge_highres = F.conv2d(lidar_orig, self.laplacian, padding=1).abs()
            curv = F.adaptive_avg_pool2d(edge_highres, fh.shape[-2:])
            curv = curv / (curv.mean() + 1e-6)
            
        gate = self.csa_gate(curv)
        fh_out = fh_mod * (1 + gate) 
        fl_out = fl_mod + self.spm(fl_mod)
        
        return fh_out, fl_out, (Us, Ue)

class SGPA_Net(nn.Module):
    def __init__(self, hsi_encoder, lidar_encoder, num_classes, dim=128, img_size=224):
        super().__init__()
        self.grid_size = img_size // 14
        self.hsi_encoder = hsi_encoder
        self.lidar_encoder = lidar_encoder
        self.ct_mamba = CTMambaBlock(dim)
        self.cmu_struct = CMU_Structural_Stage(dim)
        
        self.mu = nn.Parameter(torch.randn(num_classes, dim))
        self.log_var = nn.Parameter(torch.zeros(num_classes, dim))
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, hsi, lidar):
        th, tl = self.hsi_encoder(hsi), self.lidar_encoder(lidar)
        th_mamba, tl_mamba = self.ct_mamba(th, tl)
        
        fh_m = rearrange(th_mamba, 'b (h w) c -> b c h w', h=self.grid_size)
        fl_m = rearrange(tl_mamba, 'b (h w) c -> b c h w', h=self.grid_size)
        
        fh_final, fl_final, u_maps = self.cmu_struct(fh_m, fl_m, lidar)
        
        fh_vec = F.adaptive_avg_pool2d(fh_final, 1).flatten(1)
        fl_vec = F.adaptive_avg_pool2d(fl_final, 1).flatten(1)
        logits = self.classifier(torch.cat([fh_vec, fl_vec], dim=1))
        
        return {
            "logits": logits, 
            "features": (fh_vec, fl_vec), 
            "uncertainty": u_maps, 
            "tokens_mamba": (th_mamba, tl_mamba),
            "spatial_feats": (fh_m, fh_final, fl_final) 
        }

    def get_loss(self, outputs, labels):
        logits = outputs["logits"]
        fh_vec, fl_vec = outputs["features"]
        fh_before, fh_after, fl_after = outputs["spatial_feats"]
        mu, var = self.mu, torch.exp(self.log_var)

        l_cls = F.cross_entropy(logits, labels, label_smoothing=0.1)
        
        f_fused = (fh_vec + fl_vec) / 2
        l_proto = 0.5 * torch.mean(torch.sum(((f_fused - mu[labels])**2) / var[labels] + torch.log(var[labels]), dim=1))
        
        sim = F.cosine_similarity(fh_vec, fl_vec, dim=-1)
        l_contr = torch.mean(1 - sim)

        th_final = rearrange(fh_after, 'b c h w -> b (h w) c')
        tl_final = rearrange(fl_after, 'b c h w -> b (h w) c') 
        
        adj_h = torch.softmax(torch.bmm(th_final, th_final.transpose(1, 2)), dim=-1)
        adj_l = torch.softmax(torch.bmm(tl_final, tl_final.transpose(1, 2)), dim=-1)
        l_struct = F.mse_loss(adj_h, adj_l)

        l_mod_act = 0.1 * F.mse_loss(fh_after, fh_before) 

        total_loss = l_cls + 0.1 * l_proto + 0.1 * l_contr + 0.1 * l_struct + l_mod_act
        return total_loss