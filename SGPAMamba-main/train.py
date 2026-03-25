import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from einops import rearrange
from tqdm import tqdm
import os
import gc
import random

class dinov2_FoundationEncoder(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        for param in self.backbone.parameters():
            param.requires_grad = False 
            
        self.input_proj = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.feature_adapter = nn.Sequential(
            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x_3ch = self.input_proj(x)
        with torch.no_grad():
            features = self.backbone.forward_features(x_3ch)
            tokens = features["x_norm_patchtokens"] 
        return self.feature_adapter(tokens)

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
        fh_out = fh_mod * (1 + gate) 
        fl_out = fl_mod + self.spm(fl_mod)
        return fh_out, fl_out

class SGPA_Net(nn.Module):
    def __init__(self, hsi_bands, num_classes, dim=128, img_size=224):
        super().__init__()
        self.grid_size = img_size // 14
        self.hsi_backbone = dinov2_FoundationEncoder(hsi_bands, dim)
        self.lidar_backbone = dinov2_FoundationEncoder(1, dim)
        self.ct_mamba = CTMambaBlock(dim)
        self.cmu_struct = CMU_Structural_Stage(dim)
        self.mu = nn.Parameter(torch.randn(num_classes, dim))
        self.log_var = nn.Parameter(torch.zeros(num_classes, dim))
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, hsi, lidar):
        th, tl = self.hsi_backbone(hsi), self.lidar_backbone(lidar)
        th_mamba, tl_mamba = self.ct_mamba(th, tl)
        fh_m = rearrange(th_mamba, 'b (h w) c -> b c h w', h=self.grid_size)
        fl_m = rearrange(tl_mamba, 'b (h w) c -> b c h w', h=self.grid_size)
        fh_final, fl_final = self.cmu_struct(fh_m, fl_m, lidar)
        fh_vec = F.adaptive_avg_pool2d(fh_final, 1).flatten(1)
        fl_vec = F.adaptive_avg_pool2d(fl_final, 1).flatten(1)
        logits = self.classifier(torch.cat([fh_vec, fl_vec], dim=1))
        return {"logits": logits, "features": (fh_vec, fl_vec), "spatial_feats": (fh_m, fh_final, fl_final)}

    def get_loss(self, outputs, labels, lidar_orig):
        l_cls = F.cross_entropy(outputs["logits"], labels, label_smoothing=0.1)
        fh_vec, fl_vec = outputs["features"]
        f_fused = (fh_vec + fl_vec) / 2
        mu, var = self.mu, torch.exp(self.log_var)
        l_proto = 0.5 * torch.mean(torch.sum(((f_fused - mu[labels])**2) / var[labels] + torch.log(var[labels]), dim=1))
        l_contr = torch.mean(1 - F.cosine_similarity(fh_vec, fl_vec, dim=-1))
        
        fh_before, fh_after, fl_after = outputs["spatial_feats"]
        th_final = rearrange(fh_after, 'b c h w -> b (h w) c')
        tl_final = rearrange(fl_after, 'b c h w -> b (h w) c')
        adj_h = torch.softmax(torch.bmm(th_final, th_final.transpose(1, 2)), dim=-1)
        adj_l = torch.softmax(torch.bmm(tl_final, tl_final.transpose(1, 2)), dim=-1)
        l_struct = F.mse_loss(adj_h, adj_l)
        l_mod_act = 0.1 * F.mse_loss(fh_after, fh_before) 
        
        return l_cls + 0.1 * l_proto + 0.1 * l_contr + 0.1 * l_struct + l_mod_act

class HoustonPatchDataset(Dataset):
    def __init__(self, data_path, pca_obj, mode='train'):
        self.mode = mode
        hsi = sio.loadmat(f"{data_path}/houston_hsi.mat")['houston_hsi'].astype(np.float32)
        lidar = sio.loadmat(f"{data_path}/houston_lidar.mat")['houston_lidar'].astype(np.float32)
        gt = sio.loadmat(f"{data_path}/houston_gt.mat")['houston_gt']
        idx_mat = sio.loadmat(f"{data_path}/houston_index.mat")
        
        hsi = (hsi - np.min(hsi)) / (np.max(hsi) - np.min(hsi) + 1e-6)
        lidar = (lidar - np.min(lidar)) / (np.max(lidar) - np.min(lidar) + 1e-6)
        if lidar.ndim == 3: lidar = np.squeeze(lidar)
        
        hsi_pca = pca_obj.transform(hsi.reshape(-1, hsi.shape[-1])).reshape(hsi.shape[0], hsi.shape[1], -1)
        self.hsi_padded = np.pad(hsi_pca, ((32, 32), (32, 32), (0, 0)), 'reflect')
        self.lidar_padded = np.pad(lidar, ((32, 32), (32, 32)), 'reflect')
        
        idx_key = 'houston_train' if mode == 'train' else 'houston_test'
        mask = idx_mat[idx_key]
        r_list, c_list = (mask[:,0].astype(int), mask[:,1].astype(int)) if len(mask.shape) == 2 else np.where(mask > 0)
        
        self.points, self.labels = [], []
        for r, c in zip(r_list, c_list):
            if gt[r, c] > 0:
                self.points.append((r, c))
                self.labels.append(int(gt[r, c]) - 1)

    def __len__(self): return len(self.points)
    def __getitem__(self, i):
        r, c = self.points[i]
        hsi_patch = self.hsi_padded[r:r+64, c:c+64, :].transpose(2, 0, 1)
        lidar_patch = self.lidar_padded[r:r+64, c:c+64][np.newaxis, ...]
        
        if self.mode == 'train':
            k = random.choice([0, 1, 2, 3])
            hsi_patch = np.rot90(hsi_patch, k, axes=(1, 2))
            lidar_patch = np.rot90(lidar_patch, k, axes=(1, 2))
            if random.random() > 0.5:
                hsi_patch = np.flip(hsi_patch, axis=1)
                lidar_patch = np.flip(lidar_patch, axis=1)
        
        hsi_t = F.interpolate(torch.from_numpy(hsi_patch.copy()).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)
        lidar_t = F.interpolate(torch.from_numpy(lidar_patch.copy()).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)
        
        return hsi_t.float(), lidar_t.float(), torch.tensor(self.labels[i]).long()

def validate_with_progress(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for hsi, lidar, labels in loader:
            outputs = model(hsi.to(device), lidar.to(device))
            _, pred = outputs["logits"].max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    acc = 100. * np.trace(cm) / np.sum(cm)
    print(f"\n >>> [VALIDATE] Accuracy: {acc:.2f}%")
    return acc

def safe_save_slim(model, save_path):
    full_sd = model.state_dict()
    target_modules = ["ct_mamba", "cmu_struct", "classifier", "feature_adapter", "input_proj", "mu", "log_var"]
    slim_sd = {k: v.clone().cpu() for k, v in full_sd.items() if any(mod in k for mod in target_modules)}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "/root/autodl-tmp/FL/SGPAMamba-main/data/Houston2013"
    SAVE_DIR = "/root/autodl-tmp/FL/SGPAMamba-main/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True) 
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5}, 
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        curr_ep = epoch + 1
        model.train()
        correct, total = 0, 0
        train_loop = tqdm(train_loader, desc=f"Epoch {curr_ep}/50")
        
        for hsi, lidar, labels in train_loop:
            hsi, lidar, labels = hsi.to(device), lidar.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(hsi, lidar)
            loss = model.get_loss(outputs, labels, lidar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, pred = outputs["logits"].max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")
        
        scheduler.step()
        if curr_ep % 5 == 0 or curr_ep <= 3:
            val_acc = validate_with_progress(model, test_loader, device)
            safe_save_slim(model, os.path.join(SAVE_DIR, f"sgpa_ep{curr_ep}_acc{val_acc:.1f}.pth"))
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()