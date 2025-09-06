"""
基于你的ConvLSTM实现，改成像素级分类（语义分割风格）
- 输入: (B, T, C, H, W)，如 T=3: [2008,2012,2016]，C=landuse(1)+静态因子若干
- 输出: (B, num_classes, H, W) 的 logits；loss = CE(logits, y)；y形状(B,H,W)，值∈{0,1,2}
""" 

import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from ConvLSTM import ConvLSTMSeg  # 新增的像素级头
from tqdm import tqdm
import random
import numpy as np
import torch
class FocalLoss(nn.Module):
    """
    交叉熵的改进版：对容易分类的样本降权，聚焦难样本（少数类）。
    gamma 越大，抑制“容易样本”(比如大量的不变=0)越强。
    alpha 可以传入类别权重（tensor[K]）。
    """
    def __init__(self, gamma=1.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor or None

    def forward(self, logits, target):
    # target 保持 -1 不变
      ce = nn.functional.cross_entropy(
        logits, target, weight=self.alpha,
        reduction="none", ignore_index=-1
    )
      pt = torch.exp(-ce)
      loss = ((1 - pt) ** self.gamma) * ce
      valid = (target != -1)
      return loss[valid].mean() if valid.any() else loss.mean()


def set_global_determinism(seed=42):
    """
    设置随机种子，尽可能确保可重复的结果
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ===================== 配置 =====================
class Config:
    # 数据路径（按你的文件改）
    data_dir = r"D:\paper2DATA\inputdata"
    landuse = {
        2008: r"D:\paper2DATA\inputdata\landuse\2008_to2020.tif",
        2012: r"D:\paper2DATA\inputdata\landuse\2012_to2020.tif",
        2016: r"D:\paper2DATA\inputdata\landuse\2016_to2020.tif",
        2020: r"D:\paper2DATA\inputdata\landuse\2020.tif",
        2024: r"D:\paper2DATA\inputdata\landuse\2024_to2020.tif"
    }
    labels = {
        # 修改为 _to2020 后缀的标签文件路径
        "08_12": r"D:\paper2DATA\inputdata\convlstm\12label\12label_to2020.tif",
        "12_16": r"D:\paper2DATA\inputdata\convlstm\16label\16label_to2020.tif",
        "16_20": r"D:\paper2DATA\inputdata\convlstm\20labelnew\c20labelnew_to2020.tif",
        "20_24": r"D:\paper2DATA\inputdata\convlstm\24label\c24label_to2020.tif",
    }
    factor_files = [
        r"D:\paper2DATA\inputdata\convlstm\floodrisk\floodrisk_to2020.tif",
        # r"D:\paper2DATA\inputdata\convlstm\coastline_to2020.tif", 
        r"D:\paper2DATA\inputdata\convlstm\fibre\fibre_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\slope_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\3water1000m_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\dem\dem_to2020.tif" 
    ]
    # 在 Config 里加两行
    train_intervals = [(2008, 2012, "08_12"), (2012, 2016, "12_16"), (2016, 2020, "16_20")]  # 只用这3段训练
    holdout_interval = (2020, 2024, "20_24")                          # 只用这段做预测/评估

    # 训练超参
    time_steps = 2              # 2008/2012/2016
    patch_size = 64
    stride = 64
    batch_size = 4
    epochs = 40
    lr = 1e-4
    num_classes = 3

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型结构
    in_channels = None          # 运行时自动= 1 + len(factors)
    out_channels = [32, 32]
    kernel_size = [(3,3), (3,3)]
    num_layers = len(out_channels)

    # 输出
    save_dir = r"D:\paper2DATA\inputdata\outputs"
    import os
    os.makedirs(save_dir, exist_ok=True)


# ================ 工具函数 ==================
def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile
        nodata = src.nodata
    arr = arr.astype(np.float32)
    if nodata is not None:
        arr = np.where(np.isnan(arr) | (arr == nodata), 0.0, arr)
    return arr, profile

def minmax_norm(a, eps=1e-6):
    amin, amax = float(np.min(a)), float(np.max(a))
    if amax - amin < eps:
        return np.zeros_like(a, dtype=np.float32)
    return (a - amin) / (amax - amin)

def stack_pair_with_factors(cfg, year_a, year_b):
    """
    构造一条样本的输入序列: [year_a, year_b]，并附上静态因子
    返回 X_seq: (T=2, H, W, C)
    """
    a, prof = read_tif(cfg.landuse[year_a])
    b, _    = read_tif(cfg.landuse[year_b])

    # landuse一通道（如果已是0/1就不归一）
    la = a[..., None].astype(np.float32)
    lb = b[..., None].astype(np.float32)

    # 静态因子
    fs = []
    for f in cfg.factor_files:
        fa, _ = read_tif(f)
        fs.append(minmax_norm(fa)[..., None])
    F = np.concatenate(fs, axis=-1) if len(fs) > 0 else None

    def with_static(lu, F):
        return lu if F is None else np.concatenate([lu, F], axis=-1)

    xa = with_static(la, F)
    xb = with_static(lb, F)

    X_seq = np.stack([xa, xb], axis=0)  # (2,H,W,C)
    return X_seq, prof

import numpy as np
import rasterio

def read_label(path, num_classes=3):
    """读取三分类标签：0/1/2；NoData/NaN/非法值 → -1（训练时被 ignore）。"""
    with rasterio.open(path) as src:
        arr = src.read(1)
        nd = src.nodata
    arr = arr.astype(np.int32)
    if nd is not None:
        arr = np.where(arr == nd, -1, arr)
    arr = np.where(np.isnan(arr), -1, arr)
    # 只允许 {0,1,2}，其余一律置 -1
    arr = np.where(np.isin(arr, list(range(num_classes))), arr, -1)
    return arr

def build_all_samples(cfg):
    """构建训练样本，使用完整的训练数据"""
    # 使用所有训练区间
    train_pairs = [(2008, 2012, "08_12"), (2012, 2016, "12_16"), (2016, 2020, "16_20")]
    holdout_pair = (2020, 2024, "20_24")

    X_list, Y_list, ref_prof = [], [], None
    
    # 读取所有训练数据
    for y1, y2, key in train_pairs:
        X_seq, prof = stack_pair_with_factors(cfg, y1, y2)
        Y = read_label(cfg.labels[key], num_classes=cfg.num_classes)  # <- 保留 -1
        X_list.append(X_seq)
        Y_list.append(Y)
        ref_prof = prof
    
    # 读取预测用的数据
    X_holdout, _ = stack_pair_with_factors(cfg, *holdout_pair[:2])
    
    return X_list, Y_list, X_holdout, ref_prof


# ================ 数据集 ==================
class RasterSeqDataset(Dataset):
    def __init__(self, X_list, Y_list, patch=64, stride=64):
        """
        X_list: [ (T,H,W,C), ... ]
        Y_list: [ (H,W), ... ]
        """
        super().__init__()
        self.samples = []
        self.patch = patch
        self.stride = stride

        for sid, (X_seq, Y) in enumerate(zip(X_list, Y_list)):
            X_seq = X_seq.astype(np.float32)
            Y = Y.astype(np.int64)
            T, H, W, C = X_seq.shape
            for r in range(0, H - patch + 1, stride):
                for c in range(0, W - patch + 1, stride):
                    self.samples.append((sid, r, c, X_seq, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sid, r, c, X_seq, Y = self.samples[i]
        x = X_seq[:, r:r+self.patch, c:c+self.patch, :]  # (T,P,P,C)
        y = Y[r:r+self.patch, c:c+self.patch]            # (P,P)
        x = np.transpose(x, (0,3,1,2))                   # (T,C,P,P)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()
    
    def get_patch_coords(self, idx):
        """获取patch的坐标"""
        _, r, c, _, _ = self.samples[idx]
        return r, c


# ================ 训练 & 评估 ==================
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="train", ncols=90):
        # x: (B?,T,C,H,W) —— 我们在 collate 时需要把 T,C 排好
        # Dataset返回的是 (T,C,P,P)，DataLoader会在前面加batch维
        x = x.to(device)            # (B,T,C,P,P)
        y = y.to(device)            # (B,P,P)
        opt.zero_grad()
        logits = model(x)           # (B,num_classes,P,P)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_pix, correct = 0.0, 0, 0
    for x, y in tqdm(loader, desc="valid", ncols=90):
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total_pix += y.numel()
    pix_acc = correct / total_pix if total_pix > 0 else 0.0
    return total_loss / len(loader.dataset), pix_acc

# ================ 全图推理 & 导出 ==================
@torch.no_grad()
def predict_full_and_save(cfg: Config, model, X_seq, ref_profile):
    model.eval()
    T, H, W, C = X_seq.shape
    P = cfg.patch_size
    S = cfg.stride

    prob_map = np.zeros((H, W, cfg.num_classes), dtype=np.float32)
    hit = np.zeros((H, W), dtype=np.float32)
    
    # 修正缩进
    r_starts = list(range(0, max(H - P, 0) + 1, S))
    c_starts = list(range(0, max(W - P, 0) + 1, S))
    if r_starts[-1] != H - P:
        r_starts.append(H - P)
    if c_starts[-1] != W - P:
        c_starts.append(W - P)

    for r in r_starts:
        for c in c_starts:
            rr, cc = int(r), int(c)
            x = X_seq[:, rr:rr+P, cc:cc+P, :]              # (T,P,P,C)
            x = np.transpose(x, (0,3,1,2))                 # (T,C,P,P)
            x = torch.from_numpy(x)[None].float().to(cfg.device)  # (1,T,C,P,P)
            logits = model(x)[0]                           # (num_classes,P,P)
            prob = torch.softmax(logits, dim=0).cpu().numpy().transpose(1,2,0)  # (P,P,K)
            prob_map[rr:rr+P, cc:cc+P, :] += prob
            hit[rr:rr+P,  cc:cc+P]  += 1.0


    hit = np.maximum(hit, 1.0)
    prob_map = prob_map / hit[..., None]
    pred_cls = np.argmax(prob_map, axis=-1).astype(np.uint8)

    # 保存分类与概率
    prof = ref_profile.copy()
    prof.update(count=1, dtype=rasterio.uint8, compress="lzw")
    with rasterio.open(os.path.join(cfg.save_dir, "pred_2020_class.tif"), "w", **prof) as dst:
        dst.write(pred_cls, 1)

    for k in range(cfg.num_classes):
        prof_k = ref_profile.copy()
        prof_k.update(count=1, dtype=rasterio.uint16, compress="lzw")
        with rasterio.open(os.path.join(cfg.save_dir, f"pred_2020_prob_class{k}.tif"), "w", **prof_k) as dst:
            dst.write((prob_map[..., k] * 10000).astype(np.uint16), 1)

    # 在保存之前添加统计信息
    print("\n预测结果统计信息：")
    print("概率图范围：")
    for k in range(cfg.num_classes):
        print(f"- 类别{k}概率范围: [{prob_map[..., k].min():.3f}, {prob_map[..., k].max():.3f}]")
    
    print("\n最终分类结果：")
    unique, counts = np.unique(pred_cls, return_counts=True)
    total = pred_cls.size
    for u, c in zip(unique, counts):
        print(f"- 类别{u}: {c}个像素 ({c/total*100:.2f}%)")


def calculate_fom(true_change, pred_change, window_size=3):
    """计算 Figure of Merit (FoM) 分数"""
    from sklearn.metrics import confusion_matrix
    
    # 打印标签和预测的基本信息
    print("\n真实标签统计：")
    u_true, c_true = np.unique(true_change, return_counts=True)
    for u, c in zip(u_true, c_true):
        print(f"- 类别{u}: {c}个像素 ({c/true_change.size*100:.2f}%)")
    
    print("\n预测结果统计：")
    u_pred, c_pred = np.unique(pred_change, return_counts=True)
    for u, c in zip(u_pred, c_pred):
        print(f"- 类别{u}: {c}个像素 ({c/pred_change.size*100:.2f}%)")
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_change.flatten(), pred_change.flatten())
    print(f"\n混淆矩阵：\n{cm}")
    
    # 计算FoM
    if cm.shape[0] < 2 or cm.shape[1] < 2:
        print("警告：预测结果只包含一个类别，无法计算FoM")
        return 0.0
        
    hits = cm[1, 1]  # 正确预测的变化
    misses = cm[1, 0]  # 漏检
    false_alarms = cm[0, 1]  # 误检
    
    # 计算 FoM
    denominator = hits + misses + false_alarms
    if denominator == 0:
        print("警告：分母为0，可能是预测结果全为背景类")
        return 0.0
        
    fom = hits / denominator * 100
    print(f"\nFoM计算详情：")
    print(f"- 正确预测变化(hits): {hits}")
    print(f"- 漏检(misses): {misses}")
    print(f"- 误检(false_alarms): {false_alarms}")
    
    return fom
import numpy as np
from sklearn.metrics import confusion_matrix
def _per_class_fom(true_cls: np.ndarray, pred_cls: np.ndarray, num_classes: int = 3):
    """返回每个类别的 FoM_k 数组（百分数）。-1 视为无效像元并忽略。"""
    mask = (true_cls >= 0) & (true_cls < num_classes)
    t = true_cls[mask].ravel()
    p = pred_cls[mask].ravel()

    cm = confusion_matrix(t, p, labels=list(range(num_classes)))  # KxK
    TP = np.diag(cm).astype(float)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    den = TP + FN + FP
    
    # 打印详细信息
    print("\n混淆矩阵:")
    print(cm)
    print("\n每类统计:")
    for k in range(num_classes):
        print(f"类别 {k}:")
        print(f"- TP: {TP[k]}")
        print(f"- FN: {FN[k]}")
        print(f"- FP: {FP[k]}")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        fom_k = np.where(den > 0, TP / den * 100.0, 0.0)
    return fom_k  # shape: (K,)

def multiclass_macro_fom(true_cls: np.ndarray, pred_cls: np.ndarray, num_classes: int = 3) -> float:
    """返回单一的宏观 FoM（所有类别 FoM 的不加权平均，单位：%）"""
    fom_k = _per_class_fom(true_cls, pred_cls, num_classes)
    
    # 打印每个类别的 FoM
    print("\n各类别 FoM:")
    for k, fom in enumerate(fom_k):
        print(f"类别 {k}: {fom:.2f}%")
    
    macro_fom = float(np.mean(fom_k))
    print(f"\n宏观 FoM: {macro_fom:.2f}%")
    return macro_fom

@torch.no_grad()
def evaluate_fom(cfg, model, device):
    """评估模型的多类别 FoM 分数"""
    # 读取真实变化标签
    true_change, _ = read_label(cfg.labels["20_24"], num_classes=cfg.num_classes)
    true_change = true_change.astype(np.int32)
    
    # 读取预测结果
    pred_path = os.path.join(cfg.save_dir, "pred_2024_class.tif")
    pred_change, _ = read_tif(pred_path)
    pred_change = pred_change.astype(np.int32)
    
    # 计算多类别 FoM
    return multiclass_macro_fom(true_change, pred_change, cfg.num_classes)

def main():
    # 设置随机种子
    set_global_determinism(seed=42)
    
    cfg = Config()

  # 读入训练 & 留出预测数据
    X_list, Y_list, X_pred, ref_profile = build_all_samples(cfg)
    cfg.in_channels = X_list[0].shape[-1]

    # ★ 统计训练标签的类别频次（用两张全图标签 08→12 和 12→16，而不是用 patch）
    import numpy as np, torch
    counts = np.zeros(cfg.num_classes, dtype=np.int64)
    for Y in Y_list:
        y = np.asarray(Y, dtype=np.int32)
        y = y[(y >= 0) & (y < cfg.num_classes)]          # 只统计 0/1/2
        c = np.bincount(y.ravel(), minlength=cfg.num_classes)
        counts += c

    freq = counts / counts.sum()
    weights_np = 1.0 / np.sqrt(freq + 1e-6)   # 使用平方根来减轻权重差异
    weights_np = weights_np / weights_np.sum()
    class_weights = torch.tensor(weights_np, dtype=torch.float32, device=cfg.device)
    print("class_counts:", counts.tolist())
    print("class_weights:", weights_np.tolist())         # 例如 [0.15, 0.45, 0.40]

    # 数据集（只含两段训练数据）
    ds = RasterSeqDataset(X_list, Y_list, patch=cfg.patch_size, stride=cfg.stride)
    n_total = len(ds)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # 模型
    model = ConvLSTMSeg(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        kernel_size=cfg.kernel_size,
        num_layers=cfg.num_layers,
        num_classes=cfg.num_classes,
        batch_first=True,
        bias=True
    ).to(cfg.device)

    # ★ 用 FocalLoss + 类别权重，抑制“全0”问题
    loss_fn = FocalLoss(gamma=1.0, alpha=class_weights)  # 像素级，y形状[B,H,W]，取值0/1/2
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # 训练循环
    best_val = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(cfg.epochs):
        # 使用 tqdm 显示进度
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        # 训练
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)
        
        # 验证
        va_loss, va_pixacc = eval_one_epoch(model, val_loader, loss_fn, cfg.device)
        
        # 学习率调度
        scheduler.step(va_loss)
        current_lr = opt.param_groups[0]['lr']
        
        # 打印信息
        print(f"Train Loss: {tr_loss:.4f}")
        print(f"Valid Loss: {va_loss:.4f}")
        print(f"Pixel Acc: {va_pixacc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 早停检查
        if va_loss < best_val:
            best_val = va_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pt"))
            print("Saved best model!")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 推理并导出GeoTIFF
    model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "best_model.pt"), map_location=cfg.device))
    predict_full_and_save(cfg, model, X_pred, ref_profile)
    print("输出：outputs/pred_2024_class.tif 及各类概率图")

    # 预测完成后计算 FoM
    print("\n开始评估模型性能...")
    fom = evaluate_fom(cfg, model, cfg.device)
    
    print(f"\n=== 训练和评估完成!")
    print(f"预测结果：D:\\paper2DATA\\inputdata\\outputs\\pred_2024_class.tif")
    print(f"FoM 分数：{fom:.2f}%")

if __name__ == "__main__":
    main()
