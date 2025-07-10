# 아마도 폐 ct영상 코드
"""
ConvNeXt + CBAM  + CAM-Alignment  (BBox from CSV)
-----------------------------------------------------------
* 입력  : 원본 슬라이스 (.npy)
* 지도  : CSV에 정의된 Bounding Box → 2D 마스크 on-the-fly
* 라벨  : 파일명 끝의 _{score}.npy (1,2 → 0 / 4,5 → 1)
"""

# ════════════════════════════════════════════════════════════
# 0) 기본 라이브러리
# ════════════════════════════════════════════════════════════
import os, random, csv
from glob import glob
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
import timm, torchvision.transforms.functional as TF
from tqdm import tqdm

# ════════════════════════════════════════════════════════════
# 1) 경로·하이퍼파라미터
# ════════════════════════════════════════════════════════════
csv_path      = "/home/jiwon/project/project/lidc-idri/cam_roi.csv"  # ★ CSV 위치
orig_root     = "/home/jiwon/project/project/lidc-idri/slices"       # 원본 슬라이스 루트
batch_size    = 64
num_epochs    = 150
warmup_epochs = 3
learning_rate = 1e-4
λ_align       = 1.5                              # CAM-Align 가중치

random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════
# 2) 유틸 함수
# ════════════════════════════════════════════════════════════
def apply_window(img, center=-600, width=1500):
    lo, hi = center-width//2, center+width//2
    img = np.clip(img, lo, hi); return (img-lo)/(hi-lo)

def get_score_from_path(path):
    try:    return int(os.path.basename(path).split("_")[-1].split(".")[0])
    except: return -1

def label_from_score(score):       # 0 = 양성(Score 1,2) / 1 = 악성(4,5)
    return 0 if score in [1,2] else 1 if score in [4,5] else -1

# ════════════════════════════════════════════════════════════
# 3) CSV 로드 (path, bbox)
# ════════════════════════════════════════════════════════════
records = []  # [(path, (xmin,ymin,xmax,ymax))]
with open(csv_path, newline="") as f:
    rdr = csv.reader(f)
    for row in rdr:
        # 헤더(첫 행) 또는 빈 행 건너뛰기
        if not row or row[0].lower() in {"path","파일경로"}:
            continue

        path = row[0].strip()
        # 숫자가 아닌 항목(예: 'label')은 제외하고, 뒤에서 4개만 bbox로 사용
        nums = [float(tok) for tok in row[1:] if tok.replace(".","",1).lstrip("-").isdigit()]
        if len(nums) < 4:
            print(f"⚠️ bbox 숫자 부족: {row}"); continue

        xmin, ymin, xmax, ymax = map(int, nums[-4:])  # 뒤 4개
        records.append((path, (xmin, ymin, xmax, ymax)))

# 유효 score만 필터
records = [r for r in records if label_from_score(get_score_from_path(r[0]))!=-1]
print(f"Loaded {len(records)} slices with BBox")

# ════════════════════════════════════════════════════════════
# 4) 모델 정의 (ConvNeXt + CBAM + Non-Local)
# ════════════════════════════════════════════════════════════
class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ch, ch//r), nn.ReLU(), nn.Linear(ch//r, ch))
        self.avgp, self.maxp = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2,1,7,padding=3)
    def forward(self,x):
        b,c,_,_ = x.size()
        ca = torch.sigmoid(self.mlp(self.avgp(x).view(b,c))+self.mlp(self.maxp(x).view(b,c))).view(b,c,1,1)
        x  = x*ca
        sa = torch.sigmoid(self.conv(torch.cat([x.mean(1,True), x.max(1,True)[0]],1)))
        return x*sa

class ConvNeXt_CBAM_NL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0)
        ch = self.backbone.num_features
        self.cbam= CBAM(ch)
        self.gap, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(ch,1)
    def forward(self,x,return_feat=False):
        f = self.cbam(self.backbone.forward_features(x))
        logit = self.fc(self.gap(f).flatten(1)).squeeze(1)
        return (logit,f) if return_feat else logit

# ════════════════════════════════════════════════════════════
# 5) Dataset (원본 + BBox → 마스크)
# ════════════════════════════════════════════════════════════
class BBoxDataset(Dataset):
    def __init__(self, recs, train=True):
        self.recs, self.train = recs, train
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        path, (x1,y1,x2,y2) = self.recs[idx]
        score  = get_score_from_path(path)
        label  = label_from_score(score)
        img_np = np.load(path)                          # 원본 HU
        H,W    = img_np.shape
        # --- BBox → 마스크 ---
        mask_np       = np.zeros_like(img_np, dtype=np.float32)
        x1c,x2c = max(0,x1), min(W,x2); y1c,y2c = max(0,y1), min(H,y2)
        mask_np[y1c:y2c, x1c:x2c] = 1.0

        img  = torch.from_numpy(apply_window(img_np)).unsqueeze(0).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0)

        img  = TF.resize(img ,(224,224),antialias=True)
        mask = TF.resize(mask,(224,224),interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if random.random()<0.5: img,mask = TF.hflip(img),TF.hflip(mask)
            if random.random()<0.5:
                ang = random.uniform(-3,3)
                img  = TF.rotate(img ,ang,interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask,ang,interpolation=TF.InterpolationMode.NEAREST)

        img  = (img-0.5)/0.5
        img  = img.repeat(3,1,1)
        return img, mask, label

# ════════════════════════════════════════════════════════════
# 6) Train/Val split & DataLoader
# ════════════════════════════════════════════════════════════
random.shuffle(records)
split = int(len(records)*0.8)
train_set, val_set = records[:split], records[split:]

print("Label dist  (train):", Counter(label_from_score(get_score_from_path(p)) for p,_ in train_set))
print("Label dist  (val)  :", Counter(label_from_score(get_score_from_path(p)) for p,_ in val_set))

train_loader = DataLoader(BBoxDataset(train_set,True), batch_size, True , num_workers=4, pin_memory=True)
val_loader   = DataLoader(BBoxDataset(val_set ,False), batch_size, False, num_workers=4, pin_memory=True)

# ════════════════════════════════════════════════════════════
# 7) 손실·옵티마이저·스케줄러 (class weight)
# ════════════════════════════════════════════════════════════
# ----- Class Weight -----
labels = [label_from_score(get_score_from_path(p)) for p, _ in train_set]
num_0 = sum(l == 0 for l in labels) # 양성 개수
num_1 = sum(l == 1 for l in labels) # 악성 개수
n_total = num_0 + num_1
weight_for_0 = n_total / (2 * num_0)
weight_for_1 = n_total / (2 * num_1)
print(f"label=0(benign): {num_0}, label=1(malignant): {num_1}")
print(f"Class weights: label=0: {weight_for_0:.2f}, label=1: {weight_for_1:.2f}")

crit_cls, crit_align = nn.BCEWithLogitsLoss(reduction='none'), nn.BCELoss()

model = ConvNeXt_CBAM_NL().to(device)
optim = AdamW(model.parameters(), lr=learning_rate)

cosine = CosineAnnealingLR(optim, T_max=num_epochs-warmup_epochs, eta_min=1e-6)
sched  = GradualWarmupScheduler(optim, 1.0, warmup_epochs, cosine)

# ════════════════════════════════════════════════════════════
# 8) 학습 루프 (class weight 적용)
# ════════════════════════════════════════════════════════════
best_acc, patience, no_imp = 0, 20, 0

for epoch in range(1, num_epochs+1):
    # --- Train ---
    model.train(); tot_loss=cor=tot=0
    for img,mask,label in tqdm(train_loader, desc=f"[Train] {epoch}/{num_epochs}"):
        img,mask,label = img.to(device), mask.to(device), label.to(device)
        logit, feat = model(img, return_feat=True)
        # [핵심] 샘플별 weight 벡터 생성
        weights = torch.where(label == 0, weight_for_0, weight_for_1).to(device)
        cls_loss = crit_cls(logit, label.float())
        cls_loss = (cls_loss * weights).mean()
        # CAM Alignment Loss
        cam = torch.sigmoid(F.interpolate(feat.mean(1,True), mask.shape[-2:], mode='bilinear', align_corners=False))
        aln_loss = crit_align(cam, mask)
        loss = cls_loss + λ_align*aln_loss

        optim.zero_grad(); loss.backward(); optim.step()

        preds = (torch.sigmoid(logit)>0.5).long()
        tot_loss += loss.item()*label.size(0)
        cor += (preds==label).sum().item(); tot += label.size(0)

    print(f"Epoch {epoch:03d} | Train Loss {tot_loss/tot:.4f} | Acc {cor/tot*100:.2f}%")

    # --- Val ---
    model.eval(); v_loss=cor=tot=0
    with torch.no_grad():
        for img, mask, label in val_loader:
            img, label = img.to(device), label.to(device)
            logit = model(img)
            weights = torch.where(label == 0, weight_for_0, weight_for_1).to(device)
            loss = crit_cls(logit, label.float())
            loss = (loss * weights).mean()
            preds = (torch.sigmoid(logit) > 0.5).long()
            v_loss += loss.item() * label.size(0)
            cor    += (preds == label).sum().item(); tot += label.size(0)

    val_acc = cor/tot*100; val_loss = v_loss/tot
    print(f"              Val Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
    sched.step()

    if val_acc>best_acc:
        best_acc,no_imp = val_acc,0
        torch.save(model.state_dict(),"CAM_Best_align_1.5(class_weitgh).pth")
        print(f"  \u2705 New best! ({best_acc:.2f}%)\n")
    else:
        no_imp+=1
        print(f"  ↳ No improve {no_imp}/{patience}\n")
        if no_imp>=patience:
            print(f"Early stop @{epoch} (\u2705Best {best_acc:.2f}%)"); break

# ════════════════════════════════════════════════════════════
# 9) 최종 저장
# ════════════════════════════════════════════════════════════
torch.save(model.state_dict(),"CAM_Best_align_1.5(class_weitgh)final.pth")
print("Finished training. Final model saved.")
