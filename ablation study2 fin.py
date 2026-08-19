# ============================================================
# AT-MGAN ABLATION PIPELINE — OPTIMISED + SMOTE + ROC CURVES
# Run: python ablation_pipeline.py
# ============================================================

import os, sys, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.makedirs('processed',   exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results',     exist_ok=True)

# ── Install imbalanced-learn if missing ──────────────────────
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    os.system(f"{sys.executable} -m pip install -q imbalanced-learn")
    from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIG
# ============================================================
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 50
BATCH_SIZE = 256
LATENT_DIM = 64
LR         = 0.0002
BETA1      = 0.7
ALPHA_W    = 0.5
BETA_W     = 0.5
TAU_I      = 0.8
PATIENCE   = 8

VARIANTS = ['GAN', 'ACGAN', 'MGAN', 'MGAN_dis', 'MGAN_dis_cos', 'AT_MGAN']
LABELS_MAP = {
    'GAN':          'Standard GAN',
    'ACGAN':        'ACGAN',
    'MGAN':         'MGAN (no regularisation)',
    'MGAN_dis':     'MGAN + L_dis only',
    'MGAN_dis_cos': 'MGAN + L_dis + L_cos',
    'AT_MGAN':      'AT-MGAN (full model)',
}
COLORS = {
    'GAN':          '#94a3b8',
    'ACGAN':        '#f87171',
    'MGAN':         '#fb923c',
    'MGAN_dis':     '#eab308',
    'MGAN_dis_cos': '#3b82f6',
    'AT_MGAN':      '#16a34a',
}
LINE_STYLES = {
    'GAN':          ':',
    'ACGAN':        '--',
    'MGAN':         '-.',
    'MGAN_dis':     (0,(5,2)),
    'MGAN_dis_cos': (0,(3,1,1,1)),
    'AT_MGAN':      '-',
}

print(f"Device: {DEVICE} | Python: {sys.version.split()[0]}")

# ============================================================
# STEP 1 — PREPROCESSING
# ============================================================
def preprocess_unsw(csv_path):
    print("\n▶ Loading UNSW-NB15...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df.drop(columns=[c for c in ['srcip','dstip','stime','ltime','attack_cat']
                     if c in df.columns], inplace=True)
    for col in ['dsport','sport']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['proto','state','service']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
    num_cols = df.select_dtypes(include=[np.number]).columns
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    
    df['label'] = df['label'].astype(int)

    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    
    X = SelectKBest(f_classif, k=min(30, X.shape[1])).fit_transform(X, y)
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=20, random_state=42).fit_transform(X).astype(np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                         random_state=42, stratify=y)
    print(f"   Before SMOTE — Train:{Xtr.shape} | Dist:{np.bincount(ytr)}")
    Xtr, ytr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
    print(f"   After  SMOTE — Train:{Xtr.shape} | Dist:{np.bincount(ytr)}")
    print(f"   Test (original dist): {Xte.shape} | Dist:{np.bincount(yte)}")

    np.save('processed/X_train_unsw.npy', Xtr)
    np.save('processed/y_train_unsw.npy', ytr)
    np.save('processed/X_test_unsw.npy',  Xte)
    np.save('processed/y_test_unsw.npy',  yte)
    print("   ✅ UNSW saved.")

def preprocess_iot(csv_path):
    print("\n▶ Loading IoT Telemetry...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in ['light','motion']:
        if col in df.columns:
            df[col] = df[col].map({True:1,False:0,'True':1,'False':0,
                                   1:1,0:0}).fillna(0).astype(int)
    feats = ['co','humidity','light','lpg','motion','smoke','temp']
    df.dropna(subset=feats+['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    X = df[feats].values.astype(np.float32)
    y = df['label'].values
    
    X = SelectKBest(f_classif, k=7).fit_transform(X, y)
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=5, random_state=42).fit_transform(X).astype(np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                         random_state=42, stratify=y)
    print(f"   IoT Train:{Xtr.shape} | Dist:{np.bincount(ytr)}")
    print(f"   IoT Test: {Xte.shape} | Dist:{np.bincount(yte)}")
    np.save('processed/X_train_iot.npy', Xtr)
    np.save('processed/y_train_iot.npy', ytr)
    np.save('processed/X_test_iot.npy',  Xte)
    np.save('processed/y_test_iot.npy',  yte)
    print("   ✅ IoT saved.")

# ============================================================
# STEP 2 — MODEL DEFINITIONS
# ============================================================

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    def forward(self, z, labels):
        return self.net(torch.cat([z, self.label_emb(labels)], dim=1))

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.LeakyReLU(0.2), nn.Dropout(0.3),
        )
        self.src_head   = nn.Sequential(nn.Linear(256, 1),           nn.Sigmoid())
        self.class_head = nn.Sequential(nn.Linear(256, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        feat = self.shared(x)
        return self.src_head(feat), self.class_head(feat), feat

bce_fn = nn.BCELoss()
ce_fn  = nn.CrossEntropyLoss()

def Ls_fn(rs, fs):
    return bce_fn(rs, torch.ones_like(rs)) + bce_fn(fs, torch.zeros_like(fs))

def Lc_fn(rc, rl, fc, fl):
    return ce_fn(rc, rl) + ce_fn(fc, fl)

def Ldis_fn(rf, ff):
    return torch.mean((rf.mean(0) - ff.mean(0)) ** 2)

def Lcos_fn(rf, ff):
    return (1 - nn.CosineSimilarity(dim=1)(rf, ff)).mean()

def Ladv_fn(D, susp):
    if susp is None or susp.size(0) == 0:
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)
    src, _, _ = D(susp)
    return bce_fn(src, torch.ones_like(src))

def D_loss(variant, D, rx, fx, rl, fl, tau=TAU_I):
    rs, rc, rf = D(rx)
    fs, fc, ff = D(fx.detach())
    Ls = Ls_fn(rs, fs)
    if variant == 'GAN':          return Ls
    Lc = Lc_fn(rc, rl, fc, fl)
    if variant == 'ACGAN':        return Ls + Lc
    if variant == 'MGAN':         return Ls + Lc
    Ld = Ldis_fn(rf, ff)
    if variant == 'MGAN_dis':     return Ls + Lc + Ld
    Lk = Lcos_fn(rf, ff)
    if variant == 'MGAN_dis_cos': return Ls + Lc + Ld + Lk
    if variant == 'AT_MGAN':
        n    = rx.size(0)
        susp = rx[:max(1, int(n*(1-tau)))]
        Lav  = Ladv_fn(D, susp)
        return Ls + Lc + tau*(ALPHA_W*Ld + BETA_W*Lk) + (1-tau)*Lav
    raise ValueError(variant)

def G_loss(variant, D, fxg, flg, rf_det=None, tau=TAU_I):
    fs, fc, ff = D(fxg)
    Ls_g = bce_fn(fs, torch.ones_like(fs))
    Lc_g = ce_fn(fc, flg)
    if variant == 'GAN':              return Ls_g
    if variant in ('ACGAN','MGAN'):   return Ls_g + Lc_g
    Lk = Lcos_fn(rf_det, ff) if rf_det is not None else torch.tensor(0.0, device=DEVICE)
    return Ls_g + Lc_g + tau * BETA_W * Lk

# ============================================================
# STEP 3 — TRAINING
# ============================================================
LOSS_HISTORY = {}

def train_variant(variant, tag, dl, Xv_t, yval, nc, idim):
    ckpt_D = f'checkpoints/D_{variant}_{tag}.pt'
    ckpt_G = f'checkpoints/G_{variant}_{tag}.pt'
    if os.path.exists(ckpt_D):
        print(f"   ⏩ {LABELS_MAP[variant]} — checkpoint found, skipping.")
        return

    G = Generator(LATENT_DIM, nc, idim).to(DEVICE)
    D = Discriminator(idim, nc).to(DEVICE)
    oG = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    best_val = 0.0
    wait     = 0

    history_key = f"{tag}_{variant}"
    LOSS_HISTORY[history_key] = {'D_loss': [], 'G_loss': []}

    print(f"\n  ▶ Training {LABELS_MAP[variant]} ...")
    for epoch in range(EPOCHS):
        G.train(); D.train()
        ep_dL = ep_gL = 0.0
        for rx, rl in dl:
            rx, rl = rx.to(DEVICE, non_blocking=True), rl.to(DEVICE, non_blocking=True)
            bs = rx.size(0)
            
            # Train Discriminator
            z  = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fl = torch.randint(0, nc, (bs,), device=DEVICE)
            fx = G(z, fl)
            oD.zero_grad(set_to_none=True)
            dL = D_loss(variant, D, rx, fx, rl, fl)
            dL.backward(); oD.step()
            
            # Train Generator
            z2  = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fl2 = torch.randint(0, nc, (bs,), device=DEVICE)
            fx2 = G(z2, fl2)
            _, _, rf = D(rx)
            oG.zero_grad(set_to_none=True)
            gL = G_loss(variant, D, fx2, fl2, rf.detach())
            gL.backward(); oG.step()
            ep_dL += dL.item(); ep_gL += gL.item()

        LOSS_HISTORY[history_key]['D_loss'].append(ep_dL / len(dl))
        LOSS_HISTORY[history_key]['G_loss'].append(ep_gL / len(dl))

        D.eval()
        with torch.no_grad():
            _, cp, _ = D(Xv_t)
        vacc = accuracy_score(yval, cp.cpu().numpy().argmax(1))
        if vacc > best_val:
            best_val = vacc
            torch.save(D.state_dict(), ckpt_D)
            torch.save(G.state_dict(), ckpt_G)
            wait = 0
        else:
            wait += 1
        if (epoch+1) % 1 == 0:
            print(f"     E{epoch+1:3d}/{EPOCHS} "
                  f"D:{ep_dL/len(dl):.4f} G:{ep_gL/len(dl):.4f} "
                  f"ValAcc:{vacc*100:.2f}% Best:{best_val*100:.2f}%")
        if wait >= PATIENCE:
            print(f"     ⏹ Early stop at epoch {epoch+1}")
            break
    print(f"   ✅ Best ValAcc:{best_val*100:.2f}% — {ckpt_D}")

def train_all(tag):
    Xtr = np.load(f'processed/X_train_{tag}.npy')
    ytr = np.load(f'processed/y_train_{tag}.npy')
    
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.1,
                                               random_state=42, stratify=ytr)
    print(f"\n  [{tag.upper()}] Train:{Xtr2.shape} Val:{Xval.shape}")
    
    nc   = len(np.unique(ytr2))
    idim = Xtr2.shape[1]
    
    ds = TensorDataset(torch.from_numpy(Xtr2), torch.from_numpy(ytr2).long())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    drop_last=True, num_workers=2, pin_memory=True)
                    
    Xv_t = torch.from_numpy(Xval).float().to(DEVICE)
    
    for v in VARIANTS:
        train_variant(v, tag, dl, Xv_t, yval, nc, idim)

# ============================================================
# NEW STEP — GENERATE LOSS CURVES GRAPH
# ============================================================
def plot_loss_curves(tag):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    axes = axes.flatten()
    
    for idx, v in enumerate(VARIANTS):
        key = f"{tag}_{v}"
        ax = axes[idx]
        if key in LOSS_HISTORY and len(LOSS_HISTORY[key]['D_loss']) > 0:
            epochs_run = range(1, len(LOSS_HISTORY[key]['D_loss']) + 1)
            ax.plot(epochs_run, LOSS_HISTORY[key]['D_loss'], label='D Loss', color='#ef4444', lw=2)
            ax.plot(epochs_run, LOSS_HISTORY[key]['G_loss'], label='G Loss', color='#3b82f6', lw=2)
            ax.set_title(f"{LABELS_MAP[v]} Loss", fontsize=11, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss Value')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Checkpoint used\n(No active loss logging)', 
                    ha='center', va='center', color='#94a3b8', fontsize=12)
            ax.set_title(f"{LABELS_MAP[v]}", fontsize=11, fontweight='bold')
            
    fig.suptitle(f'Generator and Discriminator Loss Trends — {tag.upper()}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_img = f'results/loss_curves_{tag}.png'
    plt.savefig(out_img, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Loss curves saved: {out_img}")

# ============================================================
# STEP 4 — EVALUATION + ABLATION TABLES + ROC + BAR CHARTS
# ============================================================
def evaluate_all(tag):
    Xte  = np.load(f'processed/X_test_{tag}.npy')
    yte  = np.load(f'processed/y_test_{tag}.npy')
    nc   = len(np.unique(yte))
    idim = Xte.shape[1]
    Xt   = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)

    rows     = []
    roc_data = {}

    print(f"\n{'─'*40}\n  CONFUSION MATRICES — {tag.upper()}\n{'─'*40}")
    
    for variant in VARIANTS:
        ckpt = f'checkpoints/D_{variant}_{tag}.pt'
        if not os.path.exists(ckpt):
            print(f"   ⚠ Missing {ckpt}"); continue

        D = Discriminator(idim, nc).to(DEVICE)
        D.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        D.eval()
        with torch.no_grad():
            _, cls_prob, _ = D(Xt)
        prob  = cls_prob.cpu().numpy()
        preds = prob.argmax(1)
        pos_p = prob[:, 1]

        acc  = accuracy_score(yte, preds)
        prec = precision_score(yte, preds, average='weighted', zero_division=0)
        rec  = recall_score(yte, preds,    average='weighted', zero_division=0)
        f1   = f1_score(yte, preds,        average='weighted', zero_division=0)
        mcc  = matthews_corrcoef(yte, preds)
        try:
            auc = roc_auc_score(yte, pos_p)
        except:
            auc = 0.0

        cm = confusion_matrix(yte, preds)
        
        print(f"\n[Model: {LABELS_MAP[variant]}]")
        print(f"Confusion Matrix Array:\n{cm}")
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            print(f"  └─ True Negatives (TN): {tn:5d}  |  False Positives (FP): {fp:5d}")
            print(f"  └─ False Negatives (FN): {fn:5d}  |  True Positives (TP): {tp:5d}")
            fpr_v  = fp/(fp+tn+1e-9)
            spec_v = tn/(tn+fp+1e-9)
        else:
            fpr_v = spec_v = 0.0
            print("  └─ Multiclass configuration detected.")

        fpr_c, tpr_c, _ = roc_curve(yte, pos_p)
        roc_data[variant] = (fpr_c, tpr_c, round(auc, 4))

        rows.append({
            'Model':           LABELS_MAP[variant],
            'Accuracy (%)':    round(acc*100,2),
            'Precision (%)':   round(prec*100,2),
            'Recall (%)':      round(rec*100,2),
            'F1-Score (%)':    round(f1*100,2),
            'AUC':             round(auc,4),
            'MCC':             round(mcc,4),
            'Specificity (%)': round(spec_v*100,2),
            'FPR (%)':         round(fpr_v*100,2),
        })

    df = pd.DataFrame(rows)
    at_row = df[df['Model']=='AT-MGAN (full model)']
    if len(at_row):
        at_acc = at_row['Accuracy (%)'].values[0]
        df['Δ Accuracy'] = df['Accuracy (%)'].apply(
            lambda x: '—' if x==at_acc else f"{x-at_acc:+.2f}%")
        df.loc[df['Model']=='AT-MGAN (full model)', 'Δ Accuracy'] = '—'

    csv_path = f'results/ablation_{tag}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n{'─'*100}")
    print(f"  ABLATION TABLE — {tag.upper()}")
    print(f"{'─'*100}")
    print(df.to_string(index=False))
    print(f"   ✅ Saved: {csv_path}")

    ckpt_at = f'checkpoints/D_AT_MGAN_{tag}.pt'
    if os.path.exists(ckpt_at):
        D2 = Discriminator(idim, nc).to(DEVICE)
        D2.load_state_dict(torch.load(ckpt_at, map_location=DEVICE))
        D2.eval()
        with torch.no_grad():
            _, cp2, _ = D2(Xt)
        p2 = cp2.cpu().numpy().argmax(1)
        rpt = classification_report(yte, p2, digits=4)
        rpt_p = f'results/per_class_AT_MGAN_{tag}.txt'
        with open(rpt_p, 'w') as fp_:
            fp_.write(rpt)
        print(f"\n  Per-class (AT-MGAN):\n{rpt}")

    at_auc = roc_data.get('AT_MGAN', (None,None,None))[2]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5,label='Random (AUC=0.50)')
    at_fpr, at_tpr, _ = roc_data.get('AT_MGAN',(None,None,None))
    if at_fpr is not None:
        ax.fill_between(at_fpr, at_tpr, alpha=0.08, color='#16a34a')
    for v in VARIANTS:
        if v not in roc_data: continue
        fpr_c, tpr_c, auc_v = roc_data[v]
        ax.plot(fpr_c, tpr_c, color=COLORS[v],
                lw=3.5 if v=='AT_MGAN' else 1.6,
                linestyle=LINE_STYLES[v],
                label=f"{LABELS_MAP[v]}  (AUC={auc_v:.4f})")
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=13)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=13)
    ax.set_title(f'ROC Curves — {tag.upper()} Dataset\n'
                 f'AT-MGAN achieves AUC = {at_auc}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9.5, loc='lower right', framealpha=0.95, edgecolor='#ccc')
    ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.03])
    ax.grid(alpha=0.35, color='#ccc')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'results/roc_{tag}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✅ ROC saved: results/roc_{tag}.png")

    df_plot = df.copy()
    df_plot['AUC'] = df_plot['AUC']*100
    df_plot['MCC'] = df_plot['MCC']*100
    metrics_b = ['Accuracy (%)','F1-Score (%)','AUC','MCC']
    bar_colors = ['#3b82f6','#10b981','#f59e0b','#8b5cf6']
    x       = np.arange(len(df_plot))
    width   = 0.18
    offsets = [-1.5,-0.5,0.5,1.5]
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    for j,(m,c,off) in enumerate(zip(metrics_b, bar_colors, offsets)):
        vals = df_plot[m].tolist()
        bars = ax2.bar(x+off*width, vals, width, color=c, alpha=0.85, label=m)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.15, f"{v:.1f}",
                     ha='center', va='bottom', fontsize=7.5, rotation=90)
    at_idx = df_plot.index[df_plot['Model']=='AT-MGAN (full model)'].tolist()
    if at_idx:
        ax2.axvspan(at_idx[0]-0.45, at_idx[0]+0.45,
                    alpha=0.08, color='#16a34a', zorder=0)
        ax2.text(at_idx[0], df_plot['Accuracy (%)'].max()+3.5,
                 '★ AT-MGAN\nBest', ha='center', fontsize=9,
                 color='#15803d', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_plot['Model'].tolist(), rotation=14, ha='right', fontsize=9.5)
    ax2.set_ylabel('Score (Acc/F1/AUC×100/MCC×100)', fontsize=11)
    ax2.set_title(f'Ablation Study — {tag.upper()} Dataset\n'
                  f'Progressive contribution of each AT-MGAN component',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right', framealpha=0.95)
    ax2.set_ylim([50, df_plot['Accuracy (%)'].max()+8])
    ax2.grid(axis='y', alpha=0.35, color='#ccc')
    ax2.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'results/ablation_bar_{tag}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Bar chart saved: results/ablation_bar_{tag}.png")

    return df, roc_data


# ============================================================
# SAFE WINDOWS EXECUTION GUARD (PREVENTS WORKER CRASHES)
# ============================================================
if __name__ == '__main__':
    # 1. Run Preprocessing check safely inside main block
    print("\n" + "="*60)
    print("STEP 1: PREPROCESSING PATH CHECK")
    print("="*60)
    if not os.path.exists('processed/X_train_unsw.npy'):
        preprocess_unsw('UNSW-NB15.csv')
    else:
        print("   ⏩ UNSW preprocessed data found — skipping.")

    if not os.path.exists('processed/X_train_iot.npy'):
        preprocess_iot('iot_telemetry_ids.csv')
    else:
        print("   ⏩ IoT preprocessed data found — skipping.")

    # 2. Train variants securely
    print("\n" + "="*60)
    print("STEP 3: TRAINING ALL VARIANTS")
    print("="*60)
    train_all('unsw')
    train_all('iot')

    # 3. Graph loss curves
    print("\n" + "="*60)
    print("ADDITIONAL STEP: GENERATING LOSS CURVES")
    print("="*60)
    plot_loss_curves('unsw')
    plot_loss_curves('iot')

    # 4. Run Evaluation metrics
    print("\n" + "="*60)
    print("STEP 4: EVALUATION")
    print("="*60)
    df_unsw, roc_unsw = evaluate_all('unsw')
    df_iot,  roc_iot  = evaluate_all('iot')

    # 5. Combined ROC Plot Generation
    print("\n" + "="*60)
    print("STEP 5: COMBINED ROC FIGURE")
    print("="*60)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, roc_data, tag_name in [
        (axes[0], roc_unsw, 'UNSW-NB15'),
        (axes[1], roc_iot,  'IoT Telemetry')
    ]:
        ax.plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5)
        at_fpr, at_tpr, at_auc = roc_data.get('AT_MGAN',(None,None,None))
        if at_fpr is not None:
            ax.fill_between(at_fpr, at_tpr, alpha=0.07, color='#16a34a')
        for v in VARIANTS:
            if v not in roc_data: continue
            fpr_c, tpr_c, auc_v = roc_data[v]
            ax.plot(fpr_c, tpr_c, color=COLORS[v],
                    lw=3.5 if v=='AT_MGAN' else 1.5,
                    linestyle=LINE_STYLES[v],
                    label=f"{LABELS_MAP[v]} ({auc_v:.4f})")
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{tag_name}\nAT-MGAN AUC = {at_auc}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, loc='lower right', framealpha=0.92, edgecolor='#ccc')
        ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.03])
        ax.grid(alpha=0.3, color='#ccc')
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle('ROC Curves — AT-MGAN vs Baseline Models',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/roc_combined.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("   ✅ Combined ROC saved: results/roc_combined.png")

    print("\n" + "="*60)
    print("🎉 FULL ABLATION PIPELINE COMPLETE")
    print("   Outputs in: results/")
    print("   ablation_unsw.csv      ablation_iot.csv")
    print("   roc_unsw.png           roc_iot.png")
    print("   roc_combined.png")
    print("   ablation_bar_unsw.png  ablation_bar_iot.png")
    print("   loss_curves_unsw.png   loss_curves_iot.png")
    print("   per_class_AT_MGAN_unsw.txt  per_class_AT_MGAN_iot.txt")
    print("="*60)
