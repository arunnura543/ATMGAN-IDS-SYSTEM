import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from torch.utils.data import TensorDataset, DataLoader
import os, json

from models import Generator, Discriminator

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
VARIANTS   = ['GAN', 'ACGAN', 'MGAN', 'MGAN_dis', 'MGAN_dis_cos', 'AT_MGAN']
LABELS_MAP = {
    'GAN':          'Standard GAN',
    'ACGAN':        'ACGAN',
    'MGAN':         'MGAN (no regularisation)',
    'MGAN_dis':     'MGAN + L_dis only',
    'MGAN_dis_cos': 'MGAN + L_dis + L_cos',
    'AT_MGAN':      'AT-MGAN (full model)',
}

os.makedirs('results', exist_ok=True)


def evaluate_discriminator(D, X_test, y_test, num_classes, variant, tag):
    D.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():
        src, cls, _ = D(X_t)

    # Use class-head predictions for multi-class accuracy
    preds_prob = cls.cpu().numpy()
    preds      = np.argmax(preds_prob, axis=1)

    # Binary: real(1) vs fake(0) from source head
    src_prob = src.cpu().numpy().flatten()
    src_pred = (src_prob >= 0.5).astype(int)

    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec   = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1    = f1_score(y_test, preds, average='weighted', zero_division=0)
    mcc   = matthews_corrcoef(y_test, preds)

    # AUC (binary or multiclass)
    if num_classes == 2:
        auc = roc_auc_score(y_test, preds_prob[:, 1])
    else:
        try:
            auc = roc_auc_score(y_test, preds_prob, multi_class='ovr', average='weighted')
        except Exception:
            auc = 0.0

    # FPR from confusion matrix (binary approximation)
    cm  = confusion_matrix(y_test, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        fpr  = 0.0
        spec = 0.0

    result = {
        'Model':        LABELS_MAP[variant],
        'Accuracy (%)': round(acc * 100, 2),
        'Precision (%)':round(prec * 100, 2),
        'Recall (%)':   round(rec * 100, 2),
        'F1-Score (%)': round(f1 * 100, 2),
        'AUC':          round(auc, 4),
        'MCC':          round(mcc, 4),
        'Specificity (%)': round(spec * 100, 2),
        'FPR (%)':      round(fpr * 100, 2),
        'Δ vs AT-MGAN': '—',
    }
    return result


def run_evaluation(tag):
    X_test = np.load(f'processed/X_test_{tag}.npy')
    y_test = np.load(f'processed/y_test_{tag}.npy')
    num_classes = len(np.unique(y_test))
    input_dim   = X_test.shape[1]

    print(f"\n{'='*60}")
    print(f"Evaluating: {tag.upper()} | Test samples: {X_test.shape[0]}")
    print(f"{'='*60}")

    rows = []
    for variant in VARIANTS:
        ckpt = f'checkpoints/D_{variant}_{tag}.pt'
        if not os.path.exists(ckpt):
            print(f"  ⚠ Checkpoint not found: {ckpt} — skipping")
            continue

        D = Discriminator(input_dim, num_classes).to(DEVICE)
        D.load_state_dict(torch.load(ckpt, map_location=DEVICE))

        row = evaluate_discriminator(D, X_test, y_test, num_classes, variant, tag)
        rows.append(row)
        print(f"  ✓ {row['Model']} | Acc: {row['Accuracy (%)']}% | F1: {row['F1-Score (%)']}% | AUC: {row['AUC']}")

    # ── Compute Δ vs AT-MGAN ──────────────────────────────
    df = pd.DataFrame(rows)
    if 'AT-MGAN (full model)' in df['Model'].values:
        at_acc = df.loc[df['Model'] == 'AT-MGAN (full model)', 'Accuracy (%)'].values[0]
        df['Δ vs AT-MGAN'] = df['Accuracy (%)'].apply(
            lambda x: f"{round(x - at_acc, 2):+.2f}%"
        )
        df.loc[df['Model'] == 'AT-MGAN (full model)', 'Δ vs AT-MGAN'] = '—'

    # ── Save results ──────────────────────────────────────
    df.to_csv(f'results/ablation_{tag}.csv', index=False)

    print(f"\n{'─'*80}")
    print(f"ABLATION TABLE — {tag.upper()}")
    print(df.to_string(index=False))
    print(f"{'─'*80}")
    print(f"✅ Saved: results/ablation_{tag}.csv")

    # ── Per-class report for AT-MGAN ─────────────────────
    D_best = Discriminator(input_dim, num_classes).to(DEVICE)
    ckpt_best = f'checkpoints/D_AT_MGAN_{tag}.pt'
    if os.path.exists(ckpt_best):
        D_best.load_state_dict(torch.load(ckpt_best, map_location=DEVICE))
        D_best.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            _, cls, _ = D_best(X_t)
        preds = np.argmax(cls.cpu().numpy(), axis=1)
        report = classification_report(y_test, preds, digits=4)
        with open(f'results/per_class_AT_MGAN_{tag}.txt', 'w') as f:
            f.write(report)
        print(f"\nPer-class report saved: results/per_class_AT_MGAN_{tag}.txt")


if __name__ == '__main__':
    run_evaluation('unsw')
    run_evaluation('iot')
