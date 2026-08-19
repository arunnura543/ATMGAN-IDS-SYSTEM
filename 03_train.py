import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os, json

from models import Generator, Discriminator
from models import compute_discriminator_loss, compute_generator_loss

os.makedirs('checkpoints', exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 60
BATCH_SIZE = 128
LATENT_DIM = 64
LR         = 0.0002
BETA1      = 0.7
ALPHA      = 0.5
BETA       = 0.5
TAU_I      = 0.8   # Simulated trust score (clean training scenario)

VARIANTS = ['GAN', 'ACGAN', 'MGAN', 'MGAN_dis', 'MGAN_dis_cos', 'AT_MGAN']

def train_variant(variant, X_train, y_train, input_dim, num_classes, tag):
    G = Generator(LATENT_DIM, num_classes, input_dim).to(DEVICE)
    D = Discriminator(input_dim, num_classes).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    for epoch in range(EPOCHS):
        for real_x, real_labels in loader:
            real_x      = real_x.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            bs          = real_x.size(0)

            # ── Sample latent noise ──────────────────────────
            z           = torch.randn(bs, LATENT_DIM).to(DEVICE)
            fake_labels = torch.randint(0, num_classes, (bs,)).to(DEVICE)
            fake_x      = G(z, fake_labels)

            # ── Discriminator update ─────────────────────────
            opt_D.zero_grad()
            D_loss = compute_discriminator_loss(
                variant, D, real_x, fake_x, real_labels, fake_labels,
                tau_i=TAU_I, alpha=ALPHA, beta=BETA
            )
            D_loss.backward()
            opt_D.step()

            # ── Generator update ─────────────────────────────
            z2          = torch.randn(bs, LATENT_DIM).to(DEVICE)
            fake_labels2 = torch.randint(0, num_classes, (bs,)).to(DEVICE)
            fake_x2     = G(z2, fake_labels2)

            _, _, real_feat = D(real_x)

            opt_G.zero_grad()
            G_loss = compute_generator_loss(
                variant, D, fake_x2, fake_labels2,
                real_feat=real_feat.detach(), tau_i=TAU_I, beta=BETA
            )
            G_loss.backward()
            opt_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [{variant}] Epoch {epoch+1}/{EPOCHS} | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")

    torch.save(D.state_dict(), f'checkpoints/D_{variant}_{tag}.pt')
    torch.save(G.state_dict(), f'checkpoints/G_{variant}_{tag}.pt')
    return D, G


def run_all(dataset_tag):
    X = np.load(f'processed/X_{dataset_tag}.npy')
    y = np.load(f'processed/y_{dataset_tag}.npy')
    num_classes = len(np.unique(y))
    input_dim   = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save(f'processed/X_test_{dataset_tag}.npy', X_test)
    np.save(f'processed/y_test_{dataset_tag}.npy', y_test)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_tag.upper()} | Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"{'='*60}")

    for variant in VARIANTS:
        print(f"\n▶ Training {variant} on {dataset_tag}...")
        train_variant(variant, X_train, y_train, input_dim, num_classes, dataset_tag)

    print(f"\n✅ All variants trained for {dataset_tag}")


if __name__ == '__main__':
    run_all('unsw')
    run_all('iot')
