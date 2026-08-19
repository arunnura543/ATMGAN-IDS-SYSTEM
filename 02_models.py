import torch
import torch.nn as nn
import numpy as np

# ═══════════════════════════════════════════════════════════
# SHARED GENERATOR
# ═══════════════════════════════════════════════════════════
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
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# SHARED DISCRIMINATOR
# ═══════════════════════════════════════════════════════════
class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
        )
        self.src_head   = nn.Sequential(nn.Linear(256, 1),  nn.Sigmoid())
        self.class_head = nn.Sequential(nn.Linear(256, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        feat  = self.shared(x)
        src   = self.src_head(feat)
        cls   = self.class_head(feat)
        return src, cls, feat   # feat returned for feature-matching loss


# ═══════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════
def source_loss(real_src, fake_src):
    bce = nn.BCELoss()
    ones  = torch.ones_like(real_src)
    zeros = torch.zeros_like(fake_src)
    return bce(real_src, ones) + bce(fake_src, zeros)

def class_loss(real_cls, real_labels, fake_cls, fake_labels):
    ce = nn.CrossEntropyLoss()
    return ce(real_cls, real_labels) + ce(fake_cls, fake_labels)

def feature_dist_loss(real_feat, fake_feat):
    return torch.mean((real_feat.mean(0) - fake_feat.mean(0)) ** 2)

def cosine_loss(real_feat, fake_feat):
    cos = nn.CosineSimilarity(dim=1)
    sim = cos(real_feat, fake_feat)
    return (1 - sim).mean()

def adversarial_loss(discriminator, suspicious_data, tau_i=0.0):
    """Ladv: amplified adversarial loss for low-trust traffic."""
    bce = nn.BCELoss()
    if suspicious_data is None or len(suspicious_data) == 0:
        return torch.tensor(0.0, requires_grad=True)
    src, _, _ = discriminator(suspicious_data)
    target = torch.ones_like(src)
    return bce(src, target) * (1 - tau_i)


# ═══════════════════════════════════════════════════════════
# COMBINED LOSS PER MODEL VARIANT
# ═══════════════════════════════════════════════════════════
def compute_discriminator_loss(variant, D, real_x, fake_x,
                                real_labels, fake_labels,
                                tau_i=1.0, alpha=0.5, beta=0.5):
    """
    variant: 'GAN' | 'ACGAN' | 'MGAN' | 'MGAN_dis' | 'MGAN_dis_cos' | 'AT_MGAN'
    """
    real_src, real_cls, real_feat = D(real_x)
    fake_src, fake_cls, fake_feat = D(fake_x.detach())

    Ls = source_loss(real_src, fake_src)

    if variant == 'GAN':
        return Ls

    Lc = class_loss(real_cls, real_labels, fake_cls, fake_labels)

    if variant == 'ACGAN':
        return Ls + Lc

    if variant == 'MGAN':
        return Ls + Lc   # no regularisation

    Ldis = feature_dist_loss(real_feat, fake_feat)

    if variant == 'MGAN_dis':
        return Ls + Lc + Ldis

    Lcos = cosine_loss(real_feat, fake_feat)

    if variant == 'MGAN_dis_cos':
        return Ls + Lc + Ldis + Lcos

    if variant == 'AT_MGAN':
        # suspicious = low-trust samples (simulated here as bottom 20% by index)
        n = real_x.size(0)
        suspicious = real_x[:max(1, int(n * (1 - tau_i)))]
        Ladv = adversarial_loss(D, suspicious, tau_i)
        L_AT = Ls + Lc + tau_i * (alpha * Ldis + beta * Lcos) + (1 - tau_i) * Ladv
        return L_AT

    raise ValueError(f"Unknown variant: {variant}")


def compute_generator_loss(variant, D, fake_x, fake_labels, real_feat=None,
                            tau_i=1.0, beta=0.5):
    fake_src, fake_cls, fake_feat = D(fake_x)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    Ls_g = bce(fake_src, torch.ones_like(fake_src))  # fool discriminator
    Lc_g = ce(fake_cls, fake_labels)

    if variant == 'GAN':
        return Ls_g

    if variant in ('ACGAN', 'MGAN'):
        return Ls_g + Lc_g

    if real_feat is not None:
        Lcos = cosine_loss(real_feat, fake_feat)
    else:
        Lcos = torch.tensor(0.0)

    if variant == 'MGAN_dis':
        return Ls_g + Lc_g

    if variant in ('MGAN_dis_cos', 'AT_MGAN'):
        return Ls_g + Lc_g + tau_i * beta * Lcos

    raise ValueError(f"Unknown variant: {variant}")
