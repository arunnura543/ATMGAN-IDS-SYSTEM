import pandas as pd
import numpy as np
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

os.makedirs('processed', exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. UNSW-NB15 PREPROCESSING
# ═══════════════════════════════════════════════════════════
unsw = pd.read_csv('UNSW-NB15.csv')
unsw.columns = unsw.columns.str.strip()

# Drop non-feature columns
drop_cols = ['srcip', 'dstip', 'stime', 'ltime', 'attack_cat']
unsw.drop(columns=[c for c in drop_cols if c in unsw.columns], inplace=True)

# Fix mixed-type port columns
unsw['dsport'] = pd.to_numeric(unsw['dsport'], errors='coerce')
unsw['sport']  = pd.to_numeric(unsw['sport'],  errors='coerce')

# Label encode categorical columns
for col in ['proto', 'state', 'service']:
    if col in unsw.columns:
        unsw[col] = LabelEncoder().fit_transform(unsw[col].astype(str))

# Fill NaN with median for numeric columns
for col in unsw.select_dtypes(include=[np.number]).columns:
    unsw[col].fillna(unsw[col].median(), inplace=True)

unsw['label'] = unsw['label'].astype(int)

X_unsw = unsw.drop('label', axis=1).values.astype(np.float32)
y_unsw = unsw['label'].values
print(f"UNSW raw shape: {X_unsw.shape}, classes: {np.bincount(y_unsw)}")

# ── Feature Selection: Top 30 via Mutual Information ──────
selector_unsw = SelectKBest(mutual_info_classif, k=30)
X_unsw_sel = selector_unsw.fit_transform(X_unsw, y_unsw)
print(f"UNSW after feature selection: {X_unsw_sel.shape}")

# ── Feature Extraction: PCA to 20 components ──────────────
scaler_unsw = StandardScaler()
X_unsw_scaled = scaler_unsw.fit_transform(X_unsw_sel)

pca_unsw = PCA(n_components=20, random_state=42)
X_unsw_pca = pca_unsw.fit_transform(X_unsw_scaled).astype(np.float32)
print(f"UNSW after PCA: {X_unsw_pca.shape}")
print(f"Explained variance: {pca_unsw.explained_variance_ratio_.sum():.4f}")

np.save('processed/X_unsw.npy', X_unsw_pca)
np.save('processed/y_unsw.npy', y_unsw)

# ═══════════════════════════════════════════════════════════
# 2. IoT TELEMETRY PREPROCESSING
# ═══════════════════════════════════════════════════════════
iot = pd.read_csv('iot_telemetry_ids.csv')
iot.columns = iot.columns.str.strip()

# Convert boolean columns
iot['light']  = iot['light'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
iot['motion'] = iot['motion'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

feat_iot = ['co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp']
iot.dropna(subset=feat_iot + ['label'], inplace=True)
iot['label'] = iot['label'].astype(int)

X_iot = iot[feat_iot].values.astype(np.float32)
y_iot  = iot['label'].values
print(f"\nIoT raw shape: {X_iot.shape}, classes: {np.bincount(y_iot)}")

# ── Feature Selection: All 7 (small feature set) ──────────
selector_iot = SelectKBest(mutual_info_classif, k=7)
X_iot_sel = selector_iot.fit_transform(X_iot, y_iot)

# ── Feature Extraction: PCA to 5 components ───────────────
scaler_iot = StandardScaler()
X_iot_scaled = scaler_iot.fit_transform(X_iot_sel)

pca_iot = PCA(n_components=5, random_state=42)
X_iot_pca = pca_iot.fit_transform(X_iot_scaled).astype(np.float32)
print(f"IoT after PCA: {X_iot_pca.shape}")
print(f"Explained variance: {pca_iot.explained_variance_ratio_.sum():.4f}")

np.save('processed/X_iot.npy', X_iot_pca)
np.save('processed/y_iot.npy', y_iot)

print("\n✅ Preprocessing complete. Files saved to processed/")
