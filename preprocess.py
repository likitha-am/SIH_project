import numpy as np
import pandas as pd
import os, json, re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import pickle

# =====================================================
# CONFIG
# =====================================================
IN_PATH = r"C:\Users\GIRIDHAR\OneDrive\Desktop\FRA\FRA_new_dataset_augmented_fixed.xlsx"
OUT_DIR = r"C:\Users\GIRIDHAR\OneDrive\Desktop\FRA\processed_fra_pytorch"
os.makedirs(OUT_DIR, exist_ok=True)

print("\nLoading dataset from:", IN_PATH)
df = pd.read_excel(IN_PATH, engine="openpyxl")
print("Loaded rows:", df.shape[0])
print("Columns:", df.columns.tolist())

# =====================================================
# PARSER FOR ARRAYS
# =====================================================
def parse_json_array(cell):
    if cell is None:
        return None

    if isinstance(cell, (list, tuple, np.ndarray)):
        try:
            return np.array(cell, dtype=float)
        except:
            return None

    if isinstance(cell, (int, float)):
        return np.array([float(cell)], dtype=float)

    s = str(cell).strip()
    if s == "" or s.lower() in ["nan", "none", "null"]:
        return None

    # Try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple)):
            return np.array(parsed, dtype=float)
        if isinstance(parsed, (int, float)):
            return np.array([float(parsed)], dtype=float)
    except:
        pass

    # Regex fallback
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if nums:
        try:
            return np.array([float(x) for x in nums], dtype=float)
        except:
            return None
    return None

# =====================================================
# SAFE INTERPOLATION (FINAL FIXED VERSION)
# =====================================================
def safe_interpolate(freqs, mags, freq_grid):
    freqs = np.array(freqs, float)
    mags  = np.array(mags, float)

    # -----------------------------
    # FIX 1: match lengths
    # -----------------------------
    L = min(len(freqs), len(mags))
    if L < 5:
        return None

    freqs = freqs[:L]
    mags  = mags[:L]

    # -----------------------------
    # FIX 2: remove invalid values
    # -----------------------------
    mask = np.isfinite(freqs) & np.isfinite(mags)
    freqs = freqs[mask]
    mags  = mags[mask]

    if len(freqs) < 5:
        return None

    # -----------------------------
    # FIX 3: ensure sorted
    # -----------------------------
    order = np.argsort(freqs)
    freqs = freqs[order]
    mags  = mags[order]

    # -----------------------------
    # FIX 4: remove duplicate freqs
    # -----------------------------
    freqs, idx = np.unique(freqs, return_index=True)
    mags = mags[idx]

    if len(freqs) < 5:
        return None

    # -----------------------------
    # FIX 5: interpolate
    # -----------------------------
    try:
        f = interp1d(freqs, mags, bounds_error=False, fill_value="extrapolate")#type:ignore
        return f(freq_grid)
    except:
        return None

# =====================================================
# TARGET GRID (uniform 2048 points)
# =====================================================
N_GRID = 2048
freq_grid = np.logspace(np.log10(20), np.log10(1e6), N_GRID)

X_list, P_list, valid_idx = [], [], []

# =====================================================
# PROCESS EACH ROW
# =====================================================
for idx, row in df.iterrows():
    freqs = parse_json_array(row.get("frequency_hz"))
    mags  = parse_json_array(row.get("magnitude_db"))
    phs   = parse_json_array(row.get("phase_deg"))

    if freqs is None or mags is None:
        print(f"Skipping row {idx}: invalid freq/mag")
        continue

    # MAG
    mags_rs = safe_interpolate(freqs, mags, freq_grid)
    if mags_rs is None:
        print(f"Skipping row {idx}: mag failed")
        continue

    # PHASE
    if phs is None:
        phs_rs = np.zeros(N_GRID, float)
    else:
        phs_rs = safe_interpolate(freqs, phs, freq_grid)
        if phs_rs is None:
            phs_rs = np.zeros(N_GRID, float)

    X_list.append(mags_rs.astype(np.float32))
    P_list.append(phs_rs.astype(np.float32))
    valid_idx.append(idx)

# =====================================================
# STACK
# =====================================================
if len(X_list) == 0:
    raise RuntimeError("All rows invalid! Check data format.")

X = np.vstack(X_list)
P = np.vstack(P_list)
df = df.loc[valid_idx].reset_index(drop=True)

print("Final usable rows:", X.shape[0])

# =====================================================
# NORMALIZE EACH FRA CURVE
# =====================================================
means = X.mean(axis=1, keepdims=True)
stds  = X.std(axis=1, keepdims=True) + 1e-9
X_norm = (X - means) / stds

# =====================================================
# METADATA
# =====================================================
meta_cols = [
    "tap_position","winding_type","vendor_name","manufacturer",
    "transformer_rating","voltage_rating_kv","oil_temperature_c"
]
meta_df = df[[c for c in meta_cols if c in df.columns]].copy()

# tap position
if "tap_position" in meta_df:
    meta_df["tap_position"] = pd.to_numeric(meta_df["tap_position"], errors="coerce").fillna(0.0)
else:
    meta_df["tap_position"] = 0.0

# voltage rating
if "voltage_rating_kv" in meta_df:
    meta_df["voltage_rating_kv"] = (
        meta_df["voltage_rating_kv"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
    )
    meta_df["voltage_rating_kv"] = pd.to_numeric(meta_df["voltage_rating_kv"], errors="coerce").fillna(0.0)
else:
    meta_df["voltage_rating_kv"] = 0.0

# oil temp
if "oil_temperature_c" in meta_df:
    meta_df["oil_temperature_c"] = pd.to_numeric(meta_df["oil_temperature_c"], errors="coerce").fillna(0.0)
else:
    meta_df["oil_temperature_c"] = 0.0

# transformer rating
def extract_num(x):
    m = re.findall(r"(\d+\.?\d*)", str(x))
    return float(m[0]) if m else 0.0

if "transformer_rating" in meta_df:
    transformer_rating_numeric = meta_df["transformer_rating"].apply(extract_num).astype(np.float32)
else:
    transformer_rating_numeric = np.zeros(len(meta_df), np.float32)

# label encoding
def encode(col):
    if col not in meta_df:
        return np.zeros(len(meta_df), int), LabelEncoder().fit(["UNKNOWN"])
    arr = meta_df[col].fillna("UNKNOWN").astype(str)
    le = LabelEncoder()
    return le.fit_transform(arr), le

winding_enc, le_winding = encode("winding_type")
vendor_enc,  le_vendor  = encode("vendor_name")
manu_enc,    le_manu    = encode("manufacturer")

meta_array = np.column_stack([
    meta_df["tap_position"].astype(np.float32),
    winding_enc.astype(np.float32),#type:ignore
    vendor_enc.astype(np.float32),#type:ignore
    manu_enc.astype(np.float32),#type:ignore
    transformer_rating_numeric,
    meta_df["voltage_rating_kv"].astype(np.float32),
    meta_df["oil_temperature_c"].astype(np.float32)
])

# =====================================================
# LABELS
# =====================================================
le_fault = LabelEncoder()
y_fault = le_fault.fit_transform(df["fault_type"].fillna("UNKNOWN").astype(str))

le_cond = LabelEncoder()
y_cond = le_cond.fit_transform(df["condition"].fillna("UNKNOWN").astype(str))

# =====================================================
# TRANSFORMER-WISE SPLIT
# =====================================================
if "transformer_id" not in df.columns:
    df["transformer_id"] = [f"T{i}" for i in range(len(df))]

t_ids = df["transformer_id"].unique().tolist()

train_ids, test_ids = train_test_split(t_ids, test_size=0.3, random_state=42)
val_ids, test_ids   = train_test_split(test_ids, test_size=0.5, random_state=42)

def mask(ids):
    return df["transformer_id"].isin(ids).values

splits = {
    "train": mask(train_ids),
    "val":   mask(val_ids),
    "test":  mask(test_ids)
}

# =====================================================
# SAVE NPZ FILES
# =====================================================
for name, m in splits.items():
    out = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez_compressed(
        out,
        X=X_norm[m],
        P=P[m],#type:ignore
        meta=meta_array[m],#type:ignore
        y_fault=y_fault[m],#type:ignore
        y_cond=y_cond[m]#type:ignore
    )
    print(f"Saved → {out}")

# =====================================================
# SAVE ENCODERS
# =====================================================
enc = {
    "le_fault": le_fault,
    "le_cond": le_cond,
    "le_vendor": le_vendor,
    "le_winding": le_winding,
    "le_manu": le_manu
}

with open(os.path.join(OUT_DIR, "encoders.pkl"), "wb") as f:
    pickle.dump(enc, f)

print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
