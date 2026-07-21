import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Učitaj sve S2 CSV datoteke
s2_folder = r"C:\Users\Petra\Desktop\zavrsni-rad\s2_pixels\zavrsni-rad-s2"
csv_files = glob.glob(os.path.join(s2_folder, "*.csv"))
print(f'Pronađeno {len(csv_files)} CSV datoteka')

dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df)
    except:
        pass
s2_df = pd.concat(dfs, ignore_index=True)
print(f'Ukupno S2 redaka: {len(s2_df)}')

# Učitaj LFMC piksele
lfmc_df = pd.read_csv(r"C:\Users\Petra\Desktop\zavrsni-rad\lfmc_pixels.csv")
print(f'Ukupno LFMC redaka: {len(lfmc_df)}')

# Dodaj point_id u lfmc_df
unique_points = lfmc_df[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
unique_points['point_id'] = unique_points.index
lfmc_df = lfmc_df.merge(unique_points, on=['lon', 'lat'], how='left')

# Spoji po point_id i date
merged = lfmc_df.merge(
    s2_df[['point_id', 'date', 'NDVI', 'NDWI', 'NDII']],
    on=['point_id', 'date'],
    how='inner'
)
print(f'Spojenih redaka: {len(merged)}')

# Čišćenje
merged = merged[(merged['LFMC'] >= 10) & (merged['LFMC'] <= 300)]
merged = merged.dropna(subset=['NDVI', 'NDWI', 'NDII', 'LFMC'])
merged = merged[(merged['NDVI'] != 0) | (merged['NDWI'] != 0)]
print(f'Nakon čišćenja: {len(merged)}')

# Dodavanje značajke
merged['month'] = pd.to_datetime(merged['date']).dt.month
merged['NDVI_sq'] = merged['NDVI'] ** 2
merged['NDII_sq'] = merged['NDII'] ** 2
merged['NDVI_NDII'] = merged['NDVI'] * merged['NDII']
merged['NDVI_NDWI'] = merged['NDVI'] * merged['NDWI']
merged['sin_month'] = np.sin(2 * np.pi * merged['month'] / 12)
merged['cos_month'] = np.cos(2 * np.pi * merged['month'] / 12)

feature_cols = ['NDVI', 'NDWI', 'NDII', 'NDVI_sq', 'NDII_sq',
                'NDVI_NDII', 'NDVI_NDWI', 'sin_month', 'cos_month']
X = merged[feature_cols].values
y = merged['LFMC'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

# Treniranje modela
print('Treniranje modela...')
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=3,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluacija
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'\n=== REZULTATI ===')
print(f'R²:  {r2:.4f}')
print(f'MAE: {mae:.2f}%')

# Važnost značajki
print('\nVažnost značajki:')
for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    print(f'  {name}: {imp:.4f}')

# Graf
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred, alpha=0.2, s=5)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Stvarni LFMC (%)')
axes[0].set_ylabel('Predviđeni LFMC (%)')
axes[0].set_title(f'R²={r2:.4f}, MAE={mae:.2f}%')

importances = model.feature_importances_
sorted_idx = np.argsort(importances)
axes[1].barh([feature_cols[i] for i in sorted_idx], importances[sorted_idx])
axes[1].set_title('Važnost značajki')

plt.tight_layout()
plt.savefig(r"C:\Users\Petra\Desktop\zavrsni-rad\rezultati_finalni.png", dpi=150)
plt.show()
print('Graf spremljen!')