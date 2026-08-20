import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import glob
import os

# Učitaj i treniraj model
print('Treniranje modela...')
s2_folder = r"C:\Users\Petra\Desktop\zavrsni-rad\s2_pixels\zavrsni-rad-s2"
csv_files = glob.glob(os.path.join(s2_folder, "*.csv"))
dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df)
    except:
        pass
s2_df = pd.concat(dfs, ignore_index=True)

lfmc_df = pd.read_csv(r"C:\Users\Petra\Desktop\zavrsni-rad\lfmc_pixels.csv")
unique_points = lfmc_df[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
unique_points['point_id'] = unique_points.index
lfmc_df = lfmc_df.merge(unique_points, on=['lon', 'lat'], how='left')

merged = lfmc_df.merge(
    s2_df[['point_id', 'date', 'NDVI', 'NDWI', 'NDII']],
    on=['point_id', 'date'], how='inner'
)
merged = merged[(merged['LFMC'] >= 10) & (merged['LFMC'] <= 300)]
merged = merged.dropna(subset=['NDVI', 'NDWI', 'NDII', 'LFMC'])
merged = merged[(merged['NDVI'] != 0) | (merged['NDWI'] != 0)]
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
print('Model treniran!')

# Učitaj podatke za predviđanje
predict_folder = r"C:\Users\Petra\Desktop\zavrsni-rad\predict_data"

datasets = {
    'Kolovoz 2026 — datum požara (13.08.2026.)': (os.path.join(predict_folder, 'predict_08.csv'), 8),
    'Siječanj 2026 — zimski datum (15.01.2026.)': (os.path.join(predict_folder, 'predict_01.csv'), 1)
}

# Spoji s koordinatama
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (label, (path, month)) in zip(axes, datasets.items()):
    df = pd.read_csv(path)
    print(f'{label}: {len(df)} redaka')

    # Spoji s koordinatama po point_id
    df = df.merge(unique_points[['point_id', 'lon', 'lat']], on='point_id', how='left')
    df = df.dropna(subset=['lon', 'lat', 'NDVI', 'NDWI', 'NDII'])
    df = df[(df['NDVI'] != 0) | (df['NDWI'] != 0)]

    # Dodaj značajke
    df['NDVI_sq'] = df['NDVI'] ** 2
    df['NDII_sq'] = df['NDII'] ** 2
    df['NDVI_NDII'] = df['NDVI'] * df['NDII']
    df['NDVI_NDWI'] = df['NDVI'] * df['NDWI']
    df['sin_month'] = np.sin(2 * np.pi * month / 12)
    df['cos_month'] = np.cos(2 * np.pi * month / 12)

    # Predviđanje
    X_pred = df[feature_cols].values
    df['LFMC_pred'] = model.predict(X_pred)

    print(f'  Min LFMC: {df["LFMC_pred"].min():.1f}%')
    print(f'  Max LFMC: {df["LFMC_pred"].max():.1f}%')
    print(f'  Mean LFMC: {df["LFMC_pred"].mean():.1f}%')

    # Karta
    sc = ax.scatter(df['lon'], df['lat'],
                    c=df['LFMC_pred'],
                    cmap='RdYlGn',
                    vmin=40, vmax=160,
                    s=8, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Predviđeni LFMC (%)')
    ax.set_title(label, fontsize=11, pad=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)

    # Dodaj horizontalnu liniju za srednju vrijednost
    mean_val = df['LFMC_pred'].mean()
    ax.text(0.02, 0.98, f'Srednji LFMC: {mean_val:.1f}%',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
fig.suptitle('Predviđeni LFMC — Splitsko-dalmatinska županija 2026.\nUsporedba datuma požara i zimskog datuma',
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(r"C:\Users\Petra\Desktop\zavrsni-rad\predvideni_lfmc_2026.png",
            dpi=150, bbox_inches='tight')
plt.show()
print('Karte spremljene!')