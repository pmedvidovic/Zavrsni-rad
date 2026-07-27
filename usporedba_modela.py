import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

#ucita podatke
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

#pripremi dataset
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

#originalni indeksi
X_orig = merged[['NDVI', 'NDWI', 'NDII']].values

#transformirani indeksi
merged['NDVI_sq'] = merged['NDVI'] ** 2
merged['NDII_sq'] = merged['NDII'] ** 2
merged['NDVI_NDII'] = merged['NDVI'] * merged['NDII']
merged['NDVI_NDWI'] = merged['NDVI'] * merged['NDWI']
merged['sin_month'] = np.sin(2 * np.pi * merged['month'] / 12)
merged['cos_month'] = np.cos(2 * np.pi * merged['month'] / 12)
X_trans = merged[['NDVI', 'NDWI', 'NDII', 'NDVI_sq', 'NDII_sq',
                   'NDVI_NDII', 'NDVI_NDWI', 'sin_month', 'cos_month']].values

y = merged['LFMC'].values

#train/test split
X_orig_train, X_orig_test, y_train, y_test = train_test_split(X_orig, y, test_size=0.2, random_state=42)
X_trans_train, X_trans_test, _, _ = train_test_split(X_trans, y, test_size=0.2, random_state=42)

#modeli
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
}

#usporedba
print(f"{'Model':<25} {'Originalni R² train':>20} {'Originalni R² test':>20} {'Transformirani R² train':>25} {'Transformirani R² test':>25}")
print("-" * 115)

results = []
for name, model in models.items():
    #originalni indeksi
    model.fit(X_orig_train, y_train)
    r2_orig_train = r2_score(y_train, model.predict(X_orig_train))
    r2_orig_test = r2_score(y_test, model.predict(X_orig_test))
    mae_orig_test = mean_absolute_error(y_test, model.predict(X_orig_test))

    #transformirani indeksi
    model.fit(X_trans_train, y_train)
    r2_trans_train = r2_score(y_train, model.predict(X_trans_train))
    r2_trans_test = r2_score(y_test, model.predict(X_trans_test))
    mae_trans_test = mean_absolute_error(y_test, model.predict(X_trans_test))

    print(f"{name:<25} {r2_orig_train:>20.4f} {r2_orig_test:>20.4f} {r2_trans_train:>25.4f} {r2_trans_test:>25.4f}")
    
    results.append({
        'Model': name,
        'R² train (orig)': r2_orig_train,
        'R² test (orig)': r2_orig_test,
        'MAE test (orig)': mae_orig_test,
        'R² train (trans)': r2_trans_train,
        'R² test (trans)': r2_trans_test,
        'MAE test (trans)': mae_trans_test
    })

#spremi rezultate
results_df = pd.DataFrame(results)
results_df.to_csv(r"C:\Users\Petra\Desktop\zavrsni-rad\usporedba_modela.csv", index=False)
print('\nRezultati spremljeni u usporedba_modela.csv!')

#graf
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.2

ax.bar(x - 1.5*width, results_df['R² test (orig)'], width, label='Originalni - test', color='steelblue')
ax.bar(x - 0.5*width, results_df['R² train (orig)'], width, label='Originalni - train', color='lightsteelblue')
ax.bar(x + 0.5*width, results_df['R² test (trans)'], width, label='Transformirani - test', color='darkorange')
ax.bar(x + 1.5*width, results_df['R² train (trans)'], width, label='Transformirani - train', color='moccasin')

ax.set_xlabel('Model')
ax.set_ylabel('R²')
ax.set_title('Usporedba modela - originalni vs transformirani indeksi')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'])
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\Petra\Desktop\zavrsni-rad\usporedba_modela.png", dpi=150)
plt.show()
print('Graf spremljen!')