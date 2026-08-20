import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import glob
import os
import rasterio
from rasterio.warp import transform as rio_transform

ee.Initialize(project='zavrsni-rad-499917')

# Učita i trenira model
print('Učitavanje podataka i treniranje modela...')
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

# Dva datuma za predviđanje
dates_to_predict = {
    'Kolovoz 2026 — datum požara (13.08.2026.)': ('2026-08-07', '2026-08-19', 8, 20),
    'Siječanj 2026 — zimski datum (15.01.2026.)': ('2026-01-01', '2026-01-31', 1, 50)
}

# Preuzme S2 indekse za te datume
def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B8', 'B11']).rename('NDWI')
    ndii = image.normalizedDifference(['B8A', 'B11']).rename('NDII')
    return image.addBands([ndvi, ndwi, ndii]).select(['NDVI', 'NDWI', 'NDII'])

roi = ee.Geometry.Rectangle([15.8, 43.0, 17.7, 44.0])

# Uzme lokacije iz dataseta
sample_points = unique_points.sample(n=min(5000, len(unique_points)), random_state=42)
features = []
for idx, row in sample_points.iterrows():
    feat = ee.Feature(
        ee.Geometry.Point([row['lon'], row['lat']]),
        {'point_id': int(row['point_id'])}
    )
    features.append(feat)
points_fc = ee.FeatureCollection(features)

results = {}
for label, (start, end, month, cloud) in dates_to_predict.items():
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start, end) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud)) \
        .map(add_indices) \
        .median()

    task = ee.batch.Export.table.toDrive(
        collection=s2.sampleRegions(
            collection=points_fc,
            scale=10,
            geometries=False
        ).map(lambda f: f.set('month', month)),
        description=f'predict_{month:02d}_2026',
        folder='zavrsni-rad-predict',
        fileNamePrefix=f'predict_{month:02d}',
        fileFormat='CSV'
    )
    task.start()
    print(f'Export pokrenut za {label}')

print('\nSvi taskovi pokrenuti! Provjeri Google Drive za 10 minuta.')