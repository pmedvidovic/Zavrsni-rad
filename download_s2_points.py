import ee
import pandas as pd
import numpy as np

ee.Initialize(project='zavrsni-rad-499917')

# ucitava piksele
lfmc_df = pd.read_csv(r"C:\Users\Petra\Desktop\zavrsni-rad\lfmc_pixels.csv")
print(f'Ukupno redaka: {len(lfmc_df)}')

# uzima jedinstvene tocke, bez da ponavlja po datumima
unique_points = lfmc_df[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
unique_points['point_id'] = unique_points.index
print(f'Jedinstvenih točaka: {len(unique_points)}')

# uzme jedinstvene datume S2
dates = sorted(lfmc_df['date'].unique())
print(f'Jedinstvenih datuma: {len(dates)}')

# funk za izračun indeksa
def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B8', 'B11']).rename('NDWI')
    ndii = image.normalizedDifference(['B8A', 'B11']).rename('NDII')
    return image.addBands([ndvi, ndwi, ndii]).select(['NDVI', 'NDWI', 'NDII'])

# podili točke u grupe od 25000
group_size = 25000
n_groups = int(np.ceil(len(unique_points) / group_size))
print(f'Broj grupa: {n_groups}')

for date_str in dates:
    date = pd.to_datetime(date_str)
    #trazi snimke 6 dana oko datuma (gori,doli)
    start = (date - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
    end = (date + pd.Timedelta(days=6)).strftime('%Y-%m-%d')

    #dohvati snimku,filtrira oblake, racuna idekse
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start, end) \
        .filterBounds(ee.Geometry.Rectangle([15.8, 43.0, 17.7, 44.0])) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(add_indices) \
        .median()

    for g in range(n_groups):
        #tocke za tu grupu
        group = unique_points.iloc[g*group_size:(g+1)*group_size]
         #pritvori i u GEE FeatureCollection
        features = []
        for idx, row in group.iterrows():
            feat = ee.Feature(
                ee.Geometry.Point([row['lon'], row['lat']]),
                {'point_id': int(row['point_id'])}
            )
            features.append(feat)
        
        points_fc = ee.FeatureCollection(features)
        
        #vrijednosti indeksa na lokacijama tocaka
        sampled = s2.sampleRegions(
            collection=points_fc,
            scale=10,
            geometries=False
        )
        sampled = sampled.map(lambda f: f.set('date', date_str)) #doda datum svakoj tocki

    #batch export u Google Drive
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=f's2_{date_str}_g{g}',
            folder='zavrsni-rad-s2',
            fileNamePrefix=f's2_{date_str}_g{g}',
            fileFormat='CSV'
        )
        task.start()
    
    print(f'Pokrenut export za {date_str} ({n_groups} grupa)')

print(f'\nSvi taskovi pokrenuti!')
print(f'Ukupno taskova: {len(dates) * n_groups}')