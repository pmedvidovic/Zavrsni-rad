import rasterio
import numpy as np
import pandas as pd
import glob
import os
from rasterio.warp import transform as rio_transform


tif_folder = r"C:\Users\Petra\Desktop\A1.1.1 DA20 Live Fuel Moisture Content (Sentinel-2) (2)\A1.1.1 DA20 Live Fuel Moisture Content (Sentinel-2)\data"
#trazi sve TIF datoteke
tif_files = glob.glob(os.path.join(tif_folder, "*.tif"))

#izvlaci datume
def extract_date(filepath):
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    date_str = parts[5]
    return pd.to_datetime(date_str, format='%Y%m%d')

tif_dates = [(f, extract_date(f)) for f in tif_files]
tif_dates.sort(key=lambda x: x[1])

print(f'Ukupno TIF datoteka: {len(tif_dates)}')
print(f'Raspon: {tif_dates[0][1].date()} do {tif_dates[-1][1].date()}')

all_results = []
# uzima svako 50000ti piksel
step = 50000  

for tif_path, tif_date in tif_dates:
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform_obj = src.transform
        crs = src.crs

        # trazi valjane piksele
        valid_rows, valid_cols = np.where(data > 0)
        
        # uzrokuje svako n-ti pikssel
        sampled_rows = valid_rows[::step]
        sampled_cols = valid_cols[::step]
        lfmc_values = data[sampled_rows, sampled_cols]

        # pretvvara u UTM koordinate
        utm_xs = [transform_obj.c + col * transform_obj.a for col in sampled_cols]
        utm_ys = [transform_obj.f + row * transform_obj.e for row in sampled_rows]

        # pretvara UTM u WGS84 (lon/lat)
        lons, lats = rio_transform(crs.to_string(), 'EPSG:4326', utm_xs, utm_ys)

        # Sspremi rezultat za taj datum
        for lon, lat, lfmc in zip(lons, lats, lfmc_values):
            all_results.append({
                'lon': lon,
                'lat': lat,
                'date': tif_date.strftime('%Y-%m-%d'),
                'LFMC': float(lfmc)
            })

    print(f'{tif_date.date()}: {len(sampled_rows)} piksela')

#sprema sve rezultate
df = pd.DataFrame(all_results)
print(f'\nUkupno redaka: {len(df)}')
df.to_csv(r"C:\Users\Petra\Desktop\zavrsni-rad\lfmc_pixels.csv", index=False)
print('Spremljeno u lfmc_pixels.csv!')