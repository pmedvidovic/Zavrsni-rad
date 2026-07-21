import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Učitaj podatke
lfmc_df = pd.read_csv(r"C:\Users\Petra\Desktop\zavrsni-rad\lfmc_pixels.csv")
s2_df_list = []
s2_folder = r"C:\Users\Petra\Desktop\zavrsni-rad\s2_pixels\zavrsni-rad-s2"
csv_files = glob.glob(os.path.join(s2_folder, "*.csv"))
for f in csv_files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            s2_df_list.append(df)
    except:
        pass
s2_df = pd.concat(s2_df_list, ignore_index=True)

# Dodaj point_id u lfmc_df
unique_points = lfmc_df[['lon', 'lat']].drop_duplicates().reset_index(drop=True)
unique_points['point_id'] = unique_points.index
lfmc_df = lfmc_df.merge(unique_points, on=['lon', 'lat'], how='left')

# Spoji
merged = lfmc_df.merge(
    s2_df[['point_id', 'date', 'NDVI', 'NDWI', 'NDII']],
    on=['point_id', 'date'],
    how='inner'
)
merged = merged[(merged['LFMC'] >= 10) & (merged['LFMC'] <= 300)]
merged = merged.dropna(subset=['NDVI', 'NDWI', 'NDII', 'LFMC'])
merged['date'] = pd.to_datetime(merged['date'])
merged['month'] = merged['date'].dt.month

print(f'Dataset: {len(merged)} redaka')

#Vremenska analiza
daily_stats = merged.groupby('date')['LFMC'].agg(['mean', 'std']).reset_index()

fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(daily_stats['date'],
                daily_stats['mean'] - daily_stats['std'],
                daily_stats['mean'] + daily_stats['std'],
                alpha=0.3, color='steelblue', label='±1 std')

ax.plot(daily_stats['date'], daily_stats['mean'],
        'o-', color='steelblue', linewidth=2, markersize=5, label='Srednja vrijednost')

ax.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, label='Kritična granica (100%)')
ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, label='Visoki rizik (60%)')

ax.set_xlabel('Datum', fontsize=12)
ax.set_ylabel('LFMC (%)', fontsize=12)
ax.set_title('Sezonska dinamika vlage živog goriva (LFMC)\nSplitsko-dalmatinska županija, 2024.', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 200)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(r"C:\Users\Petra\Desktop\zavrsni-rad\vremenska_analiza_v2.png", dpi=150)
plt.show()
print('Vremenska analiza spremljena!')


#Prostorna karta po godišnjim dobima
seasons = {
    'Zima (Sij-Velj)': [1, 2],
    'Proljeće (Ožu-Svi)': [3, 4, 5],
    'Ljeto (Lip-Kol)': [6, 7, 8],
    'Jesen (Ruj-Stu)': [9, 10, 11]
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (season_name, months) in enumerate(seasons.items()):
    season_data = merged[merged['month'].isin(months)]
    season_mean = season_data.groupby(['lon', 'lat'])['LFMC'].mean().reset_index()

    sc = axes[idx].scatter(season_mean['lon'], season_mean['lat'],
                           c=season_mean['LFMC'], cmap='RdYlGn',
                           vmin=40, vmax=160, s=20, alpha=0.8)

    plt.colorbar(sc, ax=axes[idx], label='LFMC (%)')
    axes[idx].set_title(season_name, fontsize=12)
    axes[idx].set_xlabel('Longitude')
    axes[idx].set_ylabel('Latitude')
    axes[idx].grid(True, alpha=0.3)

fig.suptitle('Prostorna distribucija LFMC po godišnjim dobima\nSplitsko-dalmatinska županija, 2024.',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(r"C:\Users\Petra\Desktop\zavrsni-rad\prostorna_karta_v2.png", dpi=150, bbox_inches='tight')
plt.show()
print('Prostorna karta spremljena!')