import os
import pandas as pd
import geopandas as gpd
import geodatasets
from geodatasets import get_path
import matplotlib.pyplot as plt
import numpy as np



DATADIR= os.path.abspath(os.path.join(os.path.dirname(__file__)))
FILEPATH =  os.path.join(DATADIR, 'eshm20/input_shapefiles/eshm20_input_a_unified_eq_catalogue/eshm20_unified_catalogue_declustered_v02a.shp' )

catalog = gpd.read_file(FILEPATH)

print(catalog.columns)

lonname = 'longitude'
latname = 'latitude'
magname = 'magnitude'
yearname = 'year'
monthname = 'month'



#catalog = pd.read_csv(FILEPATH, sep='\t', header=0)

## select only earthquakes with magnitude greater than 4.0

I = True #catalog['Ev. type'] != '*'
I = I & (catalog[yearname] >= 2012.0)
I = I & (catalog[monthname] >= 5.0)
I = I & (catalog[magname] >= 2)
#I = I & (catalog['rb_rfact20'] == 'TRUE')

print(catalog[yearname].min(), catalog[yearname].max())

catalog = catalog[I]

plt.scatter(catalog[lonname], catalog[latname], s=1, color='blue', label='All events')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Locations')
plt.legend()





## binning


print(len(catalog[I]))



## binning the data into a grid

min_lon = catalog[lonname].min()
max_lon = catalog[lonname].max()
min_lat = catalog[latname].min()
max_lat = catalog[latname].max()

lon_center = (min_lon + max_lon) / 2
lat_center = (min_lat + max_lat) / 2

ratio = np.cos(np.radians(lat_center))

print(ratio)

delta_x = (max_lon - min_lon) * ratio
delta_y = max_lat - min_lat

plt.gca().set_aspect(delta_y/delta_x)

print(delta_y/delta_x)



plt.show()
print(delta_x, delta_y)


NBINS = 250000

# nx * ny = NBINS, nx / ny = delta_x / delta_y

nx = int(np.sqrt(NBINS * delta_x / delta_y))
ny = int(NBINS / nx)

print(nx, ny)

## pleae bin the data into a grid of nx by ny and count the number of events in each bin
x_bins = np.linspace(min_lon, max_lon, nx+1)
y_bins = np.linspace(min_lat, max_lat, ny+1)

H, xedges, yedges = np.histogram2d(catalog['Lon'], catalog['Lat'], bins=[x_bins, y_bins])
plt.imshow(np.log(H.T+1), origin='lower', extent=[min_lon, max_lon, min_lat, max_lat], cmap='hot')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Density')
plt.colorbar(label='Number of events')
plt.show()