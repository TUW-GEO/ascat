#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import numpy as np
from ascat.read_native.ragged_array_ts import SwathFileCollection
from flox.xarray import xarray_reduce
from matplotlib import pyplot as plt
from cmcrameri import cm as cmc
import cartopy.crs as ccrs
from ascat.regrid.regrid import grid_to_regular_grid

def simple_map(lons, lats, color_var, cmap, dates=None, cbar_label=None):
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.bottom_labels = False
    gl.right_labels = False
    ax.set_extent([lons.min()-5, lons.max()+5, lats.min()-5, lats.max()+5])
    # ax.set_extent([-10, 30, 35, 65])
    plt.scatter(
        lons,
        lats,
        c=color_var,
        cmap=cmap,
        s=1,
        # alpha=0.8,
        # clim=(0, 100)
    )
    if cbar_label is None:
        cbar_label = (
            f"Average {color_var.long_name}\n"
            f"({color_var.units})\n"
        )
    if dates is not None:
        cbar_label += f"\n{np.datetime_as_string(dates[0], unit='s')} - {np.datetime_as_string(dates[1], unit='s')}"

    plt.colorbar(label=(cbar_label),
                 shrink=0.5,
                 pad=0.05,
                 orientation="horizontal"
    )
    plt.tight_layout()

def plot_ds(ds, grid):
    ds["location_id"].load()
    avg_ssm_flox = xarray_reduce(ds["surface_soil_moisture"], ds["location_id"], func="mean").load()
    print(avg_ssm_flox)
    lons, lats = grid.gpi2lonlat(avg_ssm_flox['location_id'].values)
    simple_map(lons, lats, avg_ssm_flox, cmc.roma, dates=(ds.time.min(), ds.time.max()))

def main():
    name = 'W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-12.5km-H121_C_LIIB_20240117135900_20210101060000_20210101065959____.nc'
    ungridded_path = Path('/home/charriso/p14/data-write/RADAR/hsaf/h121_v1.0/swaths/metop_a/2021/')
    regridded_path = Path('/home/charriso/temp_py/')
    ungridded = SwathFileCollection.from_product_id(ungridded_path, 'H121_v1.0')
    regridded = SwathFileCollection.from_product_id(regridded_path, 'H121_v1.0')

    # print(ungridded.get_filenames(start_dt=datetime(2021, 1, 1, 5), end_dt=datetime(2021, 1, 1, 7)))

    ug_ds = ungridded.read(date_range=(datetime(2021, 1, 1, 6), datetime(2021, 1, 1, 7)))
    rg_ds = regridded.read(date_range=(datetime(2021, 1, 1, 5), datetime(2021, 1, 1, 7)))

    new_grid, _, _ = grid_to_regular_grid(ungridded.grid, 0.1)
    plot_ds(ug_ds, ungridded.grid)
    plot_ds(rg_ds, new_grid)
    plt.show()

if __name__ == '__main__':
    main()
