"""
Panel app for visualizing multiscale Zarr data on a map with time series.

Expects a Zarr store with:
  - /multiscales: regridded pyramid groups (0, 1, 2, ...) with dims
    (swath_time, spacecraft, latitude, longitude) or
    (swath_time, spacecraft, beam, latitude, longitude)
    with multiscales metadata on the group attrs
  - /timeseries: contiguous time-series store with dims (gpi, obs, [beam])
    plus n_obs, time, latitude, longitude coordinates

Usage:
    panel serve zarr_viewer.py --args /path/to/store.zarr
"""

import sys

import numpy as np
import panel as pn
import param
import holoviews as hv
from holoviews import opts, streams
import geoviews as gv
import geoviews.tile_sources as gts
from cartopy import crs as ccrs
import xarray as xr
import zarr

hv.extension("bokeh")
gv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")


def _apply_scaling(data_float, attrs):
    """Apply CF-style scale_factor and add_offset."""
    scale = attrs.get("scale_factor", 1.0)
    offset = attrs.get("add_offset", 0.0)
    if scale != 1.0 or offset != 0.0:
        valid = np.isfinite(data_float)
        data_float[valid] = data_float[valid] * scale + offset
    return data_float


def _is_cf_time_var(attrs):
    """Check if variable attributes indicate CF time encoding."""
    return "calendar" in attrs or (
        "units" in attrs and "since" in str(attrs.get("units", ""))
    )


def _decode_cf_time(data_float, attrs):
    """Decode CF-encoded time values to numpy datetime64.

    Parameters
    ----------
    data_float : np.ndarray of float64
        Raw numeric time values (NaN where fill).
    attrs : dict
        Must contain 'units' like 'days since 1970-01-01'.

    Returns
    -------
    np.ndarray of datetime64[ns]
        Decoded timestamps, NaT where input was NaN.
    """
    units_str = attrs.get("units", "days since 1970-01-01")

    parts = units_str.split(" since ")
    if len(parts) != 2:
        return data_float

    time_unit = parts[0].strip().lower()
    ref_date_str = parts[1].strip()

    try:
        ref_date = np.datetime64(ref_date_str)
    except ValueError:
        return data_float

    valid = np.isfinite(data_float)
    result = np.full(data_float.shape, np.datetime64("NaT"), dtype="datetime64[ns]")

    if not np.any(valid):
        return result

    values = data_float[valid]

    if time_unit in ("day", "days", "d"):
        td = (values * 86400e9).astype("timedelta64[ns]")
    elif time_unit in ("hour", "hours", "h", "hr"):
        td = (values * 3600e9).astype("timedelta64[ns]")
    elif time_unit in ("minute", "minutes", "min"):
        td = (values * 60e9).astype("timedelta64[ns]")
    elif time_unit in ("second", "seconds", "s", "sec"):
        td = (values * 1e9).astype("timedelta64[ns]")
    elif time_unit in ("millisecond", "milliseconds", "ms"):
        td = (values * 1e6).astype("timedelta64[ns]")
    else:
        return data_float

    result[valid] = ref_date + td
    return result


class ZarrViewer(param.Parameterized):
    zarr_path = param.String()

    variable = param.Selector(default=None, objects=[])
    swath_time_idx = param.Integer(default=0, bounds=(0, 0))
    spacecraft_idx = param.Integer(default=0, bounds=(0, 0))
    beam_idx = param.Integer(default=0, bounds=(0, 0))
    pyramid_level = param.Integer(default=0, bounds=(0, 0))

    clicked_lat = param.Number(default=None, allow_None=True)
    clicked_lon = param.Number(default=None, allow_None=True)
    clicked_gpi = param.Integer(default=None, allow_None=True)

    def __init__(self, zarr_path, **params):
        super().__init__(zarr_path=zarr_path, **params)

        self.root = zarr.open(zarr_path, mode="r")
        self.ms_root = self.root["multiscales"]
        self.ts_root = self.root["timeseries"]

        # Read multiscales metadata for pyramid level resolutions
        ms_attrs = dict(self.ms_root.attrs)
        ms_meta = ms_attrs.get("multiscales", [{}])[0]
        datasets = ms_meta.get("datasets", [])

        self.level_resolutions = []
        for ds_info in datasets:
            transforms = ds_info.get("coordinateTransformations", [])
            for t in transforms:
                if t.get("type") == "scale":
                    lat_scale = t["scale"][2]
                    self.level_resolutions.append(lat_scale)
                    break
            else:
                self.level_resolutions.append(None)

        level0 = self.ms_root["0"]
        self.has_beams = "beam" in level0
        self.swath_times = level0["swath_time"][:]
        self.spacecraft_ids = level0["spacecraft"][:]

        self.n_levels = 0
        while str(self.n_levels) in self.ms_root:
            self.n_levels += 1

        coord_names = {"swath_time", "spacecraft", "beam", "latitude", "longitude"}
        self.beam_vars = set()
        self.scalar_vars = set()
        for name in level0:
            arr = level0[name]
            if name in coord_names or not hasattr(arr, "ndim"):
                continue
            if self.has_beams and arr.ndim == 5:
                self.beam_vars.add(name)
            elif arr.ndim == 4:
                self.scalar_vars.add(name)

        all_vars = sorted(self.scalar_vars | self.beam_vars)

        self.param.variable.objects = all_vars
        if all_vars:
            self.variable = all_vars[0]
        self.param.swath_time_idx.bounds = (0, len(self.swath_times) - 1)
        self.param.spacecraft_idx.bounds = (0, len(self.spacecraft_ids) - 1)
        self.param.pyramid_level.bounds = (0, max(0, self.n_levels - 1))
        if self.has_beams:
            n_beams = level0["beam"].shape[0]
            self.param.beam_idx.bounds = (0, n_beams - 1)

        self.ts_lons = self.ts_root["longitude"][:]
        self.ts_lats = self.ts_root["latitude"][:]
        self.ts_gpis = self.ts_root["gpi"][:]
        self.ts_n_obs = self.ts_root["n_obs"][:]

        from scipy.spatial import cKDTree
        lat_rad = np.deg2rad(self.ts_lats)
        lon_rad = np.deg2rad(self.ts_lons)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        self.kdtree = cKDTree(np.column_stack([x, y, z]))

    def _find_nearest_gpi(self, lon, lat):
        lat_r = np.deg2rad(lat)
        lon_r = np.deg2rad(lon)
        qx = np.cos(lat_r) * np.cos(lon_r)
        qy = np.cos(lat_r) * np.sin(lon_r)
        qz = np.sin(lat_r)
        _, idx = self.kdtree.query([qx, qy, qz])
        return int(idx)

    def _on_map_tap(self, event):
        if event.x is not None and event.y is not None:
            clicked_x = float(event.x)
            clicked_y = float(event.y)
            clicked_lon, clicked_lat = ccrs.PlateCarree().transform_point(
                clicked_x, clicked_y, ccrs.GOOGLE_MERCATOR
            )
            self.clicked_lon = clicked_lon
            self.clicked_lat = clicked_lat
            self.clicked_gpi = self._find_nearest_gpi(clicked_lon, clicked_lat)

    def _make_hook(self, clabel, is_time_var):
        """Create a Bokeh finalize hook for tap events and colorbar formatting."""
        viewer = self

        def hook(plot, element):
            from bokeh.events import Tap
            plot.state.on_event(Tap, viewer._on_map_tap)

            # Force-update colorbar title and optionally apply datetime formatter
            fig = plot.state
            for obj in fig.right + fig.left + fig.below + fig.above:
                if hasattr(obj, "color_mapper"):
                    obj.title = clabel
                    if is_time_var:
                        from bokeh.models import CustomJSTickFormatter
                        obj.formatter = CustomJSTickFormatter(code="""
                            var ms = tick * 86400000;
                            var d = new Date(ms);
                            var y = d.getUTCFullYear();
                            var m = String(d.getUTCMonth() + 1).padStart(2, '0');
                            var day = String(d.getUTCDate()).padStart(2, '0');
                            var h = String(d.getUTCHours()).padStart(2, '0');
                            var min = String(d.getUTCMinutes()).padStart(2, '0');
                            return `${y}-${m}-${day} ${h}:${min}`;
                        """)
                    break

        return hook

    @param.depends("variable", "swath_time_idx", "spacecraft_idx",
                    "beam_idx", "pyramid_level")
    def map_view(self):
        """Render the current data slice at the selected pyramid level."""
        var = self.variable
        if var is None:
            return gv.Points([])

        level = self.pyramid_level
        level_group = self.ms_root[str(level)]
        arr = level_group[var]
        attrs = dict(arr.attrs)
        lats = level_group["latitude"][:]
        lons = level_group["longitude"][:]

        t = self.swath_time_idx
        s = self.spacecraft_idx

        if var in self.beam_vars and self.has_beams:
            data_2d = arr[t, s, self.beam_idx, :, :]
        else:
            data_2d = arr[t, s, :, :]

        fill_val = arr.metadata.fill_value
        data_float = data_2d.astype(np.float64)
        data_float[data_2d == fill_val] = np.nan
        data_float = _apply_scaling(data_float, attrs)

        # Decode CF time variables for display
        is_time_var = _is_cf_time_var(attrs)
        if is_time_var:
            data_decoded = _decode_cf_time(data_float, attrs)
            valid = ~np.isnat(data_decoded)
            if np.any(valid):
                epoch = np.datetime64("1970-01-01", "ns")
                data_float = np.full(data_decoded.shape, np.nan, dtype=np.float64)
                data_float[valid] = (
                    (data_decoded[valid] - epoch).astype("float64") / 86400e9
                )

        da = xr.DataArray(
            data_float,
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
            name=var,
        )

        try:
            time_val = np.datetime_as_string(self.swath_times[t], unit="h")
        except Exception:
            time_val = str(self.swath_times[t])

        units = attrs.get("units", "")
        if is_time_var:
            clabel = f"{var} [UTC]"
        elif units:
            clabel = f"{var} [{units}]"
        else:
            clabel = var

        level_res = self.level_resolutions[level] if level < len(self.level_resolutions) else "?"

        basemap = gts.EsriTerrain()

        img = gv.Image(
            da, kdims=["longitude", "latitude"], vdims=[var],
            crs=ccrs.PlateCarree(),
        ).opts(
            cmap="viridis",
            colorbar=True,
            clabel=clabel,
            tools=["hover", "tap"],
            alpha=0.7,
            title=(
                f"{var} | {time_val} | SC {self.spacecraft_ids[s]} "
                f"| L{level} ({level_res}\u00b0)"
            ),
            hooks=[self._make_hook(clabel, is_time_var)],
            responsive=True,
            aspect=2,
            active_tools=["wheel_zoom"],
        )

        return basemap * img

    @param.depends("clicked_lat", "clicked_lon")
    def marker_view(self):
        """Render click marker as a separate overlay."""
        if self.clicked_lat is not None and self.clicked_lon is not None:
            return gv.Points(
                [(self.clicked_lon, self.clicked_lat)],
                crs=ccrs.PlateCarree(),
            ).opts(
                color="red", size=12, marker="x", line_width=3,
            )
        return gv.Points([])

    @param.depends("clicked_gpi", "variable", "beam_idx")
    def timeseries_view(self):
        """Render the time series for the clicked GPI."""
        gpi_idx = self.clicked_gpi
        if gpi_idx is None:
            return hv.Scatter([]).opts(
                title="Click a point on the map",
                responsive=True,
                height=300,
            )

        n_obs = int(self.ts_n_obs[gpi_idx])
        if n_obs == 0:
            return hv.Scatter([]).opts(
                title=f"GPI {self.ts_gpis[gpi_idx]}: no observations",
                responsive=True,
                height=300,
            )

        var = self.variable
        if var not in self.ts_root:
            return hv.Scatter([]).opts(
                title=f"{var} not in timeseries store",
                responsive=True,
                height=300,
            )

        time_raw = self.ts_root["time"][gpi_idx, :n_obs]
        if not np.issubdtype(time_raw.dtype, np.datetime64):
            time_vals = (
                np.datetime64("1970-01-01")
                + (time_raw * 86400).astype("timedelta64[s]")
            )
        else:
            time_vals = time_raw

        ts_arr = self.ts_root[var]
        sat_id_arr = self.ts_root["sat_id"]
        attrs = dict(ts_arr.attrs)
        if ts_arr.ndim == 3:
            data = ts_arr[gpi_idx, :n_obs, self.beam_idx]
        else:
            data = ts_arr[gpi_idx, :n_obs]

        sat_id_data = sat_id_arr[gpi_idx, :n_obs]
        sat_id_fill_val = sat_id_arr.metadata.fill_value
        sat_id_data_float = sat_id_data.astype(np.float64)
        sat_id_data_float[sat_id_data==sat_id_fill_val] = np.nan
        sat_id_data_float = _apply_scaling(sat_id_data_float, dict(sat_id_arr.attrs))

        fill_val = ts_arr.metadata.fill_value
        data_float = data.astype(np.float64)
        data_float[data == fill_val] = np.nan
        data_float = _apply_scaling(data_float, attrs)

        # Decode CF time variables
        is_time_var = _is_cf_time_var(attrs)
        if is_time_var:
            data_decoded = _decode_cf_time(data_float, attrs)
            valid = ~np.isnat(data_decoded)
            if not np.any(valid):
                return hv.Scatter([]).opts(
                    title=f"GPI {self.ts_gpis[gpi_idx]}: all NaT for {var}",
                    responsive=True,
                    height=300,
                )
        else:
            data_decoded = None
            valid = np.isfinite(data_float)

        if not np.any(valid):
            return hv.Scatter([]).opts(
                title=f"GPI {self.ts_gpis[gpi_idx]}: all fill values for {var}",
                responsive=True,
                height=300,
            )

        gpi_val = self.ts_gpis[gpi_idx]
        lat_val = self.ts_lats[gpi_idx]
        lon_val = self.ts_lons[gpi_idx]

        units = attrs.get("units", "")
        if is_time_var:
            ylabel = f"{var} [UTC]"
        elif units:
            ylabel = f"{var} [{units}]"
        else:
            ylabel = var

        if data_decoded is not None:
            y_data = data_decoded[valid]
        else:
            y_data = data_float[valid]

        scatter = hv.Scatter(
            (time_vals[valid], y_data, sat_id_data_float[valid]),
            kdims=["Time"],
            vdims=[var, "sat_id"],
        ).opts(
            title=f"GPI {gpi_val} ({lat_val:.2f}, {lon_val:.2f}) \u2014 {n_obs} obs",
            ylabel=ylabel,
            responsive=True,
            height=300,
            size=3,
            color="navy",
            tools=["hover"],
        )

        return scatter

    def view(self):
        time_player = pn.widgets.Player(
            start=0,
            end=len(self.swath_times) - 1,
            value=0,
            step=1,
            name="Swath Time",
            loop_policy="loop",
            interval=500,
            width=250,
        )
        time_player.link(self, value="swath_time_idx", bidirectional=True)

        sc_slider = pn.widgets.IntSlider.from_param(
            self.param.spacecraft_idx,
            name="Spacecraft",
        )
        var_select = pn.widgets.Select.from_param(
            self.param.variable,
            name="Variable",
        )
        level_slider = pn.widgets.IntSlider.from_param(
            self.param.pyramid_level,
            name="Pyramid Level",
        )

        controls = [var_select, time_player, sc_slider, level_slider]

        if self.has_beams:
            beam_slider = pn.widgets.IntSlider.from_param(
                self.param.beam_idx,
                name="Beam",
            )
            controls.append(beam_slider)

        click_info = pn.pane.Str(object="Click on the map to select a point")

        def _update_click_info(*events):
            if self.clicked_lat is not None:
                click_info.object = (
                    f"Clicked: ({self.clicked_lat:.2f}, {self.clicked_lon:.2f}) "
                    f"-> GPI {self.ts_gpis[self.clicked_gpi]} "
                    f"(idx {self.clicked_gpi})"
                )

        self.param.watch(
            _update_click_info, ["clicked_lat", "clicked_lon", "clicked_gpi"]
        )

        sidebar = pn.Column(
            "## Controls",
            *controls,
            pn.layout.Divider(),
            click_info,
            width=280,
        )

        map_dmap = hv.DynamicMap(self.map_view)
        marker_dmap = hv.DynamicMap(self.marker_view)
        combined_map = map_dmap * marker_dmap

        map_pane = pn.pane.HoloViews(combined_map, linked_axes=False)
        ts_pane = pn.panel(self.timeseries_view, linked_axes=False)

        main = pn.Column(map_pane, pn.layout.Divider(), ts_pane)

        return pn.Row(sidebar, main)


def main():
    if len(sys.argv) < 2:
        print("Usage: panel serve zarr_viewer.py --args /path/to/store.zarr")
        sys.exit(1)

    zarr_path = sys.argv[1]
    viewer = ZarrViewer(zarr_path)
    return viewer.view()


app = main()
app.servable()
