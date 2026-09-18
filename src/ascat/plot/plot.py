# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Plot the measurements of a full resolution backscatter file on a map.

ASCAT and SCA are different instruments, but their SZF readers return the same
generic fields, so one routine plots either of them. The measurements of the
antenna beams asked for are put on a map, and clicking one shows what the file
says about it.

Plotting needs matplotlib and eomaps, which are not installed with the package.
They are in its "plot" dependency group, so ``uv sync --group plot`` adds them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ascat.eumetsat.level1 import AscatL1bFile
from ascat.eumetsat.sca.level1 import ScaL1bFile

#: Fields shown when a measurement is picked, if the product has them.
annotated_fields = [
    "sig", "inc", "azi", "f_land", "f_usable", "f_usable_user",
    "as_des_pass", "swath_indicator", "flag_generic", "flag_surface",
]

#: Beams plotted unless others are asked for, the six both instruments have.
default_beams = ["lf-vv", "lm-vv", "la-vv", "rf-vv", "rm-vv", "ra-vv"]

#: Measurements drawn at most, drawing one takes about 20 microseconds and a
#: SZF file of ASCAT holds more than eight million of them.
default_max_points = 500000


def read_szf(filename, **kwargs):
    """
    Read a full resolution backscatter file of either instrument.

    Which reader is used follows from the file name: the SCA products carry
    "SCA-1B-SZF", the ASCAT products "ASCA_SZF".

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename.
    **kwargs
        Passed on to the read method of the reader.

    Returns
    -------
    data : dict of numpy.ndarray
        Measurements, one entry per antenna beam.
    metadata : dict
        Metadata.

    Raises
    ------
    ValueError
        If the file is of neither instrument.
    """
    name = Path(filename).name

    if "SCA-1B-SZF" in name:
        return ScaL1bFile(filename).read(**kwargs)

    if "ASCA_SZF" in name:
        return AscatL1bFile(filename).read(**kwargs)

    raise ValueError(
        f"{name} is neither an ASCAT ('ASCA_SZF') nor a SCA ('SCA-1B-SZF') "
        "full resolution backscatter file.")


def to_dataframe(data, beams=None):
    """
    Put the measurements of the given beams into one table.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Measurements, one entry per antenna beam.
    beams : iterable of str, optional
        Beams to take, all of them if not given (default: None).

    Returns
    -------
    dataframe : pandas.DataFrame
        Measurements of all beams below each other, with the beam they belong
        to in a column "beam".

    Raises
    ------
    KeyError
        If the file does not hold one of the beams.
    """
    beams = list(data) if beams is None else list(beams)
    missing = [beam for beam in beams if beam not in data]

    if missing:
        raise KeyError(
            f"No beam {', '.join(missing)} in the file, it has "
            f"{', '.join(data)}.")

    frames = []
    for beam in beams:
        measurements = data[beam]
        frame = pd.DataFrame(
            {name: np.ma.filled(measurements[name], np.nan).ravel()
             if measurements[name].dtype.kind == "f"
             else np.ma.getdata(measurements[name]).ravel()
             for name in measurements.dtype.names})
        frame["beam"] = beam
        frames.append(frame)

    return pd.concat(frames).reset_index(drop=True)


def thin(dataframe, max_points=default_max_points):
    """
    Take every n-th measurement, so that a map stays quick to draw.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Measurements.
    max_points : int, optional
        Measurements to keep at most, all of them if 0 or None
        (default: :data:`default_max_points`).

    Returns
    -------
    dataframe : pandas.DataFrame
        Every n-th measurement, or all of them if there are few enough.
    """
    if not max_points or len(dataframe) <= max_points:
        return dataframe

    return dataframe.iloc[::int(np.ceil(len(dataframe) / max_points))]


def plot_szf(filename, parameter="sig", beams=None, title=None, vmin=None,
             vmax=None, cmap="viridis", max_points=default_max_points, size=1,
             **kwargs):
    """
    Plot the measurements of a full resolution backscatter file on a map.

    Parameters
    ----------
    filename : str or pathlib.Path
        ASCAT or SCA Level 1b SZF file.
    parameter : str, optional
        Field to colour the measurements by (default: "sig").
    beams : iterable of str, optional
        Beams to plot, the six beams both instruments have if not given
        (default: None).
    title : str, optional
        Title of the map, the file name if not given (default: None).
    vmin, vmax : float, optional
        Ends of the colour scale (default: None).
    cmap : str, optional
        Colour map (default: "viridis").
    max_points : int, optional
        Measurements to draw at most, all of them if 0 or None
        (default: :data:`default_max_points`). A file holding more is thinned,
        which the title of the map says.
    size : float, optional
        Size of a measurement on the map (default: 1). The measurements are
        drawn as points; the default suits a whole orbit, a single file of a
        few seconds takes a larger one.
    **kwargs
        Passed on to the read method of the reader.

    Returns
    -------
    m : eomaps.Maps
        The map, so that it can be changed or saved afterwards.
    """
    import eomaps

    data, metadata = read_szf(filename, **kwargs)
    beams = default_beams if beams is None else beams
    dataframe = to_dataframe(data, beams)

    if parameter not in dataframe:
        raise KeyError(
            f"No field {parameter} in the file, it has "
            f"{', '.join(sorted(c for c in dataframe if c != 'beam'))}.")

    m = eomaps.Maps(crs=4326)
    m.add_feature.preset.ocean()
    m.add_feature.preset.coastline()

    plotted = thin(dataframe, max_points)

    m.set_data(data=plotted, parameter=parameter, x="lon", y="lat", crs=4326)
    # points rather than the default shape, which estimates a radius from the
    # data and draws the beams of a SZF file too faintly to read
    m.set_shape.scatter_points(size=size)
    m.plot_map(vmin=vmin, vmax=vmax, cmap=cmap)
    m.add_colorbar(label=parameter)

    heading = title or Path(filename).name
    if len(plotted) < len(dataframe):
        heading += (f"\n{len(plotted):,} of {len(dataframe):,} measurements, "
                    f"every {int(np.ceil(len(dataframe) / len(plotted)))}. one")
    m.ax.set_title(heading, fontsize=8)

    shown = [f for f in annotated_fields if f in dataframe]

    def annotation(m, ID, val, pos, ind):
        """What the file says about the measurement which was picked."""
        measurement = m.data.loc[ID]
        lines = [f"beam: {measurement['beam']}",
                 f"lon: {measurement['lon']:.3f}",
                 f"lat: {measurement['lat']:.3f}",
                 f"time: {measurement['time']}"]

        for name in shown:
            value = measurement[name]
            lines.append(f"{name}: {value:.3f}" if isinstance(value, float)
                         else f"{name}: {value}")

        return "\n".join(lines)

    m.cb.pick.attach.annotate(text=annotation)

    return m
