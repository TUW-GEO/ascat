# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
The SCA products carry two quality related fields per measurement. The first,
"flag_generic", is a bit field of individual processing flags. The second,
"flag_quality", is the summary computed by the ground segment, with the values
0, 1 and 2 for nominal, close to nominal and far from nominal data.
"""

import numpy as np

#: Flag categories.
NOMINAL = 0
AMBER = 1
RED = 2

#: Flag bit definitions, "name": (bit, category).
#:
#: The bit numbers and meanings are taken from the EPS-SG SCA Level 1b product
#: format specification. Bits 2, 18, 19 and 24 to 26 are unused.
#:
#: Bits 0 to 23 flag non-nominal situations which may affect the quality of the
#: backscatter, bits 27 to 30 classify the measurement and are therefore of the
#: nominal category.
#:
#: The categories follow the naming of the flags themselves: a poor estimate
#: ("f" flags) degrades a measurement, a very poor estimate ("v" flags) and any
#: value outside its limits renders it unusable. This gives the same categories
#: that :func:`ascat.read_native.eps_native.set_flags_fmv13` assigns for the few
#: situations both instruments flag, namely a manoeuvre, a non-nominal
#: attitude, a power-gain estimate outside its limits, interference from the
#: solar array and the use of a predicted orbit. The remaining flags have no
#: ASCAT counterpart, either because they concern parts of the instrument that
#: only SCA has or because ASCAT reports a comparable situation differently.
flag_bits = {
    "pof": (0, NOMINAL),        # predicted orbit file used
    "man": (1, RED),            # manoeuvre taking place
    "yaw": (3, RED),            # satellite is not in yaw steering mode
    "trp": (4, RED),            # transponder signal is present
    "fnoise": (5, AMBER),       # noise estimate is poor
    "vnoise": (6, RED),         # noise estimate is very poor
    "rfi": (7, RED),            # noise outlier is present
    "fan": (8, AMBER),          # ancillary data is poor
    "van": (9, RED),            # ancillary data is very poor
    "ctable": (10, RED),        # characterisation table limits exceeded
    "fpg": (11, AMBER),         # power-gain estimate is poor
    "vpg": (12, RED),           # power-gain estimate is very poor
    "xpg": (13, RED),           # power-gain out of limits
    "fwg": (14, AMBER),         # waveguide loss estimate is poor
    "vwg": (15, RED),           # waveguide loss estimate is very poor
    "wtable": (16, RED),        # waveguide table limits exceeded
    "oor": (17, RED),           # echo is out of range
    "fnum": (20, AMBER),        # number of samples in window is low
    "vnum": (21, RED),          # number of samples in window is very low
    "fneg": (22, AMBER),        # negative backscatter in resampled data
    "fsol": (23, AMBER),        # possible interference from solar array
    "land": (27, NOMINAL),      # measurement over land
    "water": (28, NOMINAL),     # measurement over water
    "asc": (29, NOMINAL),       # ascending pass
    "desc": (30, NOMINAL),      # descending pass
}


def category_masks(flag_bits, rfi_red=True, ignore=()):
    """
    Build the bit masks of the amber and the red category.

    Parameters
    ----------
    flag_bits : dict
        Flag bit definitions, "name": (bit, category).
    rfi_red : bool, optional
        Treat the "rfi" flag, which reports a noise outlier, as a red flag
        (default: True). If False it is left out of the summary, in the same
        way the ASCAT readers allow ignoring their noise out of limits flag.
    ignore : iterable of str, optional
        Names of flags to leave out of the summary (default: ()).

    Returns
    -------
    amber_mask : int
        Bit mask of all flags of the amber category.
    red_mask : int
        Bit mask of all flags of the red category.

    Raises
    ------
    KeyError
        If a flag is unknown or has no bit number assigned.
    """
    unknown = set(ignore) - set(flag_bits)
    if unknown:
        raise KeyError(f"Unknown flag(s): {', '.join(sorted(unknown))}")

    if not rfi_red:
        ignore = (*ignore, "rfi")

    masks = {AMBER: 0, RED: 0}

    for name, (bit, category) in flag_bits.items():
        if name in ignore or category == NOMINAL:
            continue
        if bit is None:
            raise KeyError(
                f"Flag '{name}' has no bit number assigned. Add it to the flag "
                "bit definitions, see ascat.eumetsat.sca.flags.flag_bits.")
        masks[category] |= 1 << bit

    return masks[AMBER], masks[RED]


def set_flags(flag_generic, rfi_red=True, ignore=(), flag_bits=None):
    """
    Compute a summary flag for each measurement with a value of 0, 1 or 2
    indicating nominal, slightly degraded or severely degraded data.

    Parameters
    ----------
    flag_generic : numpy.ndarray
        Processing flags of the SCA product, as stored in "flag_generic".
    rfi_red : bool, optional
        Treat the "rfi" flag, which reports a noise outlier, as a red flag
        (default: True). If False it is left out of the summary, in the same
        way the ASCAT readers allow ignoring their noise out of limits flag.
    ignore : iterable of str, optional
        Names of flags to leave out of the summary (default: ()).
    flag_bits : dict, optional
        Flag bit definitions, "name": (bit, category). Defaults to
        :data:`flag_bits`, pass a different table to use your own categories
        without changing the module.

    Returns
    -------
    f_usable : numpy.ndarray
        Flag indicating nominal (0), slightly degraded (1) or
        severely degraded (2).
    """
    if flag_bits is None:
        flag_bits = globals()["flag_bits"]

    amber_mask, red_mask = category_masks(flag_bits, rfi_red=rfi_red,
                                          ignore=ignore)

    f_usable = np.zeros(np.shape(flag_generic), dtype=np.int8)
    f_usable[(flag_generic & amber_mask) != 0] = AMBER
    f_usable[(flag_generic & red_mask) != 0] = RED

    return f_usable
