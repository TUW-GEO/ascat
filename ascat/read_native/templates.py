import numpy as np

def template_ASCATL1():
    """
    Generic lvl1b template. (from generic IO: ASCRS009)
    """
    metadata = {'temp_name': 'ASCATL1'}

    struct = np.dtype([('jd', np.double),
                       ('sat_id', np.byte),
                       ('abs_orbit_nr', np.uint32),
                       ('node_num', np.uint8),
                       ('line_num', np.uint16),
                       ('dir', np.dtype('S1')),
                       ('swath', np.byte),
                       ('azif', np.float32),
                       ('azim', np.float32),
                       ('azia', np.float32),
                       ('incf', np.float32),
                       ('incm', np.float32),
                       ('inca', np.float32),
                       ('sigf', np.float32),
                       ('sigm', np.float32),
                       ('siga', np.float32),
                       ('kpf', np.float32),
                       ('kpm', np.float32),
                       ('kpa', np.float32),
                       ('num_obs', np.ubyte),
                       ('usable_flag', np.uint8)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset

def template_ASCATL2():
    """
    Generic lvl2 template.
    """
    metadata = {'temp_name': 'ASCATL2'}

    struct = np.dtype([('jd', np.double),
                       ('abs_orbit_nr', np.uint32),
                       ('node_num', np.uint8),
                       ('line_num', np.uint16),
                       ('dir', np.dtype('S1')),
                       ('swath', np.byte),
                       ('azif', np.float32),
                       ('azim', np.float32),
                       ('azia', np.float32),
                       ('incf', np.float32),
                       ('incm', np.float32),
                       ('inca', np.float32),
                       ('sigf', np.float32),
                       ('sigm', np.float32),
                       ('siga', np.float32),
                       ('sm', np.float32),
                       ('sm_noise', np.float32),
                       ('sm_sensitivity', np.float32),
                       ('sig40', np.float32),
                       ('sig40_noise', np.float32),
                       ('slope40', np.float32),
                       ('slope40_noise', np.float32),
                       ('dry_backscatter', np.float32),
                       ('wet_backscatter', np.float32),
                       ('mean_surf_sm', np.float32),
                       ('rain_flag', np.uint8),
                       ('correction_flag', np.uint8),
                       ('processing_flag', np.uint16),
                       ('aggregated_quality_flag', np.uint8),
                       ('snow_cover_probability', np.float32),
                       ('frozen_soil_probability', np.float32),
                       ('innudation_or_wetland', np.float32),
                       ('topographical_complexity', np.float32)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset

