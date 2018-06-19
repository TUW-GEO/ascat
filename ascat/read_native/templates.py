import numpy as np

def template_SZF__001():
    """
    Re-sampled backscatter template 001. (from generic IO)

    beam_num
    - 1 Left Fore Antenna
    - 2 Left Mid Antenna
    - 3 Left Aft Antenna
    - 4 Right Fore Antenna
    - 5 Right Mid Antenna
    - 6 Right Aft Antenna

    as_des_pass
    - 0 Ascending
    - 1 Descending

    swath_indicator
    - 0 Left
    - 1 Right
    """
    metadata = {'temp_name': 'SZF__001'}

    struct = np.dtype([('jd', np.double),
                       ('spacecraft_id', np.int8),
                       ('processor_major_version', np.int16),
                       ('processor_minor_version', np.int16),
                       ('format_major_version', np.int16),
                       ('format_minor_version', np.int16),
                       ('degraded_inst_mdr', np.int8),
                       ('degraded_proc_mdr', np.int8),
                       ('sat_track_azi', np.float32),
                       ('as_des_pass', np.int8),
                       ('swath_indicator', np.int8),
                       ('azi', np.float32),
                       ('inc', np.float32),
                       ('sig', np.float32),
                       ('lat', np.float32),
                       ('lon', np.float32),
                       ('beam_number', np.int8),
                       ('land_frac', np.float32),
                       ('flagfield_rf1', np.uint8),
                       ('flagfield_rf2', np.uint8),
                       ('flagfield_pl', np.uint8),
                       ('flagfield_gen1', np.uint8),
                       ('flagfield_gen2', np.uint8),
                       ('f_usable', np.uint8),
                       ('f_land', np.uint8)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset


def template_SZX__002():
    """
    Re-sampled backscatter template 002. (from generic IO)
    """
    metadata = {'temp_name': 'SZX__002'}

    struct = np.dtype([('jd', np.double),
                       ('spacecraft_id', np.int8),
                       ('abs_orbit_nr', np.uint32),
                       ('processor_major_version', np.int16),
                       ('processor_minor_version', np.int16),
                       ('format_major_version', np.int16),
                       ('format_minor_version', np.int16),
                       ('degraded_inst_mdr', np.int8),
                       ('degraded_proc_mdr', np.int8),
                       ('sat_track_azi', np.float32),
                       ('as_des_pass', np.int8),
                       ('swath_indicator', np.int8),
                       ('azi', np.float32, 3),
                       ('inc', np.float32, 3),
                       ('sig', np.float32, 3),
                       ('lat', np.float32),
                       ('lon', np.float32),
                       ('kp', np.float32, 3),
                       ('node_num', np.int16),
                       ('line_num', np.int32),
                       ('num_val', np.uint32, 3),
                       ('f_kp', np.int8, 3),
                       ('f_usable', np.int8, 3),
                       ('f_f', np.float32, 3),
                       ('f_v', np.float32, 3),
                       ('f_oa', np.float32, 3),
                       ('f_sa', np.float32, 3),
                       ('f_tel', np.float32, 3),
                       ('f_ref', np.float32, 3),
                       ('f_land', np.float32, 3)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset

def template_SMR__001():
    """
    Re-sampled SMR data template 001. (from generic IO)
    """
    metadata = {'temp_name': 'SMR__001'}

    struct = np.dtype([('jd', np.double),
                       ('lon', np.float32),
                       ('lat', np.float32),
                       ('spacecraft_id', np.ubyte),
                       ('sat_track_azi', np.float32),
                       ('as_des_pass', np.int8),
                       ('swath_indicator', np.ubyte),
                       ('node_num', np.int16),
                       ('azi', np.float32, 3),
                       ('inc', np.float32, 3),
                       ('sig', np.float32, 3),
                       ('kp', np.float32, 3),
                       ('f_land', np.uint16, 3),
                       ('data_quality', np.int16),
                       ('warp_nrt_version', np.uint16),
                       ('param_db_version', np.uint16),
                       ('ssm', np.float32),
                       ('ssm_noise', np.float32),
                       ('norm_sigma', np.float32),
                       ('norm_sigma_noise', np.float32),
                       ('slope', np.float32),
                       ('slope_noise', np.float32),
                       ('dry_ref', np.float32),
                       ('wet_ref', np.float32),
                       ('mean_ssm', np.float32),
                       ('ssm_sens', np.float32),
                       ('correction_flag', np.uint8),
                       ('processing_flag', np.uint16),
                       ('aggregated_flag', np.uint8),
                       ('snow', np.uint8),
                       ('frozen', np.uint8),
                       ('wetland', np.uint8),
                       ('topo', np.uint8)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset


def template_ASCRS009():
    """
    Generic lvl1b template. (from generic IO)
    """
    metadata = {'temp_name': 'ASCRS009'}

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

