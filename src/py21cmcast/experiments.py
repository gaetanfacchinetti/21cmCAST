import py21cmsense as p21s
import numpy       as np

from astropy import units
from scipy   import interpolate


def define_HERA_observation(z):
    """ 
    Define a set of HERA observation objects (see 21cmSense) 
    according to an array of observation redshifts 

    Parameters
    ----------
        z_arr: list (or numpy array) of floats
            redshifts at which the observations are done    
    """

     ## Define the layout for the 
    hera_layout = p21s.antpos.hera(
        hex_num = 11,             # number of antennas along a side
        separation= 14 * units.m,  # separation between antennas (in metres)
        dl=12.12 * units.m        # separation between rows
    )

    hera = []


    beam = p21s.beam.GaussianBeam(
        frequency = 1420.40575177 * units.MHz / (1+z),  # just a reference frequency
        dish_size = 14 * units.m
    )

    hera = p21s.Observatory(
        antpos = hera_layout,
        beam = beam,
        latitude = 0.536189 * units.radian,
        Trcv = 100 * units.K
    )

    observation = p21s.Observation(
        observatory   = hera,
        n_channels    = 80, 
        bandwidth     = 8 * units.MHz,
        time_per_day  = 6 * units.hour,   # Number of hours of observation per day
        n_days        = 166.6667,         # Number of days of observation
    )

    return observation



def extract_noise_from_fiducial(k, dsqr, observation) :
    """
    Give the noise associated to power spectra delta_arr

    Params:
    -------
    k       : list of floats 
        array of modes k in [Mpc^{-1}]
    dsqr    : list of floats 
        Power spectrum in [mK^2] ordered with the modes k in 
    observation : Observation object (c.f. 21cmSense)

    Returns:
    --------
    k_sens       : list of list of floats
    std_sens     : the standard deviation of the sensitivity [mK]

    """


    sensitivity       = p21s.PowerSpectrum(observation=observation, k_21 = k / units.Mpc, 
                                            delta_21 = dsqr * (units.mK**2), 
                                            foreground_model='moderate') 
    
    k_sens            = sensitivity.k1d.value * p21s.config.COSMO.h
    std_21cmSense     = sensitivity.calculate_sensitivity_1d(thermal = True, sample = True).value

    #print(k, k_sens, std_21cmSense)

    std = interpolate.interp1d(k_sens, std_21cmSense, bounds_error=False, fill_value=np.inf)(k)

    return std

