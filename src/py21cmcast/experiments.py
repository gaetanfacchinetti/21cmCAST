##################################################################################
# This file is part of 21cmCAST.
#
# Copyright (c) 2022, Gaétan Facchinetti
#
# 21cmCAST is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. 21cmCAST is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
##################################################################################


import numpy  as np
from astropy import units
from scipy   import interpolate

PY21CMSENSE = True

## This part of the code rely on 21cmSense however as many of
## 21cmCAST abilities do not rely on 21cmSense we allow using 
## 21cmCAST without 21cmSense functionalities


# create a simple observations class
class ObservationSet:
    
    def __init__(self, name, *, observations = None):
        self._observations = observations
        self._name = name

    @property
    def observations(self):
        return self._observations
    
    @property 
    def name(self):
        return self._name


try :

    import py21cmsense as p21s
    

    def default_HERA_observatories(z_arr:np.ndarray):

        ## Define the layout for the 
        hera_layout = p21s.antpos.hera(
            hex_num = 11,             # number of antennas along a side
            separation= 14 * units.m,  # separation between antennas (in metres)
            row_separation=12.12 * units.m        # separation between rows
        )

        hera_observatories = []
        
        for z in z_arr:
            beam = p21s.beam.GaussianBeam(
               frequency = 1420.40575177 * units.MHz / (1+z),  # just a reference frequency
              dish_size = 14 * units.m
            )

            hera_observatories.append(p21s.Observatory(
                antpos = hera_layout,
                beam = beam,
                latitude = 0.536189 * units.radian,
                Trcv = 100 * units.K
            ))

        return hera_observatories


    def default_observation_set(z_arr:np.ndarray):
  
        default_observatories = default_HERA_observatories(z_arr)

        observations = []
        for observatory in default_observatories:

                observations.append(p21s.Observation(
                    observatory   = observatory,
                    n_channels    = 80, 
                    bandwidth     = 8 * units.MHz,
                    time_per_day  = 6 * units.hour,   # Number of hours of observation per day
                    n_days        = 166.6676,         # Number of days of observation
                ))

        return ObservationSet("default_HERA", observations = observations)
        

    class ComputedModel(p21s.theory.TheoryModel):

        use_littleh = False

        def __init__(self, k_21:np.ndarray, delta_21_sqr:np.ndarray) -> None:
            
            self.k = k_21
            self.delta_sqr = delta_21_sqr

            # interpolation function for this redshift
            self.interp = interpolate.interp1d(k_21, delta_21_sqr / (units.mK)**2)

        # Note that here z is a dummy variable
        def delta_squared(self, z: float, k: np.ndarray) -> units.Quantity[units.mK**2]:
            return self.interp((k * units.Mpc)) * (units.mK)**2



    def extract_noise_from_fiducial(k_in, dsqr_in, k_out, observation) :
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

        #sensitivity       = p21s.PowerSpectrum(observation=observation, k_21 = k_in / units.Mpc, 
        #                                        delta_21 = dsqr_in * (units.mK**2), 
        #                                        foreground_model='moderate') 
        
        ## implementation compatible with the new version of 21cmSense
        theory_model = ComputedModel(k_21 = k_in / units.Mpc, delta_21_sqr = dsqr_in * (units.mK**2))
        sensitivity  = p21s.PowerSpectrum(observation = observation, theory_model = theory_model, foreground_model = 'moderate')
        
        k_sens            = sensitivity.k1d.value * sensitivity.cosmo.h
        std_21cmSense     = sensitivity.calculate_sensitivity_1d(thermal = True, sample = True).value

        std = interpolate.interp1d(k_sens, std_21cmSense, bounds_error=False, fill_value=np.inf)(k_out)

        return std

except ImportError:

    PY21CMSENSE = False
    
    def define_HERA_observation(z):
        return None

    def extract_noise_from_fiducial(k_in, dsqr_in, k_out, observation):
        return None