##################################################################################
# This file is part of 21cmCAST.
#
# Copyright (c) 2023, Gaétan Facchinetti
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
#
# -------------------------------------------------------------------------------
#
# Parts of this code have been copied and modified 
# from https://github.com/charlottenosam/21cmfish
# 
# - MIT License
# -
# - Copyright (c) 2019, Charlotte Mason
# - 
# - Permission is hereby granted, free of charge, to any person obtaining a copy
# - of this software and associated documentation files (the "Software"), to deal
# - in the Software without restriction, including without limitation the rights
# - to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# - copies of the Software, and to permit persons to whom the Software is
# - furnished to do so, subject to the following conditions:
# - 
# - The above copyright notice and this permission notice shall be included in all
# - copies or substantial portions of the Software.
# - 
# - THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# - IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# - FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# - AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# - LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# - OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# - SOFTWARE.
##################################################################################

import glob


import numpy as np
import pickle
from scipy   import interpolate
from astropy import units
import warnings

PY21CMFAST = True

try:
    import py21cmfast as p21f
except ImportError:
    PY21CMFAST = False
    p21f = None
    warnings.warn("21cmFAST could not be found! 21cmCAST can work on cached data but will not be able to process any new lightcones.")

from py21cmcast import power         as p21c_p
from py21cmcast import tools         as p21c_tools
from py21cmcast import experiments   as p21c_exp



def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    _filename = 'py21cmcast/' + filename.split("/")[-1]
    return ' %s:%s: %s (%s)\n' % (_filename, lineno, message, category.__name__)

warnings.formatwarning = warning_on_one_line


def compare_arrays(array_1 : np.ndarray, array_2 : np.ndarray, eps : float):
    """
    ## Compare arrays to a certain precision

    Parameters
    ----------
    array_1 : np.ndarray
        First array to compare
    array_1 : np.ndarray
        Second array to compare
    eps : float
        Relative precision asked for the comparision

    Returns:
    --------
    bool
        True if arrays are equal to the precision eps, False otherwise
    """

    if len(array_1) != len(array_2):
        return False
    
    return bool(np.all(2* np.abs((array_1 - array_2)/(array_1 + array_2)) < eps))
    



class Run:

    """
    # Class to define and analyse one run (and its associated lightcone)

    Fetch/Compute and store derived quantities such as global quantities
    or the power_spectrum of the brightness temperature.
    """

    def __init__(self, dir_path : str, lightcone_name : str,  z_bins : list = None, z_array : list = None, 
                k_bins : list = None, logk : bool = False, p : float = 0., load : bool = True, save : bool = True, 
                 verbose : bool  = True) -> None: 
        """
        Parameters
        ----------
        dir_path : str
            Path to the directory where the lightcone is saved
        lightcone_name : str
            Name of the lightcone file
        z_bins : list
            Array of redshift bin edges
        z_centers : list
            Array of redshift bin centers
        k_bins : list
            Array of mode bin edges
        logk : bool
            Indicates that the k mode edges are logarithmically distributed
        p : float, optional
            Associated value of the parameter associated to the run
        load : bool, optional
            When True, first search for a cached precomputed tables of
            the quantities derived in the class (power_spectrum, noise, etc.)
            The cached tables have to be compatible with k_bins and z_bins
            (i.e. preomputed for the same arrays z_bins and k_bins)
            Default is True
        save : bool, optional
            When True, save the derived quantities in a cached table
            Default is True
        verbose : bool, optional
            When true, outputs information for the user
            Default is True
        """
        
        self._dir_path        = dir_path
        self._name            = lightcone_name
        self._filename_data   = self._dir_path + '/cache/Table_' + self._name + '.npz'    
        self._filename_params = self._dir_path + '/cache/Param_' + self._name + '.pkl' 

        # Create the database directory if it does not already exists
        p21c_tools.make_directory(self._dir_path + "/cache", clean_existing_dir=False)
        p21c_tools.make_directory(self._dir_path + "/power_spectra", clean_existing_dir=False)
        p21c_tools.make_directory(self._dir_path + "/derivatives", clean_existing_dir=False)
        p21c_tools.make_directory(self._dir_path + "/global_quantities", clean_existing_dir=False)

        self._verbose = verbose

        ## if load true we try loading precomputed tables
        if load is True : 

            _load_successfull = self._load()

            _params_match = True

            # If we load the file correctly and if the input parameter correspond
            # then we don't need to go further and can skip the full computation again
            if _load_successfull is True:
                if z_bins is not None:
                    if compare_arrays(self._z_bins, z_bins, 1e-5) is not True:
                        _params_match = False
                        warnings.warn("z-bins in input are different than the one used to precompute the tables")
                if k_bins is not None:
                    if compare_arrays(self._k_bins, k_bins, 1e-5) is not True:
                        _params_match = False
                        warnings.warn("k-bins in input are different than the one used to precompute the tables")
                if z_array is not None:
                    if compare_arrays(self._z_array, z_array, 1e-5) is not True:
                        _params_match = False
                        warnings.warn("z-array in input is different than the one used to precompute the tables")

                if _params_match is True:
                    return None
                else:
                    print("recomputing the tables with the new bins")

            if (z_bins is None or k_bins is None or z_array is None) and _load_successfull is False:
                raise ValueError("Need to pass z_bins and k_bins as inputs")


        if load is False or _load_successfull is False or _params_match is False:

            self._z_bins    = z_bins
            self._k_bins    = k_bins
            self._z_array   = z_array
            self._logk      = logk
            self._p         = p

            # Get the power spectrum from the Lightcone
            if PY21CMFAST:
                self._lightcone   = p21f.LightCone.read(self._dir_path + "/" + self._name)
            else : 
                ImportError("21cmFAST is needed to read new lightcones")

            self._astro_params = dict(self._lightcone.astro_params.self)
            self._user_params  = dict(self._lightcone.user_params.self)
            self._cosmo_params = dict(self._lightcone.cosmo_params.self)
            self._flag_options = dict(self._lightcone.flag_options.self)

            # Compute the power spectrum and fetch global quantities
            self._get_power_spectra()    
            self._get_global_quantities()

            # Compute the optical depth to reionization
            if PY21CMFAST:
                try: 
                    self._tau_ion = p21f.compute_tau(redshifts=self._z_glob, global_xHI = self.xH_box, user_params=self._user_params, cosmo_params=self._cosmo_params)
                except:
                    warnings.warn("Impossible to compute the optical depth to reionization tau_ion")
                    self._tau_ion = 0.0
            else : 
                ImportError("21cmFAST is needed to evaluate the optical depth to reionization")

            if save is True:
                self._save()


    def _get_power_spectra(self):

        self._lc_redshifts        = self._lightcone.lightcone_redshifts
        self._chunk_indices       = [np.argmin(np.abs(self._lc_redshifts - z)) for z in self._z_bins]
        _, _data_arr  = p21c_p.compute_powerspectra_1D(self._lightcone, chunk_indices = self._chunk_indices, 
                                                                n_psbins=self._k_bins.value, logk=self._logk, 
                                                                remove_nans=False, vb=False)
        
        self._power_spectrum   = np.array([data['delta'] for data in _data_arr])
        self._ps_poisson_noise = np.array([data['err_delta'] for data in _data_arr])

        _k_array = np.array([data['k'] for data in _data_arr])
        assert np.all(np.abs(np.diff(_k_array, axis=0)[0]/_k_array) <= 1e-3)
        self._k_array = _k_array[0]

    
    def _get_global_quantities(self):

        _lc_glob_redshifts = self._lightcone.node_redshifts
        self._z_glob       = np.linspace(_lc_glob_redshifts[-1], _lc_glob_redshifts[0], 200)
  
        _global_signal = self._lightcone.global_quantities.get('brightness_temp', np.zeros(len(_lc_glob_redshifts), dtype=np.float64))
        _xH_box        = self._lightcone.global_quantities.get('xH_box', np.zeros(len(_lc_glob_redshifts), dtype=np.float64))
        _x_e_box       = self._lightcone.global_quantities.get('x_e_box', np.zeros(len(_lc_glob_redshifts), dtype=np.float64))
        _Ts_box        = self._lightcone.global_quantities.get('Ts_box', np.zeros(len(_lc_glob_redshifts), dtype=np.float64))
        _Tk_box        = self._lightcone.global_quantities.get('Tk_box', np.zeros(len(_lc_glob_redshifts), dtype=np.float64))

        self._global_signal = interpolate.interp1d(_lc_glob_redshifts, _global_signal)(self._z_glob)
        self._xH_box        = interpolate.interp1d(_lc_glob_redshifts, _xH_box)(self._z_glob)
        self._x_e_box       = interpolate.interp1d(_lc_glob_redshifts, _x_e_box)(self._z_glob)
        self._Ts_box        = interpolate.interp1d(_lc_glob_redshifts, _Ts_box)(self._z_glob)
        self._Tk_box        = interpolate.interp1d(_lc_glob_redshifts, _Tk_box)(self._z_glob)


    def _save(self):
        """
        ##  Saves all the attributes of the class to be easily reload later if necessary

        numpy arrays are saved in an .npz format
        scalar parameters / attributes are saved in a dictionnary in a .pkl format
        """

        with open(self._filename_data, 'wb') as file: 
            np.savez(file, power_spectrum     = self.power_spectrum,
                            ps_poisson_noise  = self.ps_poisson_noise,
                            global_signal     = self.global_signal,
                            chunk_indices     = self.chunk_indices,
                            xH_box            = self.xH_box,
                            x_e_box           = self.x_e_box,
                            Ts_box            = self.Ts_box,
                            Tk_box            = self.Tk_box,
                            z_array           = self.z_array,
                            k_array           = self.k_array,
                            z_bins            = self.z_bins,
                            k_bins            = self.k_bins,
                            z_glob            = self.z_glob, 
                            tau_ion           = self.tau_ion)
        
        # Prepare the dictionnary of parameters
        param_dict = {'logk' : self.logk, 'p' : self.p, 
                    'astro_params': self.astro_params, 
                    'user_params': self.user_params,
                    'flag_options': self.flag_options,
                    'cosmo_params' : self.cosmo_params}
        
        with open(self._filename_params, 'wb') as file:
            pickle.dump(param_dict, file)
        

    def _load(self):
        """
        ##  Loads all the attributes of the class
        """

        data   = None
        params = None

        try:

            with open(self._filename_data, 'rb') as file: 
                data = np.load(file)
                
                self._power_spectrum    = data['power_spectrum']
                self._ps_poisson_noise  = data['ps_poisson_noise']
                self._chunk_indices     = data['chunk_indices']
                self._z_array           = data['z_array']
                self._k_array           = data['k_array']
                self._z_bins            = data['z_bins']
                self._k_bins            = data['k_bins']  / units.Mpc
                self._global_signal     = data['global_signal']
                self._xH_box            = data['xH_box']
                self._x_e_box           = data.get('x_e_box', None)
                self._Ts_box            = data.get('Ts_box', None)
                self._Tk_box            = data.get('Tk_box', None)
                self._z_glob            = data['z_glob']
                self._tau_ion           = data['tau_ion'].item()


            with open(self._filename_params, 'rb') as file:
                params = pickle.load(file)

                self._logk          = params['logk']
                self._p             = params['p']
                self._astro_params  = params['astro_params']
                self._user_params   = params['user_params']
                self._flag_options  = params['flag_options']  
                self._cosmo_params  = params['cosmo_params']

            return True

        except FileNotFoundError:
            if self._verbose is True:
                print("No existing data found for " + self._name)
            
            return False
        

    def plot_power_spectrum(self, std = None, figname = None, plot=True, ps_modeling_noise=None) :  

        error = self.ps_poisson_noise
        if ps_modeling_noise is not None :
            error = np.sqrt(self.ps_poisson_noise**2 + self.ps_modeling_noise**2)

        fig = p21c_tools.plot_func_vs_z_and_k(self.z_array, self.k_array, self.power_spectrum, 
                                                func_err = error,
                                                std = std, title=r'$\Delta_{21}^2 ~{\rm [mK^2]}$', 
                                                xlim = [self._k_bins[0].value, self._k_bins[-1].value], xlog=self._logk, ylog=True)
        
        if plot is True : 

            if figname is None:
                figname = self._dir_path + "/power_spectra/power_spectrum.pdf"
        
            fig.savefig(figname, bbox_layout='tight')

        return fig

    @property
    def power_spectrum(self):
        return self._power_spectrum

    @property
    def ps_poisson_noise(self):
        return self._ps_poisson_noise
    
    @property
    def global_signal(self):
        return self._global_signal
    
    @property
    def xH_box(self):
        return self._xH_box
    
    @property
    def x_e_box(self):
        return self._x_e_box
    
    @property
    def Ts_box(self):
        return self._Ts_box

    @property
    def Tk_box(self):
        return self._Tk_box
    
    @property
    def z_glob(self):
        return self._z_glob

    @property
    def z_bins(self):
        return self._z_bins

    @property
    def k_bins(self): 
        return self._k_bins

    @property
    def z_array(self):
        return self._z_array

    @property
    def k_array(self):
        return self._k_array

    @property
    def logk(self): 
        return self._logk

    @property
    def chunk_indices(self):
        return self._chunk_indices

    @property
    def p(self):
        return self._p

    ## Lighcone properties
    @property
    def astro_params(self):
        return self._astro_params

    @property
    def user_params(self):
        return self._user_params

    @property
    def flag_options(self):
        return self._flag_options

    @property
    def cosmo_params(self):
        return self._cosmo_params

    @property
    def tau_ion(self):
        return self._tau_ion





class Fiducial(Run): 
    """
    # Class to define the Fiducial run
    """

    def __init__(self, dir_path : str, z_bins : list, z_array : list, k_bins : list, logk : bool,
                frac_noise : float = 0.2, rs : int = None, ps_modeling_noise : np.ndarray = None, 
                load : bool = True, save : bool = True, verbose : bool = False, **kwargs) -> None:

        """
        Parameters
        ----------
        dir_path : str
            Path to the directory where the FIDUCIAL lightcone is saved
        z_bins : list
            Array of redshift bin edges
        z_array : list
            Array of redshift bin centers
        k_bins : list
            Array of mode bin edges
        logk : bool
            Indicates that the k mode edges are logarithmically distributed
        frac_noise : float, optional
            Percentage level of modeling noise to add to the power spectrum
            Default is 0.2
        rs : int, optional
            Value of the random seed used in 21cmFAST to compute the FIDUCIAL
            Default is None
        ps_modeling_noise : np.ndarray, optional
            Overrides frac_noise by giving the modelign noise in a 2D table depending
            on the redshift bin and the mode bin (of the same size as the power_spectrum).
            Default is None
        load : bool, optional
            When True, first search for a cached precomputed tables of
            the quantities derived in the class (power_spectrum, noise, etc.)
            The cached tables have to be compatible with k_bins and z_bins
            (i.e. preomputed for the same arrays z_bins and k_bins)
            Default is True
        save : bool, optional
            When True, save the derived quantities in a cached table
            Default is True
        verbose : bool, optional
            When true, outputs information for the user
            Default is True

   
        """

        self._dir_path             = dir_path
        self._filename_exp_noise   = self._dir_path + '/cache/Table_exp_noise'
        self._verbose              = verbose

        # get the lightcones filenames
        _loading_success = True
        if load is True:
            # here we look in the cache folder
            _str_file_name = self._dir_path + "/cache/Table_Lightcone_rs*_FIDUCIAL.h5.npz" if (rs is None) else self._dir_path + "/cache/Table_Lightcone_rs" + str(rs) + "_FIDUCIAL.h5.npz"
            _lightcone_file_name = glob.glob(_str_file_name)
            
            if len(_lightcone_file_name) == 1:
                _loading_success = True
                _file_name = (_lightcone_file_name[0].split('/')[-1])[6:-4] # remove the dir_path and 'Table_' ... '.npz'
            else:
                _loading_success = False

          # if we do not load, one has to find a lightcone for the fiducial
        if load is False or _loading_success is False : 
            _str_file_name = self._dir_path + "/Lightcone_rs*_FIDUCIAL.h5" if (rs is None) else self._dir_path + "/Lightcone_rs" + str(rs) + "_FIDUCIAL.h5"
            _lightcone_file_name = glob.glob(_str_file_name)
            _file_name = _lightcone_file_name[0].split('/')[-1] 

        assert len(_lightcone_file_name) == 1, 'No fiducial lightcone found or too many'
        
        self._rs   = _file_name.split('_')[-2][2:]

        # Initialise the parent object
        super().__init__(self._dir_path, _file_name, z_bins, z_array, k_bins, logk, load=load, save=save, verbose = verbose)
        
        # Getting the noise associated to the fiducial
        self._frac_noise         = frac_noise
        self._ps_modeling_noise  = ps_modeling_noise if (ps_modeling_noise is not None) else self._frac_noise * self._power_spectrum
        self._astro_params       = self._astro_params

        self._is_ps_sens_computed = False


    def _save_ps_exp_noise(self, observation):
         
         with open(self._filename_exp_noise + '_' + observation + '.npz', 'wb') as file: 
            np.savez(file, ps_exp_noise         = self.ps_exp_noise,
                            z_array             = self.z_array,
                            k_array             = self.k_array,
                            z_bins              = self.z_bins,
                            k_bins              = self.k_bins)
            

    def _load_ps_exp_noise(self, observation_set):

        data   = None

        try:

            with open(self._filename_exp_noise + '_' + observation_set + '.npz' , 'rb') as file: 
                data = np.load(file)

                _z_array           = data['z_array']
                _k_array           = data['k_array']
                _z_bins            = data['z_bins']
                _k_bins            = data['k_bins']  / units.Mpc

                # if the binning are compatibles then we can load the sensitivity
                if compare_arrays(_z_array, self._z_array, 1e-3) \
                    and compare_arrays(_k_array, self._k_array, 1e-3) \
                    and compare_arrays(_z_bins, self._z_bins, 1e-3) \
                    and compare_arrays(_k_bins, self._k_bins, 1e-3) :

                    self._ps_exp_noise      = data['ps_exp_noise']
                else:
                    return False

            return True

        except FileNotFoundError:
            if self._verbose is True:
                warnings.warn("No existing data found for " + observation_set + 'with this binning')

            return False

    @property
    def dir_path(self):
        return self._dir_path

    @property
    def astro_params(self):
        return self._astro_params

    @property
    def frac_noise(self):
        return self._frac_noise

    @frac_noise.setter
    def frac_noise(self, value):
        if value != self._frac_noise:
            warnings.warn("frac noise and modeling noise have been changed, all related quantities should be recomputed")
            self._frac_noise        = value
            self._ps_modeling_noise = self._frac_noise * self._power_spectrum

    @property
    def ps_exp_noise(self):
        return np.array(self._ps_exp_noise)
    
    @property
    def ps_modeling_noise(self):
        return self._ps_modeling_noise

    @ps_modeling_noise.setter
    def ps_modeling_noise(self, value):
        if value != self._ps_modeling_noise:
            warnings.warn("Modeling noise has been changed, all related quantities should be recomputed")
            warnings.warn("Changind modeling noise directly is not advised, you should maybe change it through frac_noise")
            self._ps_modeling_noise = value
    
    @property
    def rs(self):
        return self._rs
    
    @property
    def k_array_sens(self):
        return self._k_array_sens
    
    @property
    def power_spectrum_sens(self):
        return self._power_spectrum_sens
    

    def compute_power_spectrum_sensitivity(self):

        # Get the Lightcone quantity
        if PY21CMFAST:
            self._lightcone   = p21f.LightCone.read(self._dir_path + "/" + self._name)
        else : 
            ImportError("21cmFAST is needed to read new lightcones")

        # In order to compute the sensitivity we need the full range of z and k possible
        # We recompute a second power spectra for the sensitivity on a broader range
        # We use the same redshift chunks but let the array of mode free (and here logarithmically spaced by default)
        # This makes the code longer but it is necessary to have everything well defined and no numerical problems
        _, _data_arr   = p21c_p.compute_powerspectra_1D(self._lightcone, chunk_indices = self._chunk_indices, remove_nans=False, vb=False)

        self._k_array_sens         = np.array([data['k'] for data in _data_arr])
        self._power_spectrum_sens  = np.array([data['delta'] for data in _data_arr])

        self._is_ps_sens_computed = True



    def compute_sensitivity(self, observation_set = None, load = True, save = True):

        # Getting the name of the observation settings
        obs_name = "default_HERA" if (observation_set is None) else observation_set.name

        # Trying to load the senstivity from the observations sent
        # Here we just need the name of the observation object
        _load_succesfull = self._load_ps_exp_noise(obs_name) if load else False

        if _load_succesfull:
            print("Successfully loaded sensitivity for ", obs_name)
            return

        # If we cannot load then we need 21cmSense, make sure that we can find the module
        if p21c_exp.PY21CMSENSE is False:
            raise ImportError("The module 21cmSense could not be imported or was not found, sensitivity cannot be computed")
    
        # if observation_set is None we initialise it to default HERA
        _observation_set = p21c_exp.default_observation_set(self.z_array) if (observation_set is None) else observation_set
        
        assert (observation_set.observations is not None), ValueError("observations in ObservationSet must be a set of observations from 21cmSense")
        assert (len(_observation_set.observations) == len(self.z_array)), ValueError("We must have one observation for each redshift bin") 

        _std = None
        self._ps_exp_noise = None

        # if the power spectrum is not computed we do compute it
        if self._is_ps_sens_computed is False:
            self.compute_power_spectrum_sensitivity()

        _std = [p21c_exp.extract_noise_from_fiducial(self.k_array_sens[iz], self.power_spectrum_sens[iz], self.k_array, _observation_set.observations[iz]) for iz, _ in enumerate(self.z_array)]

        self._ps_exp_noise = _std
            
        if save is True:
            self._save_ps_exp_noise(_observation_set.name)


    
    def chi2_UV_luminosity_functions(self, data_set = 'Bouwens21', plot = True):

        z_uv, m_uv, l_uv, sigma_l_uv   = p21c_tools.load_uv_luminosity_functions(data_set)
        
        if PY21CMFAST:
            m_uv_sim , _ , log10_l_uv_sim  = p21f.compute_luminosity_function(redshifts = z_uv, 
                                                    user_params  = self._user_params, 
                                                    astro_params = self._astro_params, 
                                                    flag_options = self._flag_options)
        else : 
            ImportError("21cmFAST is needed to compute the UV luminosity functions")

        
        if plot is True:
            fig = p21c_tools.plot_func(m_uv_sim, 10**log10_l_uv_sim, ylog=True, xlim=[-14, -25], ylim =[1e-12, 1], xlabel=r'$M_{\rm UV}$', ylabel=r'$\phi_{\rm UV}$')
            for iz, z in enumerate(z_uv) :
                    fig.gca().errorbar(m_uv[iz], l_uv[iz], yerr=sigma_l_uv[iz], linestyle='', capsize=2, elinewidth=0.5, label=r'${}$'.format(z))

            fig.gca().legend()
            fig.savefig(self._dir_path + '/global_quantities/UV_lum_func_FIDUCIAL.pdf', bbox_inches='tight')

        ## The chi2 is given by a sum on the elements where z > 6 (as we cannot trust 21cmFAST below)

        _chi2 = 0
        _nval = 0
        for iz, z in enumerate(z_uv):
            if z > 6:
                log10_l_uv_func = interpolate.interp1d(m_uv_sim[iz], log10_l_uv_sim[iz])
                for im, m in enumerate(m_uv[iz]) :     
                    _chi2 = _chi2 + (10**log10_l_uv_func(m) - l_uv[iz][im])**2/(sigma_l_uv[iz][im]**2)
                    _nval = _nval + 1

        ## Now compute the reduced chi2
        _reduced_chi2 = _chi2 / _nval

        return _reduced_chi2
    

    def plot_power_spectrum(self):
        super().plot_power_spectrum(std=self._ps_exp_noise, figname = self._dir_path + "/power_spectra/power_spectrum_FIDUCIAL.pdf")


    def plot_xH_box(self):
        fig = p21c_tools.plot_func(self.z_glob, self.xH_box,
                                    xlabel=r'$z$',
                                    ylabel=r'$x_{\rm H_{I}}$')
        fig.savefig(self._dir_path + '/global_quantities/xH_FIDUCIAL.pdf', bbox_inches='tight')


    def plot_global_signal(self):
        fig = p21c_tools.plot_func(self.z_glob, self.global_signal, ylim=[-150, 50],
                                    xlabel=r'$z$',
                                    ylabel=r'$\overline{T_{\rm b}}~\rm [mK]$')
        fig.savefig(self._dir_path + '/global_quantities/global_signal_FIDUCIAL.pdf', bbox_inches='tight')





class Parameter:

    def __init__(self, fiducial, name, plot = True, verbose = True, **kwargs):
        
        self._fiducial       = fiducial
        self._name           = name
        self._plot           = plot

        self._dir_path       = self._fiducial.dir_path
        self._astro_params   = self._fiducial.astro_params
        self._cosmo_params   = self._fiducial.cosmo_params
        self._z_bins         = self._fiducial.z_bins
        self._z_array        = self._fiducial.z_array
        self._k_bins         = self._fiducial.k_bins
        self._logk           = self._fiducial.logk

        self._verbose        = verbose

        # Additional parameters to specify the value of the parameter
        self._values         =  kwargs.get('values', None)
        self._add_name       =  kwargs.get('add_name', '')
        
        if self._add_name != '':
            self._add_name = '_' + self._add_name
            
        __params_plot  = p21c_tools._PARAMS_PLOT.get(self._name, None)
        self._tex_name = __params_plot['tex_name'] if (__params_plot is not None) else p21c_tools._PARAMS_PLOT.get('theta', None)['tex_name']
       
        self._load = kwargs.get('load', True)

        if (name not in self._astro_params) or (name not in self._cosmo_params):
            ValueError("ERROR: the name does not corresponds to any parameter")

        if name in self._astro_params:
            self._param_fid = self._astro_params[self._name]
        
        if name in self._cosmo_params:
            self._param_fid = self._cosmo_params[self._name]

        _loading_success = True

        ############################################################
        # get the lightcones / saved cached values from the filenames
        if self._load is True:
              
            _file_name = self._dir_path + "/cache/Table_Lightcone_rs" + str(self._fiducial.rs) + "_*" + self._name + self._add_name + "_"
            
            
            if self._values is None :
                # if we do not specify values for the parameter
                _lightcone_file_name = glob.glob(_file_name + "[0-9\-]*" + ".h5.npz")
                _lightcone_file_name = [fname[:-4] for fname in _lightcone_file_name] 
            else: 
                # if we specify a value for the parameter
                if not isinstance(self._values, list):
                    self._values = [self._values]
                _lval = [None] * len(self._values)
                
                for ival, val in enumerate(self._values):
                    _lval[ival] = glob.glob(_file_name + '{:.4e}'.format(val) + ".h5.npz")

                if len(_lval[0]) > 0:
                    _lightcone_file_name = [lval[0][:-4] for lval in _lval] 
                else:
                    _lightcone_file_name = []
            
            # test if saved lightcones have been found successfully
            _loading_success = True if len(_lightcone_file_name) > 0 else False

        if self._load is False or _loading_success is False:
            
            _file_name = self._dir_path + "/Lightcone_rs" + str(self._fiducial.rs) + "_*" + self._name + self._add_name + "_"
            if self._values is None :
                _lightcone_file_name = glob.glob(_file_name + "[0-9\-]*" + ".h5")
            else: 
                
                if not isinstance(self._values, list):
                    self._values = [self._values]
                _lval = [None] * len(self._values)
                
                for ival, val in enumerate(self._values):
                    _lval[ival] = glob.glob(_file_name + '{:.4e}'.format(val) + ".h5")
                    assert len(_lval[0]) > 0,  FileNotFoundError(f'No files found for the lightcone associated to ' + self._name + self._add_name + f' variations for value: ' + '{:.4e}'.format(val))
                
                _lightcone_file_name = [lval[0] for lval in _lval]        
             
        assert len(_lightcone_file_name) > 0,  FileNotFoundError(f'No files found for the lightcone associated to ' + str(self._name) + f' variations')
        ############################################################

        # get (from the filenames) the quantity by which the parameter has been varies from the fiducial
        self._p_value = []
        for file_name in _lightcone_file_name:
            # Get the value of p from the thing
            self._p_value.append(float(file_name.split("_")[-1][:-3]))

        # If more than one file with the same p_values we remove all identical numbers
        self._p_value = list(set(self._p_value))

        # Sort the p_values from the smallest to the largest
        self._p_value = np.sort(self._p_value)

        # Check that the p_values are consistant
        _param_fid = self._astro_params[self._name]
        assert len(self._p_value) == 1 or (len(self._p_value) == 2 and ((self._p_value[0] - _param_fid) * (self._p_value[1] - _param_fid)) < 0), "p_value : " + str(self._p_value)

        if verbose is True: 
            print("------------------------------------------")
            print(self._name  + " has been varied with p = " + str(self._p_value))
            print("Loading the lightcones and computing the power spectra")
        else :
            if self._add_name != '':
                print("Treating parameter " + self._name + ' (' + self._add_name[1:] + ')')
            else:
                print("Treating parameter " + self._name)

        # We get the lightcones and then create the corresponding runs objects
        self._runs =  [Run(self._dir_path, 
                                    'Lightcone_rs' + str(self._fiducial.rs) + '_' + self._name + self._add_name + '_{:.4e}'.format(p) + '.h5', 
                                    self._z_bins, self._z_array, self._k_bins, 
                                    self._logk, p, **kwargs) for p in self._p_value]

        if verbose is True: 
            print("Power spectra of " + self._name  + " computed")
      
        ## Check that the k-arrays, z-arrays, k-bins and z-bins correspond 
        for run in self._runs:
            assert compare_arrays(run.z_array, self._fiducial.z_array, 1e-2)
            assert compare_arrays(run.k_array, self._fiducial.k_array, 1e-2)
            assert compare_arrays(run.z_bins,  self._fiducial.z_bins,  1e-2)
            assert compare_arrays(run.k_bins,  self._fiducial.k_bins,  1e-2)

        ## Define unique k and z arrays
        self._k_array = self._fiducial.k_array
        self._z_array = self._fiducial.z_array
        self._k_bins  = self._fiducial.k_bins
        self._z_bins  = self._fiducial.z_bins

        if verbose is True: 
            print("Computing the derivatives of " + self._name)

        self.compute_ps_derivative()

        if verbose is True: 
            print("Derivative of " + self._name  + " computed")
            print("------------------------------------------")
            
        # Plotting the derivatives
        if self._plot is True:
            self.plot_ps_derivative()
            self.plot_weighted_ps_derivative()
            self.plot_power_spectra()

    @property
    def ps_derivative(self):
        return self._ps_derivative

    @property
    def k_array(self):
        return self._k_array

    @property
    def z_array(self):
        return self._z_array

    @property
    def k_bins(self):
        return self._k_bins

    @property
    def z_bins(self):
        return self._z_bins

    @property
    def fiducial(self):
        return self._fiducial

    @property
    def name(self):
        return self._name


    def compute_ps_derivative(self):
        
        _der = [None] * len(self._z_array)
        
        #_param_fid = self._astro_params[self._name]

        # get all the parameters and sort them
        _params = np.zeros(len(self._runs))

        for irun, run in enumerate(self._runs):
            _params[irun] = run.p

        _params         = np.append(_params, self._param_fid)
        _params_sorted  = np.sort(_params)
        _mixing_params  = np.argsort(_params)


        # loop over all the redshift bins
        for iz, z in enumerate(self._z_array) :   

            # get an array of power spectra in the same order as the parameters
            _power_spectra = [run.power_spectrum[iz] for run in self._runs]
            _power_spectra.append(self._fiducial.power_spectrum[iz])

            # rearrange the power spectra in the same order of the parameters
            _power_spectra_sorted = np.array(_power_spectra)[_mixing_params]        

            # evaluate the derivative as a gradient
            _der[iz] = np.gradient(_power_spectra_sorted, _params_sorted, axis=0)

        # arrange the derivative whether they are left, right or centred
        self._ps_derivative  = {'left' : None, 'right' : None, 'centred' : None}

        if len(self._p_value) == 2:
            self._ps_derivative['left']    = [_der[iz][0] for iz, _ in enumerate(self._z_array)]
            self._ps_derivative['centred'] = [_der[iz][1] for iz, _ in enumerate(self._z_array)]
            self._ps_derivative['right']   = [_der[iz][2] for iz, _ in enumerate(self._z_array)]

        if len(self._p_value) == 1 and self._p_value[0] < 0 :
            self._ps_derivative['left'] = [_der[iz][0] for iz, _ in enumerate(self._z_array)]
        
        if len(self._p_value) == 1 and self._p_value[0] > 0 :
            self._ps_derivative['right'] = [_der[iz][0] for iz, _ in enumerate(self._z_array)]

        

    def weighted_ps_derivative(self, kind: str ='centred'):
       
        """
        ## Weighted derivative of the power spectrum with respect to the parameter
        
        Params:
        -------
        kind : str, optional
            choice of derivative ('left', 'right', or 'centred')

        Returns:
        --------
        The value of the derivative devided by the error
        """
        
        # experimental error
        ps_exp_noise   = self._fiducial.ps_exp_noise  

        # theoretical uncertainty from the simulation              
        ps_poisson_noise  = self._fiducial.ps_poisson_noise 
        ps_modeling_noise = self._fiducial.ps_modeling_noise
    
        
        # if fiducial as no standard deviation defined yet, return None
        if ps_exp_noise is None:
            return None

        # Initialize the derivative array to None
        der = None

        # If two sided derivative is asked then we are good here
        if kind == 'centred' :
            der = self._ps_derivative.get('centred', None)
        
        # Value to check if we use the one sided derivative 
        # by choice or because we cannot use the two-sided one
        _force_to_one_side = False

        # If there is no centred derivative
        if der is None or kind == 'left':
            if der is None and kind != 'left' and kind != 'right': 
                # Means that we could not read a value yet 
                # but not that we chose to use the left
                _force_to_one_side = True
            der = self._ps_derivative.get('left', None)
        
        if der is None or kind == 'right':
            if der is None and kind != 'right' and kind != 'left': 
                # Means that we could not read a value yet 
                # but not that we chose to use the right
                _force_to_one_side = True
            der = self._ps_derivative.get('right', None)

        if _force_to_one_side is True and self._verbose is True:
            print("Weighted derivative computed from the one_sided derivative")

        # We sum (quadratically) the two errors
        return der / np.sqrt(ps_exp_noise**2 + ps_poisson_noise**2 + ps_modeling_noise**2)  


    def plot_ps_derivative(self):

        der_array = []
        for key in self._ps_derivative.keys():
            if self._ps_derivative[key] is not None:
                der_array.append( self._ps_derivative[key])
        
        fig = p21c_tools.plot_func_vs_z_and_k(self._z_array, self._k_array, der_array, marker='.', markersize=2, 
                                                title=r'$\frac{\partial \Delta_{21}^2}{\partial ' + self._tex_name + r'}$', 
                                                xlim = [0.1, 1], xlog=self._logk, ylog=False)
        fig.savefig(self._dir_path + "/derivatives/derivatives_" + self._name + self._add_name + ".pdf")
        return fig


    def plot_power_spectra(self, **kwargs):

        _ps        = [self._fiducial.power_spectrum]
        _ps_errors = [self._fiducial.ps_poisson_noise]
        _p_vals    = [self._param_fid]

        for run in self._runs:
            _ps.append(run.power_spectrum)
            _ps_errors.append(run.ps_poisson_noise)
            _p_vals.append(run.p)

        _order     = np.argsort(_p_vals)
        _ps        = np.array(_ps)[_order]
        _ps_errors = np.array(_ps_errors)[_order]

        fig = p21c_tools.plot_func_vs_z_and_k(self.z_array, self.k_array, _ps, func_err = _ps_errors, 
                                                std = self._fiducial.ps_exp_noise, 
                                                title=r'$\Delta_{21}^2 ~ {\rm [mK^2]}$', 
                                                xlog=self._logk, ylog=True, istd = _order[0], **kwargs)

        fig.savefig(self._dir_path + "/power_spectra/power_spectra_" + self._name + self._add_name + ".pdf")
        return fig


    def plot_weighted_ps_derivative(self):

        if self.fiducial.ps_exp_noise is None:
            ValueError("Error: cannot plot the weighted derivatives if the error \
                        if the experimental error is not defined in the fiducial")


        der_array = [self.weighted_ps_derivative(kind=key) for key in self._ps_derivative.keys()]
        fig = p21c_tools.plot_func_vs_z_and_k(self._z_array, self._k_array, der_array, marker='.', markersize=2, 
                                                title=r'$\frac{1}{\sigma}\frac{\partial \Delta_{21}^2}{\partial ' + self._tex_name + r'}$', 
                                                xlim = [0.1, 1], xlog=self._logk, ylog=False)
        fig.savefig(self._dir_path + "/derivatives/weighted_derivatives_" + self._name + self._add_name + ".pdf")
        return fig




def evaluate_fisher_matrix(parameters):
    """
    ## Fisher matrix evaluator

    Params:
    -------
    parameters: array of Parameters objects
        parameters on which to compute the Fisher matrix
    
    Returns:
    --------
    dictionnary with keys: 'matrix' for the matrix itself and 'name' for 
    the parameter names associated to the matrix in the same order
    """
    
    # Get the standard deviation
    n_params = len(parameters)
    fisher_matrix = np.zeros((n_params, n_params))

    name_arr     = [''] * n_params
    weighted_der = [None] * n_params

    for ip, param in enumerate(parameters):
        name_arr[ip]      = param.name
        weighted_der[ip]  = param.weighted_ps_derivative()

    for i in range(0, n_params) :
        for j in range(0, n_params) :        
            fisher_matrix[i][j] = np.nansum(weighted_der[i] * weighted_der[j])
            
    return {'matrix' : fisher_matrix, 'name' : name_arr}
    

    




class CombinedRuns:
    """
    ## Smart collection of the same runs with different random seeds
    """

    def __init__(self, dir_path, name, z_bins = None, k_bins = None, logk=False, p : float = 0, save=True, load=True, verbose = True, **kwargs) -> None:
        
        self._z_bins  = z_bins
        self._k_bins  = k_bins
        self._logk    = logk
        self._p       = p

        # fetch all the lightcone files that correspond to runs the same parameters but different seed
        _lightcone_file_name = glob.glob(self._dir_path + "/Lightcone_*" + self._name + ".h5")
        
        if len(_lightcone_file_name) > 1:
            print("For " + self._name + ": grouping a total of " + str(len(_lightcone_file_name)) + " runs")
        
        # Create the array of runs
        self._runs =  [Run(dir_path, file_name, self._z_bins, self._k_bins, logk, p) for file_name in _lightcone_file_name]

        assert len(self._runs) > 0, "ERROR when searching for lightcones with a given name" 

        ## check that there is no problem and all z- and k- arrays have the same properties
        for irun in range(1, len(self._runs)) :

            # check that all with the same q-value have the same bins 
            assert compare_arrays(self._runs[0].k_array, self._runs[irun].k_array, 1e-5) 
            assert compare_arrays(self._runs[0].z_array, self._runs[irun].z_array, 1e-5) 
            assert compare_arrays(self._runs[0].k_bins,  self._runs[irun].k_bins,  1e-5) 
            assert compare_arrays(self._runs[0].z_bins,  self._runs[irun].z_bins,  1e-5) 
            
            # check that all with the same q-value have the same astro_params
            assert self._runs[0].astro_params  == self._runs[irun].astro_params
            assert self._runs[0].user_params   == self._runs[irun].user_params
            assert self._runs[0].flag_options  == self._runs[irun].flag_options
            assert self._runs[0].cosmo_params  == self._runs[irun].cosmo_params

        self._z_array = self._runs[0].z_array
        self._k_array = self._runs[0].k_array
        self._z_glob  = self._runs[0].z_glob

        self._average_quantities()


    def _average_quantities(self):
        
        ## compute the average values and the spread 
        self._power_spectrum    = np.average([run.power_spectrum for run in self._runs], axis=0)
        self._ps_poisson_noise  = np.average([run.ps_poisson_noise for run in self._runs], axis=0)
        self._ps_modeling_noise = np.std([run.power_spectrum for run in self._runs], axis=0)
        self._global_signal     = np.average([run.global_signal for run in self._runs], axis=0)
        self._xH_box            = np.average([run.xH_box for run in self._runs], axis=0)
        self._astro_params      = self._runs[0].astro_params 
        self._user_params       = self._runs[0].user_params
        self._flag_options      = self._runs[0].flag_options
        self._cosmo_params      = self._runs[0].cosmo_params

    @property
    def power_spectrum(self):
        return self._power_spectrum
    
    @property
    def ps_poisson_noise(self):
        return self._ps_poisson_noise

    @property
    def ps_modeling_noise(self):
        return self._ps_modeling_noise
    
    @property
    def global_signal(self):
        return self._global_signal
    
    @property
    def xH_box(self):
        return self._xH_box
    
    @property
    def z_glob(self):
        return self._z_glob

    @property
    def z_array(self):
        return self._z_array

    @property
    def k_array(self): 
        return self._k_array
    
    @property
    def z_bins(self):
        return self._z_bins

    @property
    def k_bins(self): 
        return self._k_bins
 
    @property
    def logk(self): 
        return self._logk

    @property
    def p(self):
        return self._p

    # lightcone properties
    @property
    def astro_params(self): 
        return self._astro_params
    
    @property
    def user_params(self):
        return self._user_params

    @property
    def flag_options(self):
        return self._flag_options

    @property
    def cosmo_params(self):
        return self._cosmo_params

    