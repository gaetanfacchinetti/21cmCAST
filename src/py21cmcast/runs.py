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

import configparser
import os

from py21cmcast import tools as p21c_tools
import py21cmfast   as p21f
import warnings

import numpy as np


def init_runs(config_file: str, q_scale: float = 3., clean_existing_dir: bool = False, verbose = False) -> None :

    """ 
    Initialise the runs for a fisher analysis according to 
    a fiducial model defined in config_file
    
    Params : 
    --------
    config_file : str
        Path to the config file representing the fiducial
    q_scale : float, optional
        Gives the points where to compute the derivative in pourcentage of the fiducial parameters
    erase_dir : bool, optional
        If True forces the creation of a folder 
    """

    config = configparser.ConfigParser(delimiters=':')
    config.optionxform = str

    config.read(config_file)

    name            = config.get('run', 'name')
    output_dir      = config.get('run', 'output_dir')
    cache_dir       = config.get('run', 'cache_dir')

    extra_params = {}

    try:
        lightcone_q       = p21c_tools.read_config_params(config.items('lightcone_quantities'))
    except configparser.NoSectionError:
        if verbose: 
            warnings.warn("No section lightcone_quantities provided")
        lightcone_q = {}

    try:
        global_q       = p21c_tools.read_config_params(config.items('global_quantities'))
    except configparser.NoSectionError:
        if verbose: 
            warnings.warn("No section global_quantities provided")
        global_q = {}    

    
    extra_params      = p21c_tools.read_config_params(config.items('extra_params'), int_type = False)
    user_params       = p21c_tools.read_config_params(config.items('user_params'))
    flag_options      = p21c_tools.read_config_params(config.items('flag_options'))
    astro_params_fid  = p21c_tools.read_config_params(config.items('astro_params'), int_type = False)
    cosmo_params_fid  = p21c_tools.read_config_params(config.items('cosmo_params'), int_type = False)

    try: 
        astro_params_vary = p21c_tools.read_config_params(config.items('astro_params_vary'), int_type = False)
    except configparser.NoSectionError:
        if verbose: 
            warnings.warn("No section astro_params_vary provided")
        astro_params_vary = {}

    try: 
        cosmo_params_vary = p21c_tools.read_config_params(config.items('cosmo_params_vary'), int_type = False)
    except configparser.NoSectionError:
        if verbose: 
            warnings.warn("No section cosmo_params_vary provided")
        astro_params_vary = {}

    try:
        one_side_der      = p21c_tools.read_config_params(config.items('one_side_param_derivative'))
    except configparser.NoSectionError:
        if verbose: 
            warnings.warn("No section one_side_param_derivative provided")
        one_side_der = {}

    try: 
        percentage        = p21c_tools.read_config_params(config.items('percentage'))
    except configparser.NoSectionError:
        if verbose: 
            warnings.warn("No section percentage provided")
        percentage = {}


    # Make the directory to store the outputs and everything
    output_run_dir = output_dir + "/" + name.upper() + "/"
    existing_dir = p21c_tools.make_directory(output_run_dir, clean_existing_dir = clean_existing_dir)

    if existing_dir is True:
        print('WARNING: Cannot create a clean new folder because clean_existing_dir is False')
        return 

    astro_params_run_all = {}
    astro_params_run_all['FIDUCIAL'] = astro_params_fid

    cosmo_params_run_all = {}
    cosmo_params_run_all['FIDUCIAL'] = cosmo_params_fid

    params_vary  = {}
    class_params = {}
    
    for param, value in astro_params_vary.items():
        params_vary[param] = value
        class_params[param] = "ASTRO"
    for param, value in cosmo_params_vary.items():
        params_vary[param] = value
        class_params[param] = "COSMO"

    for param, value in params_vary.items(): 
        
        _side_der = one_side_der.get(param, None)
        _percentage = percentage.get(param, True)
        
        if _side_der is None :
            vary_array = np.array([-100., 100.])
        else:
            if _side_der == '+': 
                vary_array = np.array([+100.])
            elif _side_der == '-':
                vary_array = np.array([-100.])
            else: 
                ValueError("The arguments of one_side_param_derivative have to be '+' or '-'")

        if class_params[param] == "ASTRO":
            p_fid = astro_params_fid[param] 
        elif class_params[param] == "COSMO":
            p_fid = cosmo_params_fid[param]
        else:
            ValueError("Parameter varied is neither astro or cosmo.")
       
        if isinstance(value, float) and value > 0:
            q = value/vary_array
        else : 
            q = q_scale/vary_array

        if _percentage is False:
            p = 100.*q
        else:
            p = p_fid*(1+q)
            print('Parameter ' + str(param) + ' varied by ' + str(value) + ' percent of the fiducial')

        astro_params_run = astro_params_fid.copy()
        cosmo_params_run = cosmo_params_fid.copy()

        for i, pp in enumerate(p):
            if class_params[param] == "ASTRO":
                astro_params_run[param] = pp
                astro_params_run_all[param + '_{:.4e}'.format(pp)] = astro_params_run.copy()
            elif class_params[param] == "COSMO":
                cosmo_params_run[param] = pp
                cosmo_params_run_all[param + '_{:.4e}'.format(pp)] = cosmo_params_run.copy()
                
    # Write down the separate config files
    irun = 0
    for key, astro_params in astro_params_run_all.items() : 
        p21c_tools.write_config_params(output_run_dir + '/Config_' + key + ".config", name, output_run_dir, cache_dir, 
                                       lightcone_q, global_q, extra_params, user_params, flag_options, astro_params, cosmo_params_fid, key)
        irun = irun + 1
        
    for key, cosmo_params in cosmo_params_run_all.items() : 
        p21c_tools.write_config_params(output_run_dir + '/Config_' + key + ".config", name, output_run_dir, cache_dir, 
                                       lightcone_q, global_q, extra_params, user_params, flag_options, astro_params_fid, cosmo_params, key)
        irun = irun + 1
    
    return 



def make_config_one_varying_param(config_file: str, param_name: str, values, **kwargs):

    config = configparser.ConfigParser(delimiters=':')
    config.optionxform = str

    config.read(config_file)

    name            = config.get('run', 'name')
    output_dir      = config.get('run', 'output_dir')
    cache_dir       = config.get('run', 'cache_dir')

    extra_params = {}
    
    try: 
        extra_params['min_redshift']  = float(config.get('extra_params','min_redshift'))
    except configparser.NoOptionError: 
        print("Warning: min_redshift set to 5 by default")
        extra_params['min_redshift']  = 5

    try:
        extra_params['max_redshift']    = float(config.get('extra_params','max_redshift'))
    except configparser.NoOptionError:
        print("Warning: max_redshift set to 35 by default")
        extra_params['max_redshift']  = 35   

    try:
        extra_params['coarsen_factor']  = int(config.get('extra_params', 'coarsen_factor'))
    except configparser.NoOptionError:
        print("Warning: coarsen factor undifined")

    user_params       = p21c_tools.read_config_params(config.items('user_params'))
    flag_options      = p21c_tools.read_config_params(config.items('flag_options'))
    astro_params_fid  = p21c_tools.read_config_params(config.items('astro_params'), int_type = False)
    
    # Make the directory to store the outputs and everything
    output_run_dir = output_dir + "/" + name.upper() + "/"

    mod_astro_params = kwargs.get('mod_astro_params', None)
    mod_flag_options = kwargs.get('mod_flag_options', None)
    add_file_name    = kwargs.get('add_file_name', '')

    if add_file_name != '':
        add_file_name = add_file_name + '_' 

    if mod_astro_params is not None:
        for param in mod_astro_params.keys():
            if not param in astro_params_fid:
                raise KeyError('This parameter is not primarly defined in the config file')
            astro_params_fid[param] = mod_astro_params[param]

    if mod_flag_options is not None:
        for flag in mod_flag_options.keys():
            if not flag in flag_options:
                raise KeyError('This flag_option is not primarly defined in the config file')
            flag_options[flag] = mod_flag_options[flag]

    astro_params = astro_params_fid

    if not param_name in astro_params_fid:
        raise KeyError('This parameter is not primarly defined in the config file')
    
    for ival, val in enumerate(values):
        astro_params[param_name] = val
        filename = output_run_dir + '/Config_' + param_name + '_' + add_file_name + '{:.4e}'.format(val) + ".config"
        key      = param_name + '_' + add_file_name + '{:.4e}'.format(val)
        #print(filename, astro_params, flag_options, key)
        p21c_tools.write_config_params(filename, name, cache_dir, extra_params, user_params, flag_options, astro_params, key)





def run_lightcone_from_config(config_file: str, n_omp: int = None, random_seed: int = None, **kwargs) :

    """ 
    ## Run a lightcone from a config file 

    Parameters
    ----------
    config_file: str
        path of the configuration file
    n_omp: int
        number of threads to use
    run_id : str, optional
        id of the run we are currently looking at
    random_seed : int
        random_seed with which to run 21cmFAST

    Returns
    ----------
    lightcone: Lightcone object (see 21cmFAST)
        lightcone of the run
    run_id: string
        identifier of the run
    """

    ####################### Getting the data ############################

    config = configparser.ConfigParser(delimiters=':')
    config.optionxform = str
    config.read(config_file)

    name            = config.get('run', 'name')
    run_id          = config.get('run', 'run_id', fallback='') 
    output_dir      = config.get('run', 'output_dir')
    cache_dir       = config.get('run', 'cache_dir')
    

    ## Need to implement this next time
    lightcone_q     = p21c_tools.read_config_params(config.items('lightcone_quantities'))
    global_q        = p21c_tools.read_config_params(config.items('global_quantities'))

    extra_params    = p21c_tools.read_config_params(config.items('extra_params'), int_type=False)
    user_params     = p21c_tools.read_config_params(config.items('user_params'))
    flag_options    = p21c_tools.read_config_params(config.items('flag_options'))
    astro_params    = p21c_tools.read_config_params(config.items('astro_params'), int_type=False)


    # manually set the number of threads
    if n_omp is not None: 
        user_params['N_THREADS'] = int(n_omp)

    seed_str : str = ''
    if random_seed is not None:
        seed_str = 'rs' + str(random_seed) + '_'

    cache_path = cache_dir + name.upper() + '/run_' + seed_str + run_id + '/'
    
    ####################### Running the lightcone ############################

    lightcone_quantities = ()
    global_quantities = ()

    for key, value in lightcone_q.items():
        if value is True:
            lightcone_quantities += (key,)
    
    for key, value in global_q.items():
        if value is True:
            global_quantities += (key,)

    ## Set default values to output if nothing is set in the config file
    if len(lightcone_quantities) == 0:
        lightcone_quantities = ('xH_box',)

    if len(global_quantities) == 0:
        global_quantities = ('brightness_temp', 'xH_box',)

    try: 

        lightcone = p21f.run_lightcone(
                user_params          = user_params,
                astro_params         = astro_params,
                flag_options         = flag_options,
                lightcone_quantities = lightcone_quantities,
                global_quantities    = global_quantities,
                direc                = cache_path, 
                random_seed          = random_seed,
                **extra_params,
                **kwargs,
            )
        
    except Exception as e :

        print(e)

        lightcone = None
        run_id    = None

    try: 
        # at the end, we clear the cache 
        p21f.cache_tools.clear_cache(direc=cache_path)
        # delete the directory once it has been emptied
        os.rmdir(cache_path) 
    except FileNotFoundError:
        pass

    return lightcone, run_id, output_dir


    
 