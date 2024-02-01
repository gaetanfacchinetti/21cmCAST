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



from matplotlib.pyplot import *

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
import copy
import os
import shutil

import copy



def make_directory(path: str, clean_existing_dir:bool = False):
    
    if not os.path.exists(path): 
        os.mkdir(path)
    else:
        if clean_existing_dir is True:
            clean_directory(path)
        else:
            return True

    return False



def clean_directory(path: str):
    """ Clean the directory at the path: path """

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def read_config_params(config_items, int_type = True):
    """
    Read ints and booleans from config files
    Use for user_params and flag_options only
    
    Parameters
    ----------
    item : str
        config dictionary item as a string
    Return
    ------
    config dictionary item as an int, bool or str
    """

    output_dict = dict()

    for key, value in dict(config_items).items():

        try:
            if int_type is True:
                cast_val = int(value)
            else:
                cast_val = float(value)
        except:
            if value == 'True':
                cast_val =  True
            elif value == 'False':
                cast_val =  False
            else:
                cast_val = value
    
        output_dict[key] = cast_val
        
    return output_dict



def write_config_params(filename, name, output_dir, cache_dir, 
                        lightcone_quantities, global_quantities, 
                        extra_params, user_params, flag_options, 
                        astro_params, cosmo_params, key):

    with open(filename, 'w') as f:
       
        print("# Parameter file for : " + key, file = f)
        print('', file=f)

        print("[run]", file=f)
        print("name      : " + name, file=f)
        print("run_id    : " + key, file=f)
        print("output_dir : " + output_dir, file=f)
        print("cache_dir : " + cache_dir, file=f)
        print('', file=f)
        
        print("[extra_params]", file=f)
        
        for key, value in extra_params.items():
            print(key + " : " + str(value), file=f)

        
        print('', file=f)
        print("[lightcone_quantities]", file=f)

        for key, value in lightcone_quantities.items():
            print(key + " : " + str(value), file=f)

        print('', file = f)
        print("[global_quantities]", file=f)

        for key, value in global_quantities.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[user_params]", file=f)


        for key, value in user_params.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[flag_options]", file=f)

        for key, value in flag_options.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[astro_params]", file=f)

        for key, value in astro_params.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[cosmo_params]", file=f)

        for key, value in cosmo_params.items():
            print(key + " : " + str(value), file=f)



def read_power_spectra(folder_name: str):
    """ 
    Read the power spectra from a folder 
    The folder must contain a redshift array file: folder_name/redshift_chucnks.txt
    The power spectra must be labelled and organised as folder_name/ps_z_<<"{0:.1f}".format(z)>>.txt
    The units must be Mpc for k_arr, mK**2 for delta_arr and err_arr

    Parameters
    ----------
        folder_name: str
            path to the folder where the power_spectra_are_stored
    
    Returns
    -------
        z_arr: list[float]
            list of redshifts where the power_spectra are evaluated
        k_arr: list[list[float]] (Mpc^{-1})
            list of k values for every redshift
        delta_arr: list[list[float]] (mK^2)
            list of power_spectrum value for every redshift (in correspondance to k_arr)
        err_arr: list[list[float]]  (mK^2)
            list of the error on the power spectrum (in correspondance to k_arr and delta_arr)
    """

    z_arr     = np.genfromtxt(folder_name + '/power_spectra_vs_k/redshift_chunks.txt')

    k_arr     = []
    delta_arr = []
    err_arr   = []

    for iz, z in enumerate(z_arr):
        data = np.genfromtxt(folder_name + '/power_spectra_vs_k/ps_z_' + "{0:.1f}".format(z) + '.txt')

        k_arr.append(data[:, 0])
        delta_arr.append(data[:, 1])
        err_arr.append(data[:, 2])

    return z_arr, k_arr, delta_arr, err_arr






def confidence_ellipse(cov, mean_x, mean_y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def ellipse_from_covariance(cov_matrix, fiducial):
    """ Returns arrays for drawing the covariance matrix
        This function is mainly used as a cross-check of confidence_ellipse()"""
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    theta = np.linspace(0, 2*np.pi, 200)
    ellipse_x = (np.sqrt(eigenvalues[0])*np.cos(theta)*eigenvectors[0,0] + np.sqrt(eigenvalues[1])*np.sin(theta)*eigenvectors[0,1]) + fiducial[0]
    ellipse_y = (np.sqrt(eigenvalues[0])*np.cos(theta)*eigenvectors[1,0] + np.sqrt(eigenvalues[1])*np.sin(theta)*eigenvectors[1,1]) + fiducial[1]

    return ellipse_x, ellipse_y



_PARAMS_PLOT = {
    'theta'             : {'tex_name' : r'\theta'}, # default parameters
    'F_STAR10'          : {'tex_name' : r'\log_{10}[f_{\star, 10}]'},
    'F_STAR7_MINI'      : {'tex_name' : r'\log_{10}[f_{\star, 7}^{\rm mini}]'},
    'ALPHA_STAR'        : {'tex_name' : r'\alpha_{\star}', 'min': -0.7, 'max' : 1.7, 'ticks' : []},
    'ALPHA_STAR_MINI'   : {'tex_name' : r'\alpha_{\star}^{\rm mini}', 'min': -0.7, 'max' : 1.7, 'ticks' : []},
    't_STAR'            : {'tex_name' : r't_{\star}', 'min': -0.3, 'max' : 1.3, 'ticks' : []},
    'F_ESC10'           : {'tex_name' : r'\log_{10}[f_{\rm esc, 10}]', 'min': -2, 'max' : 0.1},
    'F_ESC7_MINI'       : {'tex_name' : r'\log_{10}[f_{\rm esc, 7}^{\rm mini}]', 'min': -2, 'max' : 0.1},
    'ALPHA_ESC'         : {'tex_name' : r'\alpha_{\rm esc}', 'min': -1.4, 'max': 0.6, 'ticks' : []},
    'L_X'               : {'tex_name' : r'\log_{10}\left[\frac{L_X}{\rm units}\right]', 'min': 40, 'max': 41},
    'L_X_MINI'          : {'tex_name' : r'\log_{10}\left[\frac{L_X^{\rm mini}}{\rm units}\right]', 'min': 40, 'max': 41},
    'M_TURN'            : {'tex_name' : r'\log_{10}\left[\frac{M_{\rm turn}}{{\rm M}_\odot}\right]', 'min': 7.6, 'max': 8.9, 'ticks' : []},
    'NU_X_THRESH'       : {'tex_name' : r'E_0~[\rm eV]', 'min': 300, 'max': 700, 'ticks' : []},
    'DM_LOG10_LIFETIME' : {'tex_name' : r'\log_{\rm 10}\left[\frac{\tau_{\chi}}{\rm s}\right]', 'min': 25.6, 'max': 26.4, 'ticks' : [], 'positive' : False},
    'DM_DECAY_RATE'     : {'tex_name' : r'\Gamma ~ [\rm s^{-1}]', 'positive' : True, 'val' : 0},
    'DM_LOG10_MASS'     : {'tex_name' : r'\log_{10}[\frac{m_{\chi}}{\rm eV}]', 'min': 3, 'max': 11,  'ticks' : []},
    'DM_FHEAT_APPROX_PARAM_LOG10_F0' : {'tex_name' : r'\log_{10}[f_0]', 'min': -2, 'max': 1,  'ticks' : []},
    'DM_FHEAT_APPROX_PARAM_A'  : {'tex_name' : r'a', 'min': -2, 'max': 1,  'ticks' : []},
    'DM_FHEAT_APPROX_PARAM_B'  : {'tex_name' : r'b', 'min': -2, 'max': 1,  'ticks' : []},
    'LOG10_XION_at_Z_HEAT_MAX' : {'tex_name' : r'\log_{10}[\bar x_e^{\rm init}]', 'min': -2, 'max': 1,  'ticks' : []},
    'LOG10_TK_at_Z_HEAT_MAX'   : {'tex_name' : r'\log_{10}[\frac{\bar T_K^{\rm init}}{\rm K}]', 'min': -2, 'max': 1,  'ticks' : []},
    }


def make_triangle_plot(covariance, fiducial_params, **kwargs) : 

    #####################################
    ## Choose the data we want to look at
    
    if not isinstance(fiducial_params, list):
        fiducial_params[fiducial_params]

    if not isinstance(covariance, list):
        all_name_params = list(covariance.keys())
        covariance  = [covariance]  
    else:
        # Define all name_params
        all_name_params = []
        for name in [list(cov.keys()) for cov in covariance]:
            all_name_params += name
        all_name_params = list(set(all_name_params))  

    params_to_plot = kwargs.get('params_to_plot', all_name_params) 

    # add the fiducial values in the infos
    val_min = {}
    val_max = {}
    
    _default_info  = {'tex_name' : r'\theta', 'min' : None, 'max' : None, 'ticks' : [], 'positive' : False, 'val' : None}

    color = plt.rcParams['axes.prop_cycle'].by_key()['color']     
    color = kwargs.get('color', color)
    alpha = kwargs.get('alpha', [0.8]*len(covariance))

    if not isinstance(color, list):
        color = [color]

    for i_cov, cov in enumerate(covariance):
        for i_name in covariance[i_cov].keys():

            val = fiducial_params[i_cov][i_name]                   

            _param_info  = _PARAMS_PLOT.get(i_name, _default_info)
            if _param_info.get('val', None) is not None:
                val = _param_info.get('val', None)

            _sigma = np.sqrt(cov[i_name][i_name])
    
            val_min[i_name] = np.min([val - 3.5*_sigma, val_min.get(i_name, val)])
            val_max[i_name] = np.max([val + 3.5*_sigma, val_max.get(i_name, val)])

            if _param_info.get('positive', False) is True:
                val_min[i_name] = np.max([val_min[i_name], 0])
            
    
    fig, axs = prepare_triangle_plot(params_to_plot, val_min, val_max)
    for i_cov, cov in enumerate(covariance):
        fill_triangle_plot(cov, fiducial_params[i_cov], axs, color=color[i_cov], alpha = alpha[i_cov])

    axs[params_to_plot[0]][params_to_plot[0]].set_yticks([]) 
 
    return fig


def prepare_triangle_plot(params_to_plot, val_min, val_max):
    
    ngrid = len(params_to_plot)
    
    #####################################
    ##  Prepare the triangle plot
    fig = plt.figure(constrained_layout=False, figsize=(1.2*ngrid, 1.2*ngrid))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    gs = GridSpec(ngrid, ngrid, figure=fig)
    axs = {param : {} for param in params_to_plot}
    
    for j, j_name in enumerate(params_to_plot) : 
        for i, i_name in enumerate(params_to_plot): 
            
            if i < j+1:

                ## Here i represents the x axis while j goes along the y axis
                axs[j_name][i_name] = fig.add_subplot(gs[j:j+1, i:i+1])

                ## Information concertning this case
                _default_info  = {'tex_name' : r'\theta', 'min' : None, 'max' : None, 'ticks' : [], 'positive' : False, 'val' : None}
                
                _param_info_x  = _PARAMS_PLOT.get(i_name, _default_info)
                _param_info_y  = _PARAMS_PLOT.get(j_name, _default_info)

                x_min = val_min[i_name]
                x_max = val_max[i_name]
                y_min = val_min[j_name]
                y_max = val_max[j_name]

                axs[j_name][i_name].set_xlim([x_min, x_max])

                if i != j :     
                    axs[j_name][i_name].set_ylim([y_min, y_max])

                if i == j :
                    axs[i_name][i_name].set_ylim([0, 1.2])


                ##############
                # Setting the plot

                # we remove the ticks if necessary for some parts of this case
                if j < ngrid -1 :
                    axs[j_name][i_name].xaxis.set_ticklabels([])
                if i > 0 : 
                    axs[j_name][i_name].yaxis.set_ticklabels([])

                if j == ngrid -1 :
                    axs[j_name][i_name].set_xlabel(r'${}$'.format(_param_info_x.get('tex_name', r'$\theta$')))
                    x_ticks = _param_info_x.get('ticks', [])
                    if x_ticks != [] :
                        axs[j_name][i_name].set_xticks(_param_info_x.get('ticks', []))
                    axs[j_name][i_name].tick_params(axis='x', labelsize=8)
                    for tick in axs[j_name][i_name].get_xticklabels():
                        tick.set_rotation(55)

                if i == 0 and j > 0:
                    axs[j_name][i_name].set_ylabel(r'${}$'.format(_param_info_y.get('tex_name', r'$\theta$')))
                    y_ticks = _param_info_y.get('ticks', [])
                    if y_ticks != [] :
                        axs[j_name][i_name].set_yticks(_param_info_y.get('ticks', []))
                    axs[j_name][i_name].tick_params(axis='y', labelsize=8)
                    for tick in axs[j_name][i_name].get_yticklabels():
                        tick.set_rotation(55)
           
    return fig, axs


def fill_triangle_plot(covariance, fiducial_params, axs, **kwargs):
    
    color = kwargs.get('color', None)
    alpha = kwargs.get('alpha', 0.8)

    params_to_plot = list(axs.keys())
    params_cov     = list(covariance.keys())

    params = []
    for param_p in params_to_plot:
        if param_p in params_cov:
            params.append(param_p)


    for j, j_name in enumerate(params) : 
        for i, i_name in enumerate(params): 

            if i < j+1:

                ##############
                val_x = fiducial_params[i_name]  
                val_y = fiducial_params[j_name]  

                ## Make the plots now
                if i != j : 

                    # Countour plot for the scatter
                    sub_cov = np.zeros((2, 2))
                    sub_cov[0, 0] = covariance[i_name][i_name]
                    sub_cov[0, 1] = covariance[i_name][j_name]
                    sub_cov[1, 0] = covariance[j_name][i_name]
                    sub_cov[1, 1] = covariance[j_name][j_name]

                    #ellipse_x, ellipse_y = ellipse_from_covariance(sub_cov, [val_x, val_y])
                    #axs[j_name][i_name].plot(ellipse_x, ellipse_y, linewidth=0.5, color='blue')

                    confidence_ellipse(sub_cov, val_x, val_y, axs[j_name][i_name],  n_std=2, facecolor=color, alpha=alpha/2.)
                    confidence_ellipse(sub_cov, val_x, val_y, axs[j_name][i_name],  n_std=1, facecolor=color, alpha=alpha)

                if i == j :

                    axs[i_name][i_name].set_ylim([0, 1.2])
                    axs[i_name][i_name].set_title(r'${}$'.format(val_x) + f'\n' + r'$\pm{:.3}$'.format(np.sqrt(covariance[i_name][i_name])), fontsize=10)
        
                    # Plot the gaussian approximation in that panel
                    sigma     = np.sqrt(covariance[i_name][i_name])
                    val_arr   = np.linspace(val_x-5*sigma, val_x+5*sigma, 100)
                    gaussian  = exp(-(val_arr - val_x)**2/2./sigma**2)

                    axs[i_name][i_name].plot(val_arr, gaussian, color=color)




def plot_func_vs_z_and_k(z, k, func, func_err = None, std = None, istd  : float = 0, **kwargs) :

    """ 
        Function that plots the power spectra with the sensitivity bounds from extract_noise_from_fiducial()
        We can plot on top more power spectra for comparison

        Params
        ------
        k : 1D array of floats
            modes 
        z : 1D array of floats
            redshifts
        func : (list of) 2D arrays of floats
            function(s) to plot in terms of the redshift and modes
        std : 1D array of floats
            standard deviation associated to func (or func[istd])
        istd : float, optional
            index of the func array where to attach the standard deviation
    """
    n_lines = int(len(z) / 5)

    if len(z) / 5 > n_lines:
        n_lines = n_lines + 1
    
    fig = plt.figure(constrained_layout=False, figsize=(10, n_lines*2))
    #fig.subplots_adjust()


    gs = GridSpec(n_lines, 5, figure=fig, wspace=0.5, hspace = 0.5)
    axs = [[None for j in range(0, 5)] for i in range(0, n_lines)]

    if not isinstance(func[0][0], (list, np.ndarray)) :
        func = [func]

    if func_err is None:
        func_err = [None] * len(func)
    else:
        if not isinstance(func_err[0][0], (list, np.ndarray)) : 
            func_err = [func_err]


    if len(func) > 1:
        
        cmap = matplotlib.cm.get_cmap('Spectral')
        a_lin = (0.99-0.2)/(len(func)-1) if len(func) > 1 else 1
        b_lin = 0.2 if len(func) > 1 else 0.5

        _default_color_list     = [cmap(i) for i in np.arange(0, len(k))*a_lin + b_lin]
        _default_linestyle_list = ['-' for i in np.arange(0, len(k))]
    
    else:

        _default_color_list     = ['b']
        _default_linestyle_list = ['-']


    
    color_list     = kwargs.get('color', _default_color_list)
    linestyle_list = kwargs.get('linestyle', _default_linestyle_list)
    ylabel         = kwargs.get('ylabel', None)
    title          = kwargs.get('title', None)
    marker         = kwargs.get('marker', None)
    markersize     = kwargs.get('markersize', None)
    
    no_line = Line2D([],[],color='k',linestyle='-',linewidth=0, alpha = 0) 
  

    iz = 0
    for i in range(0, n_lines):
        for j in range(0, 5):
            
            if iz < len(z): 

                axs[i][j] = fig.add_subplot(gs[i:i+1, j:j+1])

                # Plot the power spectrum at every redshift
                for jf, f in enumerate(func) : 
                    if marker is None : 
                        # By default we plot steps
                        axs[i][j].step(k, f[iz], where='mid', alpha = 1, color=color_list[jf], 
                                        linestyle = linestyle_list[jf], linewidth=0.5)
                    else: 
                        # Otherwise we use markers
                        axs[i][j].plot(k, f[iz], alpha = 1, color=color_list[jf], linestyle = linestyle_list[jf], 
                                        marker=marker, markersize=markersize, markerfacecolor=color_list[jf], 
                                        linewidth=0.8)
                    
                    if func_err[jf] is not None:
                        axs[i][j].fill_between(k, f[iz] - func_err[jf][iz], f[iz] + func_err[jf][iz], 
                                                color=color_list[jf], linestyle=linestyle_list[jf], 
                                                step='mid', alpha=0.1)
                

                # Plot the standard deviation bars if standard deviation is given
                if std is not None : 
                    axs[i][j].fill_between(k, func[istd][iz] - std[iz], func[istd][iz] + std[iz], step='mid', alpha = 0.2, color=color_list[istd])
                
                
                xlim = kwargs.get('xlim', None)
                ylim = kwargs.get('ylim', None)

                if xlim is not None: 
                    axs[i][j].set_xlim(xlim)
                if ylim is not None: 
                    axs[i][j].set_ylim(ylim)

                xlog = kwargs.get('xlog', False)
                ylog = kwargs.get('ylog', False)

                if xlog is True:
                    axs[i][j].set_xscale('log')
                if ylog is True:
                    axs[i][j].set_yscale('log')
                

                axs[i][j].legend([no_line], [r'$\rm z = {0:.1f}$'.format(z[iz])], loc='lower right', handlelength=0, handletextpad=0, 
                                    bbox_to_anchor=(0.98, 1), fontsize=8, frameon=False,
                                    borderpad = 0, borderaxespad = 0, framealpha=0)

            iz = iz+1

    if ylabel is not None:
        axs[int(n_lines/2)-1][0].set_ylabel('{}'.format(ylabel))

    axs[n_lines-1][0].set_xlabel(r'$k ~{\rm [Mpc^{-1}]}$')


    if title is not None: 
        fig.suptitle('{}'.format(title), fontsize=14)


    return fig


def prepare_plot(**kwargs) :

    """ 
        Function that plots simple functions of one variable on the same graph
    """
    
    fig = plt.figure(figsize = (5,4))
    ax = fig.gca()

    xlim   = kwargs.get('xlim', None)
    ylim   = kwargs.get('ylim', None)
    xlog   = kwargs.get('xlog', False)
    ylog   = kwargs.get('ylog', False)
    xlabel = kwargs.get('xlabel', r'$x$')
    ylabel = kwargs.get('ylabel', r'$y$')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('{}'.format(xlabel))
    ax.set_ylabel('{}'.format(ylabel))

    if xlog is True:
        ax.set_xscale('log')
    if ylog is True:
        ax.set_yscale('log')

    return fig, ax



def prepare_2subplots(gridspec_kw = None, **kwargs) :

    """ 
        Function that plots simple functions of one variable on the same graph
    """
    
   

    xlim     = kwargs.get('xlim', None)
    ylim_1   = kwargs.get('ylim_1', None)
    ylim_2   = kwargs.get('ylim_2', None)
    xlog     = kwargs.get('xlog', False)
    ylog_1   = kwargs.get('ylog_1', False)
    ylog_2   = kwargs.get('ylog_2', False)
    xlabel   = kwargs.get('xlabel', r'$x$')
    ylabel_1 = kwargs.get('ylabel_1', r'$y$')
    ylabel_2 = kwargs.get('ylabel_2', r'$y$')
    figsize  = kwargs.get('figsize', (5, 4))

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw=gridspec_kw, sharex=True, figsize=figsize, facecolor="White")

    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim_1 is not None:
        ax1.set_ylim(ylim_1)
    if ylim_2 is not None:
        ax2.set_ylim(ylim_2)

    ax2.set_xlabel('{}'.format(xlabel))
    ax1.set_ylabel('{}'.format(ylabel_1))
    ax2.set_ylabel('{}'.format(ylabel_2))

    if xlog is True:
        ax1.set_xscale('log')
    if ylog_1 is True:
        ax1.set_yscale('log')
    if ylog_2 is True:
        ax2.set_yscale('log')

    return fig, (ax1, ax2)




def plot_func(x, func, **kwargs) :

    """ 
        Function that plots simple functions of one variable on the same graph

        Params
        ------
        x : 1D array of floats
            redshifts
        func : (list of) 1D arrays of floats
            function(s) to plot in terms of x
    """

        
    fig, ax = prepare_plot(**kwargs)
    rax     = kwargs.get('rax', False)
    color   = kwargs.get('color', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    
    if isinstance(func[0], (int, float, np.float64, np.float32)) : 
        func = [func]

    if isinstance(x[0], (int, float, np.float64, np.float32)) : 
        x = [x]*len(func)

    for ifunc, f in enumerate(func):
        ax.plot(x[ifunc], f, color=color[ifunc])

    if rax is False:
        return fig
    else:
        return fig, ax



def display_matrix(matrix, names = None):

    if names is not None:
        print('            ', end='')
        for name in names:
            print(name.ljust(10)[:10] + ' | ', end = '')

    print('')
    
    for i in range(0, len(matrix)):
        if names is not None:
            print(names[i].ljust(10)[:10] + ' | ', end='')
        for j in range(0, len(matrix)):
            add = ''
            if matrix[i][j] > 0:
                add = ' '
            print(add + "{:.1e}".format(matrix[i][j]) + '  |  ', end='')
        print('')




def load_uv_luminosity_functions(data_set = 'Bouwens21'):

    file_name = os.path.dirname(os.path.abspath(__file__)) + '/_data/' + data_set + '.npz'

    with open(file_name, 'rb') as file:  
        data = np.load(file, allow_pickle = True)   
       
        z_uv       = data['redshifts']
        m_uv       = data['M_uv']
        l_uv       = data['l_uv']
        sigma_l_uv = data['sigma_l_uv']


    return z_uv, m_uv, l_uv, sigma_l_uv

