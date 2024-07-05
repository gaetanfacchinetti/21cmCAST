import argparse
import py21cmcast as p21c
from astropy import units
import os, gc, importlib
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type = str, help="Path to config file")
parser.add_argument("-nomp", "--n_omp", type = int, help="number of OMP threads available")
parser.add_argument("-sid", "--slurm_id", type = int, help="id of the slurm array")
parser.add_argument("-imin", "--i_min", type = int, help="first index of the batch")
parser.add_argument("-imax", "--i_max", type = int, help="last index of the batch")
parser.add_argument("-rs", "--random_seed", type = int, help="random seed of the runs")
args = parser.parse_args()

config_dir    = args.config_dir
n_omp         = args.n_omp       if (args.n_omp       is not None) else 1
s_id          = args.slurm_id    if (args.slurm_id    is not None) else 0
imin          = args.i_min       if (args.i_min       is not None) else 0
imax          = args.i_max       if (args.i_max       is not None) else 1      
random_seed   = args.random_seed if (args.random_seed is not None) else None

z_bins, z_centers, k_bins = p21c.define_grid_modes_redshifts(6., 8 * units.MHz, z_max = 22, k_min = 0.1 / units.Mpc, k_max = 1 / units.Mpc)

print(str(s_id) + " : Treating config files from " + str(imin) + " to " + str(imax))

for i in range(imin, imax):

    try:

        # run the lightcone with the given seed
        lightcone, run_id, output_dir, params = p21c.run_lightcone_from_config(os.path.join(config_dir, "Config_" + str(i) + ".config"), n_omp, random_seed)

        # test if was able to pass the reionization test (if it applies)
        if params.get('Q_max', 1.0) < 0.5:
            p21c.make_directory(os.path.join(output_dir, "cache"), clean_existing_dir=False)        
            with open(os.path.join(output_dir, 'cache/LateReionization_Run_' + run_id + '.pkl'), 'wb') as file:
                pickle.dump(params, file)

        if lightcone is not None :

            # define the p21c object to save
            run = p21c.Fiducial(output_dir, z_bins, z_centers, k_bins, False, lightcone = lightcone, frac_noise = 0.2,
                        load = False, save = True, verbose = False, name = "Run_" + str(i), rs = lightcone.random_seed)
            
            # compute the noise associated
            run.compute_sensitivity()

            # enforce freeing memory
            del run
            del lightcone
            gc.collect()
    
    except Exception as e:
        print(str(s_id) + " : Run " + str(i) + " failed: \n" + str(e))

