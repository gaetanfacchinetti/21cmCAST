import argparse
import py21cmcast as p21c
from astropy import units
import os.path
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("config_file", type = str, help="Path to config file")
parser.add_argument("-nomp", "--n_omp", type = int, help="number of OMP threads available")
parser.add_argument("-nruns", "--n_runs", type = int, help="number of times to run the code < 25")
parser.add_argument("-rs", "--random_seed", type = int, help="the first random seed of the runs")
parser.add_argument("-pp", "--pre_process", help = "pre-process the lightcone and only extract part of the data in the cache", action = 'store_true')
args = parser.parse_args()

config_file   = args.config_file
n_omp         = args.n_omp       if (args.n_omp       is not None) else 1
n_runs        = args.n_runs      if (args.n_runs      is not None) else 1
random_seed   = args.random_seed if (args.random_seed is not None) else None
preprocess    = args.pre_process 

# Run the lightcone with the given seed
lightcone, run_id, output_dir, params = p21c.run_lightcone_from_config(config_file, n_omp, random_seed)

# test if was able to pass the reionization test (if it applies)
if params.get('Q_max', 1.0) < 0.7 or params.get('Q_mid', 1.0) < 0.4:
    p21c.make_directory(os.path.join(output_dir, "cache"), clean_existing_dir=False)        
    with open(os.path.join(output_dir, 'cache/LateReionization_Run_' + run_id + '.pkl'), 'wb') as file:
        pickle.dump(params, file)
    
if lightcone is not None:
    if preprocess is True:
        z_bins, z_centers, k_bins = p21c.define_grid_modes_redshifts(6., 8 * units.MHz, z_max = 22, k_min = 0.1 / units.Mpc, k_max = 1 / units.Mpc)
        p21c.Run(output_dir, "Lightcone_rs" + str(lightcone.random_seed) + "_" + run_id + ".h5", z_bins, z_centers, k_bins, False, lightcone = lightcone, load = False, save = True, verbose = False)
    else:
        lightcone.save(fname = "Lightcone_rs" + str(lightcone.random_seed) + "_" + run_id + ".h5", direc = output_dir)

