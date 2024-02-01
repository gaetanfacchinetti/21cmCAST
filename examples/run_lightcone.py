import argparse
import py21cmcast as p21c

parser = argparse.ArgumentParser()
parser.add_argument("config_file", type = str, help="Path to config file")
parser.add_argument("-nomp", "--n_omp", type = int, help="number of OMP threads available")
parser.add_argument("-nruns", "--n_runs", type = int, help="number of times to run the code < 25")
parser.add_argument("-rs", "--random_seed", type = int, help="the first random seed of the runs")
args = parser.parse_args()

config_file   = args.config_file
n_omp         = args.n_omp       if (args.n_omp       is not None) else 1
n_runs        = args.n_runs      if (args.n_runs      is not None) else 1
random_seed   = args.random_seed if (args.random_seed is not None) else None

# Run the lightcone with the given seed
lightcone, run_id, output_dir = p21c.run_lightcone_from_config(config_file, n_omp, random_seed)
lightcone.save(fname = "Lightcone_rs" + str(lightcone.random_seed) + "_" + run_id + ".h5", direc = output_dir)