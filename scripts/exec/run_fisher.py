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
random_seed_0 = args.random_seed if (args.random_seed is not None) else None

# Get the parent folder, where to save everything
output_folder = ''
for chain in config_file.split('/')[:-1]:
    output_folder = output_folder + chain + "/"

i : int = 0
j : int = 0

n_runs_max = 25

while j < n_runs and i < n_runs_max:
    
    if random_seed_0 is not None:
        # Choose a definite random seed
        random_seed = random_seed_0 + i 
        print('Random seed set to: ' + str(random_seed))
    else:
        # Let 21cmFAST choose the seed by setting None here
        random_seed = None

    # Run the lightcone with the given seed
    lightcone, run_id = p21c.run_lightcone_from_config(config_file, n_omp, random_seed, heating_rate_output = 'eps_heat.txt')

    if lightcone is not None:

        if random_seed is not None :
            lightcone.save(fname = "Lightcone_rs" + str(random_seed) + "_" + run_id + ".h5", direc = output_folder)
        else:
            lightcone.save(fname = "Lightcone_rs" + str(0) + "_" + run_id + ".h5", direc = output_folder)

        j = j+1

    else:
        print('Run with seed: ' + str(random_seed) + ' has failed.')

    i = i+1

if i == n_runs_max:
    print('WARNING: maximal number of iterations has been reached in the script run_fisher.py')

