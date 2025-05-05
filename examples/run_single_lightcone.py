import py21cmcast as p21c
from astropy import units
import os, gc, sys
import pickle

config_file = sys.argv[1]
output_path = sys.argv[2]
n_omp = int(sys.argv[3])
random_seed = None if sys.argv[4] == "None" else int(sys.argv[4])
i = int(sys.argv[5])


z_bins, z_centers, k_bins = p21c.define_grid_modes_redshifts(6., 8 * units.MHz, z_max = 22, k_min = 0.1 / units.Mpc, k_max = 1 / units.Mpc)

lightcone, run_id, output_dir, params = p21c.run_lightcone_from_config(config_file, n_omp, random_seed)

if params.get('Q_max', 1.0) < 0.7 or params.get('Q_mid', 1.0) < 0.4:
    p21c.make_directory(os.path.join(output_dir, "cache"), clean_existing_dir=False)
    with open(os.path.join(output_dir, f'cache/LateReionization_Run_{run_id}.pkl'), 'wb') as file:
        pickle.dump(params, file)

if lightcone is not None:
    run = p21c.Fiducial(output_dir, z_bins, z_centers, k_bins, False, lightcone=lightcone, frac_noise=0.2,
                        load=False, save=True, verbose=False, name=f"Run_{i}", rs=lightcone.random_seed)
    run.compute_sensitivity()
    del run
    del lightcone
    gc.collect()

