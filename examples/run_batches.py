import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type = str, help="Path to config file")
parser.add_argument("exec_dir", type = str, help="Path to executable file")
parser.add_argument("cache_dir", type = str, help="Path to cached files")
parser.add_argument("-nomp", "--n_omp", type = int, help="number of OMP threads available")
parser.add_argument("-sid", "--slurm_id", type = int, help="id of the slurm array")
parser.add_argument("-imin", "--i_min", type = int, help="first index of the batch")
parser.add_argument("-imax", "--i_max", type = int, help="last index of the batch")
parser.add_argument("-rs", "--random_seed", type = int, help="random seed of the runs")
args = parser.parse_args()

config_dir    = args.config_dir
exec_dir      = args.exec_dir
cache_dir     = args.cache_dir
n_omp         = args.n_omp       if (args.n_omp       is not None) else 1
s_id          = args.slurm_id    if (args.slurm_id    is not None) else 0
imin          = args.i_min       if (args.i_min       is not None) else 0
imax          = args.i_max       if (args.i_max       is not None) else 1      
random_seed   = args.random_seed if (args.random_seed is not None) else None

for i in range(imin, imax):

    print(f"{s_id} : Treating file {i}", flush=True)
    
    try:
        config_file = os.path.join(config_dir, f"Config_{i}.config")
        exec_file = os.path.join(exec_dir, "run_single_lightcone.py")
        cmd = [
            "python", exec_file, config_file, config_dir,
            str(n_omp), str(random_seed), str(i)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout, flush=True)
        if result.returncode != 0:
            print(f"{s_id} : Run {i} failed with return code {result.returncode}", flush=True)
            print(result.stderr, flush=True)

    except Exception as e:
        print(f"{s_id} : Run {i} crashed: \n{e}", flush=True)

        #make some cleanup
        cache_path = os.path.join(cache_dir, f"run_{i}")
        os.rmdir(cache_path)