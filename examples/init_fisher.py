import py21cmcast as p21c
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("config_file", type = str, help="Path to config file")
parser.add_argument("-q", "--q", type = float, help="default percentage of variation")
parser.add_argument("-f", "--force_overwrite", help="overwrite existing config directory", action = 'store_true')
parser.add_argument("-v", "--verbose", help="the first random seed of the runs", action = 'store_true')
parser.add_argument("-o", "--output", type = str, help = "output directory (if not given in config file)")
parser.add_argument("-c", "--cache", type = str, help = "cache directory (if not given in config file)")
parser.add_argument("-gs", "--generate_script", help = "whether to generate a script", action = 'store_true')
parser.add_argument("-rs", "--random_seed", type = int, help = "random seed for the runs")
parser.add_argument("-em", "--email_address", type = str, help = "email address for the cluster script")


args = parser.parse_args()

config_file = args.config_file

q           = args.q
overwrite   = args.force_overwrite
verbose     = args.verbose
output      = args.output
cache       = args.cache
gen_script  = args.generate_script
random_seed = args.random_seed
email       = args.email_address

p21c.init_runs(config_file, q, overwrite, verbose, output, cache, gen_script, random_seed=random_seed, email_address = email)

