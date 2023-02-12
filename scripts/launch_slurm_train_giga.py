from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

# if LOCAL:
#     N_EXPS_IN_PARALLEL = 5
# else:
#     N_EXPS_IN_PARALLEL = 3

N_CORES = 6 #N_EXPS_IN_PARALLEL # HRZ nodes have 96 cores
# MEMORY_SINGLE_JOB = 1000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
MEMORY_PER_CORE = 3500
PARTITION = 'dgx' # 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1'# if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'GIGA-6DoF'

launcher = Launcher(
    exp_name='train_HiRes64_giga',
    exp_file='train_giga', # local path without .py
    # exp_file='/work/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/scripts/generate_data_gpg_parallel', # without .py
    # project_name='project01907',  # for hrz cluster
    n_seeds=N_SEEDS,
    # n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=2,
    hours=0,
    minutes=0,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)


# Experiment configs (In this case, they are all argparse arguments for the main python file)
launcher.add_experiment(
    net="giga_hr",
    dataset="/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/pile/data_pile_train_constructed_4M_HighRes_radomized_views",
    dataset_raw="/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/pile/data_pile_train_random_raw_4M_radomized_views",
    epochs=30,
    batch_size=128,
    )

launcher.run(LOCAL, TEST)