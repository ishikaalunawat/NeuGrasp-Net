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

N_CORES = 12 #N_EXPS_IN_PARALLEL # HRZ nodes have 96 cores
# MEMORY_SINGLE_JOB = 1000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
MEMORY_PER_CORE = 8000
PARTITION = 'rtx2' # 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'GIGA-6DoF'

launcher = Launcher(
    exp_name='train_vgn_PACKED',
    exp_file='train_vgn', # local path without .py
    # exp_file='/work/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/scripts/generate_data_gpg_parallel', # without .py
    # project_name='project02123',  # for hrz cluster
    n_seeds=N_SEEDS,
    # n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)


# Experiment configs (In this case, they are all argparse arguments for the main python file)
launcher.add_experiment(
    # net="giga_classic_hr_deeper",
    # logdir="/work/scratch/sj93qicy/potato-net/runs",
    dataset="/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/packed/vgn_data_packed/data_packed_constructed",
    dataset_raw="/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/packed/vgn_data_packed/data_packed_raw",
    epochs=35,
    # batch_size=32,
    # num_workers=32,
    # epoch_length_frac=0.5,
    description="VGN_PACKED",
    log_wandb=True
    )

launcher.run(LOCAL, TEST)