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

N_CORES = 45 #N_EXPS_IN_PARALLEL # HRZ nodes have 96 cores
# MEMORY_SINGLE_JOB = 1000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
MEMORY_PER_CORE = 1200
# PARTITION = 'dgx' # 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'GIGA-classic'

launcher = Launcher(
    exp_name='train_giga_classic_GPG_balanced_no_tab_PACKED',
    exp_file='train_giga', # local path without .py
    # exp_file='/work/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/scripts/generate_data_gpg_parallel', # without .py
    project_name='project02123',  # for hrz cluster
    n_seeds=N_SEEDS,
    # n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=23,
    minutes=59,
    seconds=0,
    # partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)


# Experiment configs (In this case, they are all argparse arguments for the main python file)
launcher.add_experiment(
    net="giga_classic_hr_deeper",
    logdir="/work/scratch/sj93qicy/potato-net/runs",
    dataset="/work/scratch/sj93qicy/potato-net/data/packed/data_packed_train_constructed_4M_GPG_60_randomized_view_no_tab_packked",
    dataset_raw="/work/scratch/sj93qicy/potato-net/data/packed/data_packed_train_random_raw_4M_GPG_60_packked",
    epochs=35,
    batch_size=32,
    num_workers=32,
    epoch_length_frac=0.5,
    description="Classic_GIGA_HR_Deeper_no_tab_PACKED",
    log_wandb=True
    )

launcher.run(LOCAL, TEST)