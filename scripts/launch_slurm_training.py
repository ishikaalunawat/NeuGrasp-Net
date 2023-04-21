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

N_CORES = 48 #N_EXPS_IN_PARALLEL # HRZ nodes have 96 cores
# MEMORY_SINGLE_JOB = 1000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
MEMORY_PER_CORE = 2000
# PARTITION = 'dgx' # 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'GIGA-6DoF'

launcher = Launcher(
    exp_name='train_neural_grasp_pn_NO_TAB_GT_g_cloud_no_local_cloud',
    exp_file='train_neu_grasp', # local path without .py
    # exp_file='/work/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/scripts/generate_data_gpg_parallel', # without .py
    project_name='project01907',  # for hrz cluster
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
    net="neu_grasp_pn_no_local_cloud",
    logdir="/work/scratch/sj93qicy/potato-net/runs",
    dataset="/work/scratch/sj93qicy/potato-net/data/pile/data_pile_train_constructed_4M_HighRes_radomized_views_no_table",
    dataset_raw="/work/scratch/sj93qicy/potato-net/data/pile/data_pile_train_random_raw_4M_GPG_60",
    epochs=35,
    batch_size=128,#64,#16,
    num_workers=43,#55,#10,
    lr=1e-4,
    epoch_length_frac=0.5,
    description="pn_NO_TAB_GT_g_cloud_no_local_cloud",
    # load_path="/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/data/runs/23-04-13-11-17-22_dataset=data_pile_train_constructed_4M_HighRes_radomized_views,augment=False,net=6d_neu_grasp_pn,batch_size=16,lr=1e-04,Continued_PN_with_noisy_g_clouds/best_neural_grasp_neu_grasp_pn_val_acc=0.8960.pt",
    log_wandb=True
    )

launcher.run(LOCAL, TEST)