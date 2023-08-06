from itertools import product

from experiment_launcher import Launcher#, is_local

LOCAL = False#is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

# if LOCAL:
#     N_EXPS_IN_PARALLEL = 5
# else:
#     N_EXPS_IN_PARALLEL = 3

N_CORES = 5 #N_EXPS_IN_PARALLEL # HRZ nodes have 96 cores
# MEMORY_SINGLE_JOB = 1000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
MEMORY_PER_CORE = 10000
PARTITION = 'gpu'#'dgx' # 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:rtx3090' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'GIGA-6DoF'

name = 'giga_classic_hr_deeper_fixed_view_egad_no_scaling'

launcher = Launcher(
    exp_name='eval_'+name,
    exp_file='sim_grasp_multiple', # local path without .py
    # exp_file='/work/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/scripts/generate_data_gpg_parallel', # without .py
    # project_name='project01907',  # for hrz cluster
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
    num_view=1,
    num_rounds=100,
    resolution=64,

    type='giga_classic_hr_deeper',
    model='/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs_relevant/23-05-11-19-34-37_dataset=data_pile_train_constructed_4M_HighRes_radomized_views_no_table,augment=False,net=6d_giga_classic_hr_deeper,batch_size=32,lr=1e-04,Classic_GIGA_HR_Deeper_no_tab/best_vgn_giga_classic_hr_deeper_val_acc=0.7847.pt',
    # see_table=True, # Don't pass if False
    qual_th=0.5,
    # object_set='pile/test',
    scene='egad',
    # randomize_view=True, # Don't pass if False
    # tight_view=True, # Don't pass if False
    # seen_pc_only=True, # Don't pass if False
    result_path='/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/results/'+name,
    )

launcher.run(LOCAL, TEST)