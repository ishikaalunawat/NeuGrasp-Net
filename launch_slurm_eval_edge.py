from itertools import product

from experiment_launcher import Launcher#, is_local

LOCAL = False#is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

N_CORES = 5

MEMORY_PER_CORE = 10000
PARTITION = 'dgx'#'rtx2'#, 'rtx'
GRES = 'gpu:1' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'edge_grasp'

name = 'edge_grasp_net_vn_fixed_view_pile2_45deg'
launcher = Launcher(
    exp_name='eval_'+name,
    exp_file='test_clutter_grasp', # local path without .py    
    # project_name='project01907',  # for hrz cluster
    n_seeds=N_SEEDS,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=6,
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
    scene='pile',
    object_set='pile/test',
    # randomize_view=True, # Don't pass if False
    # tight_view=True, # Don't pass if False
    # self_trained_model=True, # Don't pass if False
    result_path='/home/jauhri/IAS_WS/potato-net/Edge-Grasp-Network/results/'+name,
    )

launcher.run(LOCAL, TEST)