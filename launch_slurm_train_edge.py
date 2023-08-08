from itertools import product

from experiment_launcher import Launcher#, is_local

LOCAL = False#is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

N_CORES = 5

MEMORY_PER_CORE = 10000
PARTITION = 'rtx2'#, 'rtx'
GRES = 'gpu:1' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'edge_grasp'

launcher = Launcher(
    exp_name='train_EdgeGraspNet_VN_final',
    exp_file='train', # local path without .py    
    # project_name='project01907',  # for hrz cluster
    n_seeds=N_SEEDS,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=1,
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
    dataset_dir='./raw_data_final',
    vn_save_dir='./vn_edge_finaltrain_para',
    test_interval=10,
    save_interval=10,
    epoch=200,
    # net="neu_grasp_pn_deeper",
    # net_with_grasp_occ=True, # Don't pass if not True
    # # logdir="/work/scratch/sj93qicy/potato-net/runs",
    # dataset="/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/pile/data_pile_train_constructed_4M_HighRes_radomized_views",
    # dataset_raw="/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/pile/data_pile_train_random_raw_4M_radomized_views",
    # epochs=35,
    # batch_size=16,
    # num_workers=16,
    # lr=5e-5,
    # epoch_length_frac=0.5,
    # val_split=0.05,
    # description="PN_deeper_DIMS_WITH_OCC",
    # # load_path="/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs/23-04-28-21-10-32_dataset=data_pile_train_constructed_4M_HighRes_radomized_views,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_deeper_DIMS/neural_grasp_neu_grasp_pn_deeper_183258.pt",
    # log_wandb=True
    )

launcher.run(LOCAL, TEST)