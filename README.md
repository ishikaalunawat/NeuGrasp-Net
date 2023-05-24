# Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering

[Snehal Jauhri](https://irosalab.com/people/snehal-jauhri/), [Ishikaa Lunawat](), [Georgia Chalvatzaki](https://irosalab.com/people/georgia-chalvatzaki/), 

Submitted to NeurIPS 2023

<!-- [Project](https://sites.google.com/view/rpl-giga2021) | [arxiv](http://arxiv.org/abs/2104.01542)  -->

## Introduction

NeuGraspNet, a novel method for 6DoF grasp detection that leverages recent advances in neural volumetric representations and surface rendering. Our approach learns both global (scene-level) and local (grasp-level) neural surface representations, enabling effective and fully implicit 6DoF grasp quality prediction, even in unseen parts of the scene. Further, we reinterpret grasping as a local neural surface rendering problem, allowing the model to encode the interaction between the robot's end-effector and the object's surface geometry. NeuGraspNet operates on single viewpoints and can sample grasp candidates in occluded scenes, outperforming existing implicit and semi-implicit baseline methods in the literature. We demonstrate the real-world applicability of NeuGraspNet with a mobile manipulator robot, grasping in open spaces with clutter by rendering the scene, reasoning about graspable areas of different objects, and selecting grasps likely to succeed without colliding with the environment.



If you find our work useful in your research, please consider [citing](#citing).

## Installation

1. Create a conda environment.

2. Install packages list in [requirements.txt](requirements.txt). Then install torch related stuff using: `pip install torch==1.7.1 torchvision==0.8.2 pytorch-ignite==0.4.4 torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.7.0+cu11.1.html`
<!-- # torch-scatter` following [here](https://github.com/rusty1s/pytorch_scatter), based on `pytorch` version and `cuda` version. -->

3. Go to the root directory and install the project locally using `pip`

```
pip install -e .
```

4. Build ConvONets dependents by running `python scripts/convonet_setup.py build_ext --inplace`.

5. Download the [data](<TODO>), then unzip and place the data folder under the repo's root. Pretrained models of NeuGraspNet varianss are in `data/models`.

## Data Generation

### 1. Raw simulated grasping trials

Pile scenario:

```bash
python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/pile/data_pile_train_random_raw_4M
```

Packed scenario:
```bash
python scripts/generate_data_parallel.py --scene packed --object-set packed/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/pile/data_packed_train_random_raw_4M
```

Please run `python scripts/generate_data_parallel.py -h` to print all options.

### 2. Data clean and processing

First clean and balance the data using:

```bash
python scripts/clean_balance_data.py /path/to/raw/data
```
### 3. Construct the dataset (add noise):

a. Generate GPG Grasp candidates
```bash
python generate_data_gpg_parallel.py --root path/to/new/partial/raw/data --previous_root path/to/raw/data --use_previous_scenes True --num_proc 96 --grasps_per_scene 60 --grasps_per_scene_gpg 60 --partial_pc True --save_scene True --random True
```
b. Construct dataset
```bash
python scripts/construct_dataset_parallel.py --num-proc 40 --single-view --add-noise dex /path/to/raw/data /path/to/new/constructed/data
```

c. Generate grasp surface clouds
```bash
python generate_data_grasp_surface_clouds.py --raw_root path/to/raw/data --num_proc 96 --save_occ_values True --add_noise True
```

### 4. Save occupancy data

Sampling occupancy data on the fly can be very slow and block the training, so I sample and store the occupancy data in files beforehand:

```bash
python scripts/save_occ_data_parallel.py /path/to/raw/data 100000 2 --num-proc 40
```

Please run `python scripts/save_occ_data_parallel.py -h` to print all options.


## Training
### Train NeuGraspNet (PN. Without Occupancies)
```bash
python scripts/train_neu_grasp.py --net neu_grasp_pn_deeper --dataset /path/to/constructed/data --dataset_raw /path/to/raw/data  --epoch_length_frac 0.5
```

### Train NeuGraspNert (PN. With Occupancies)
```bash
python scripts/train_neu_grasp.py --net neu_grasp_pn_deeper --dataset /path/to/constructed/data --dataset_raw /path/to/raw/data --net_with_grasp_occ True --epoch_length_frac 0.5
```
Please run `python scripts/train_neu_grasp.py -h` to print all options.

## Simulated grasping

```bash
python scripts/sim_grasp_multiple.py --num-view 1 --object-set (packed/test|pile/test) --scene (packed|pile) --num-rounds 20 --add-noise dex --force --best --model path/to/model.pt --resolution=64 --type neu_grasp_pn_deeper --result-path path/to/result --vis

```

This commands will run experiment with each seed specified in the arguments.

Run `python scripts/sim_grasp_multiple.py -h` to print a complete list of optional arguments.


## Visualize reconstructions and TSDFs:
```bash
python generate_visuals.py --model_path path/to/model --root path/to/constructed/data --raw_root path/to/raw/data --random -seed 1

```

## Pre-trained models and pre-generated data

### Pre-trained models

Pretrained models are also in the [data.zip](<TODO>). They are in `data/models`.

### Pre-generated data

As mentioned in the [issue](https://github.com/UT-Austin-RPL/GIGA/issues/3), data generation is very costly. So we upload the generated data. Because the occupancy data takes too much space (over 100G), we do not upload the occupancy data, you can generate them following the instruction in this [section](#save-occupancy-data). This generation won't take too long time.

| Scenario | Raw data | Processed data |
| ----------- | ----------- | ----------- |
| Pile | [link](https://utexas.box.com/s/w1abs6xfe8d2fo0h9k4bxsdgtnvuwprj) | [link](https://utexas.box.com/s/l3zpzlc1p6mtnu7ashiedasl2m3xrtg2) |
| Packed | [link](https://utexas.box.com/s/roaozwxiikr27rgeauxs3gsgpwry7gk7) | [link](https://utexas.box.com/s/h48jfsqq85gt9u5lvb82s5ft6k2hqdcn) |

## Related Repositories

1. Our code is largely based on [VGN](https://github.com/ethz-asl/vgn) and GIGA[https://github.com/UT-Austin-RPL/GIGA]

2. We use [ConvONets](https://github.com/autonomousvision/convolutional_occupancy_networks) and [PointNet] (https://github.com/charlesq34/pointnet) as our backbone.

<!-- ## Citing

```
@article{jiang2021synergies,
 author = {Jiang, Zhenyu and Zhu, Yifeng and Svetlik, Maxwell and Fang, Kuan and Zhu, Yuke},
 journal = {Robotics: science and systems},
 title = {Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations},
 year = {2021}
} -->
```
